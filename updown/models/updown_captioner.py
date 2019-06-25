import functools
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits

from updown.modules import UpDownCell


class UpDownCaptioner(nn.Module):
    r"""
    Image captioning model using bottom-up top-down attention, as in
    `Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`. At training time, this model
    maximizes the likelihood of ground truth caption, given image features. At inference time,
    given image features, captions are decoded using beam search.

    Extended Summary
    ----------------
    This captioner is basically a recurrent language model for caption sequences. Internally, it
    runs :class:`~updown.modules.updown_cell.UpDownCell` for multiple time-steps. If this class is
    analogous to an :class:`~torch.nn.LSTM`, then :class:`~updown.modules.updown_cell.UpDownCell`
    would be analogous to :class:`~torch.nn.LSTMCell`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_feature_size: int
        Size of the bottom-up image features.
    embedding_size: int
        Size of the word embedding input to the captioner.
    hidden_size: int
        Size of the hidden / cell states of attention LSTM and language LSTM of the captioner.
    attention_projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    max_caption_length: int, optional (default = 20)
        Maximum length of caption sequences for language modeling. Captions longer than this will
        be truncated to maximum length.
    beam_size: int, optional (default = 1)
        Beam size for finding the most likely caption during decoding time (evaluation).
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_feature_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_projection_size: int,
        max_caption_length: int = 20,
        beam_size: int = 1,
    ) -> None:
        super().__init__()
        self._vocabulary = vocabulary

        self.image_feature_size = image_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_projection_size = attention_projection_size

        # Short hand variable names for convenience
        vocab_size = vocabulary.get_vocab_size()
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self._boundary_index = vocabulary.get_token_index("@@BOUNDARY@@")

        self._embedding_layer = nn.Embedding(
            vocab_size, embedding_size, padding_idx=self._pad_index
        )

        self._updown_cell = UpDownCell(
            image_feature_size, embedding_size, hidden_size, attention_projection_size
        )

        self._output_layer = nn.Linear(hidden_size, vocab_size)
        self._log_softmax = nn.LogSoftmax(dim=1)

        # We use beam search to find the most likely caption during inference.
        self._beam_size = beam_size
        self._beam_search = BeamSearch(
            self._boundary_index,
            max_steps=max_caption_length,
            beam_size=beam_size,
            per_node_beam_size=beam_size // 2,
        )
        self._max_caption_length = max_caption_length

    def forward(
        self, image_features: torch.Tensor, caption_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        r"""
        Given bottom-up image features, maximize the likelihood of paired captions during
        training. During evaluation, decode captions given image features using beam search.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes * image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        caption_tokens: torch.Tensor, optional (default = None)
            A tensor of shape ``(batch_size, max_caption_length)`` of tokenized captions. This
            tensor does not contain ``@@BOUNDARY@@`` tokens yet. Captions are not provided
            during evaluation.

        Returns
        -------
        Dict[str, torch.Tensor]
            Decoded captions and/or per-instance cross entropy loss, dict with keys either
            ``{"predictions"}`` or ``{"loss"}``.
        """

        # shape: (batch_size, num_boxes * image_feature_size) for adaptive features.
        # shape: (batch_size, num_boxes, image_feature_size) for fixed features.
        batch_size = image_features.size(0)

        # shape: (batch_size, num_boxes, image_feature_size)
        image_features = image_features.view(batch_size, -1, self.image_feature_size)

        # Initialize states at zero-th timestep.
        states = None

        if self.training and caption_tokens is not None:
            # Add "@@BOUNDARY@@" tokens to caption sequences.
            caption_tokens, _ = add_sentence_boundary_token_ids(
                caption_tokens,
                (caption_tokens != self._pad_index),
                self._boundary_index,
                self._boundary_index,
            )

            _, max_caption_length = caption_tokens.size()

            # shape: (batch_size, max_caption_length)
            tokens_mask = caption_tokens != self._pad_index

            # The last input from the target is either padding or the boundary token.
            # Either way, we don't have to process it.
            num_decoding_steps = max_caption_length - 1

            step_logits: List[torch.Tensor] = []
            for timestep in range(num_decoding_steps):
                # shape: (batch_size,)
                input_tokens = caption_tokens[:, timestep]

                # shape: (batch_size, num_classes)
                output_logits, states = self._decode_step(image_features, input_tokens, states)

                # list of tensors, shape: (batch_size, 1, vocab_size)
                step_logits.append(output_logits.unsqueeze(1))

            # shape: (batch_size, num_decoding_steps)
            logits = torch.cat(step_logits, 1)

            # Skip first time-step from targets for calculating loss.
            output_dict = {
                "loss": self._get_loss(
                    logits, caption_tokens[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous()
                )
            }
        else:
            num_decoding_steps = self._max_caption_length

            start_predictions = image_features.new_full(
                (batch_size,), fill_value=self._boundary_index
            ).long()

            # Add image features as a default argument to match callable signature acceptable by
            # beam search class (previous predictions and states only).
            beam_decode_step = functools.partial(self._decode_step, image_features)

            # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, beam_size)
            all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, states, beam_decode_step
            )

            # Pick the first beam as predictions.
            best_predictions = all_top_k_predictions[:, 0, :]
            output_dict = {"predictions": best_predictions}

        return output_dict

    def _decode_step(
        self,
        image_features: torch.Tensor,
        previous_predictions: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Given image features, tokens predicted at previous time-step and LSTM states of the
        :class:`~updown.modules.updown_cell.UpDownCell`, take a decoding step. This is also
        called by the beam search class.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``.
        previous_predictions: torch.Tensor
            A tensor of shape ``(batch_size, )`` containing tokens predicted at previous
            time-step -- one for each instances in a batch.
        states: [Dict[str, torch.Tensor], optional (default = None)
            LSTM states of the :class:`~updown.modules.updown_cell.UpDownCell`. These are
            initialized as zero tensors if not provided (at first time-step).
        """

        # Expand and repeat image features while doing beam search (during inference).
        if not self.training and image_features.size(0) != previous_predictions.size(0):

            # Add beam dimension and repeat image features.
            image_features = image_features.unsqueeze(1).repeat(1, self._beam_size, 1, 1)
            batch_size, beam_size, num_boxes, image_feature_size = image_features.size()

            # shape: (batch_size * beam_size, num_boxes, image_feature_size)
            image_features = image_features.view(
                batch_size * beam_size, num_boxes, image_feature_size
            )

        # shape: (batch_size, )
        current_input = previous_predictions

        # shape: (batch_size, embedding_size)
        token_embeddings = self._embedding_layer(current_input)

        # shape: (batch_size, hidden_size)
        updown_output, states = self._updown_cell(image_features, token_embeddings, states)

        # shape: (batch_size, vocab_size)
        output_logits = self._output_layer(updown_output)
        output_class_logprobs = self._log_softmax(output_logits)

        # Return logits while training, to further calculate cross entropy loss.
        # Return logprobs during inference, because beam search needs them.
        # Note:: This means NO BEAM SEARCH DURING TRAINING.
        outputs = output_logits if self.training else output_class_logprobs

        return outputs, states  # type: ignore

    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute cross entropy loss of predicted caption (logits) w.r.t. target caption. The cross
        entropy loss of caption is cross entropy loss at each time-step, summed.

        Parameters
        ----------
        logits: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1, vocab_size)`` containing
            unnormalized log-probabilities of predicted captions.
        targets: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1)`` of tokenized target
            captions.
        target_mask: torch.Tensor
            A mask over target captions, elements where mask is zero are ignored from loss
            computation. Here, we ignore ``@@UNKNOWN@@`` token (and hence padding tokens too
            because they are basically the same).

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, )`` containing cross entropy loss of captions, summed
            across time-steps.
        """

        # shape: (batch_size, )
        target_lengths = torch.sum(target_mask, dim=-1).float()

        # Multiply (length normalized) negative logprobs of the sequence with its length.
        # shape: (batch_size, )
        return target_lengths * sequence_cross_entropy_with_logits(
            logits, targets, target_mask, average=None
        )
