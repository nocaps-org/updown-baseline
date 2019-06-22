import functools
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits

from updown.modules.updown_cell import UpDownCell


class UpDownCaptioner(nn.Module):
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
        self._max_caption_length = max_caption_length

        # Short hand notations for convenience
        vocab_size = vocabulary.get_vocab_size()

        # Short hand variable names for convenience.
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

        self._caption_loss = nn.CrossEntropyLoss(ignore_index=self._pad_index)

        # We use beam search to find the most likely caption during inference.
        self._beam_size = beam_size
        self._beam_search = BeamSearch(
            self._boundary_index,
            max_steps=max_caption_length,
            beam_size=beam_size,
            per_node_beam_size=beam_size // 2,
        )

    def forward(
        self, image_features: torch.FloatTensor, caption_tokens: Optional[torch.LongTensor] = None
    ):

        batch_size = image_features.size(0)

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
        image_features: torch.FloatTensor,
        previous_predictions: torch.LongTensor,
        states: Optional[Dict[str, torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:

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
        self, logits: torch.FloatTensor, targets: torch.LongTensor, target_mask: torch.LongTensor
    ):

        # shape: (batch_size, )
        target_lengths = torch.sum(target_mask, dim=-1).float()

        # Multiply (length normalized) negative logprobs of the sequence with its length.
        # shape: (batch_size, )
        return target_lengths * sequence_cross_entropy_with_logits(
            logits, targets, target_mask, average=None
        )

        # return (
        #     self._caption_loss(
        #         logits.view(-1, self._vocabulary.get_vocab_size()), targets.view(-1)
        #     )
        #     * self._max_caption_length
        # )
