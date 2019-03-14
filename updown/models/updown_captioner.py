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
    ):
        super().__init__()

        self._vocabulary = vocabulary
        self._max_caption_length = max_caption_length

        # Short hand notations for convenience
        vocab_size = vocabulary.get_vocab_size()

        # Short hand variable names for convenience.
        self._pad_index = vocabulary.get_token_index("@@PADDING@@")
        self._unk_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self._sos_index = vocabulary.get_token_index("@start@")
        self._eos_index = vocabulary.get_token_index("@end@")

        self._embedding_layer = nn.Embedding(
            vocab_size, embedding_size, padding_idx=self._pad_index
        )

        self._updown_cell = UpDownCell(
            image_feature_size, embedding_size, hidden_size, attention_projection_size
        )

        self._output_layer = nn.Linear(hidden_size, vocab_size)

        self._log_softmax = nn.LogSoftmax(dim=1)

        # We use beam search to find the most likely caption during inference.
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_caption_length, beam_size=beam_size
        )

    def forward(
        self, image_features: torch.FloatTensor, caption_tokens: Optional[torch.LongTensor] = None
    ):

        # Initialize states at zero-th timestep.
        states = None

        # Add "@start@" and "@end@" tokens to caption sequences.
        caption_tokens, _ = add_sentence_boundary_token_ids(
            caption_tokens, (caption_tokens != self._pad_index), self._start_index, self._end_index
        )

        if self.training and caption_tokens is not None:
            batch_size, max_caption_length = caption_tokens.size()

            # shape: (batch_size, max_caption_length)
            tokens_mask = caption_tokens != self._pad_index

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = max_caption_length - 1

            step_logits: List[torch.Tensor] = []
            step_predictions: List[torch.Tensor] = []
            for timestep in range(num_decoding_steps):
                # shape: (batch_size,)
                input_tokens = caption_tokens[:, timestep]

                # shape: (batch_size, num_classes)
                output_logits, states = self._decode_step(image_features, input_tokens, states)

                # list of tensors, shape: (batch_size, 1, vocab_size)
                step_logits.append(output_logits.unsqueeze(1))

                # shape: (batch_size, vocab_size)
                token_probabilities = F.softmax(output_logits, dim=-1)

                # Perform categorical sampling, don't sample @@PADDING@@, @@UNKNOWN@@, @start@.
                token_probabilities[:, self._pad_index] = 0
                token_probabilities[:, self._unk_index] = 0
                token_probabilities[:, self._start_index] = 0

                # shape: (batch_size, )
                predicted_tokens = torch.multinomial(token_probabilities, 1)
                step_predictions.append(predicted_tokens)

            # shape: (batch_size, num_decoding_steps)
            predictions = torch.cat(step_predictions, 1)
            logits = torch.cat(step_logits, 1)

            output_dict = {
                "predictions": predictions,
                "loss": self._get_loss(logits, caption_tokens, tokens_mask),
            }
        else:
            num_decoding_steps = self._max_caption_length

            batch_size = image_features.size(0)
            start_predictions = image_features.new_full((batch_size, ), fill_value=self._start_index)

            # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, beam_size)
            all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, states, self._decode_step)

            # Pick the first beam as predictions.
            output_dict = {
                "predictions": all_top_k_predictions[:, 0, :],
            }

        return output_dict

    def _decode_step(
        self,
        image_features: torch.FloatTensor,
        previous_predictions: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:

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

    @staticmethod
    def _get_loss(
        logits: torch.FloatTensor, targets: torch.LongTensor, target_mask: torch.LongTensor
    ):

        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, average=None
        )
