from typing import Callable, Dict, Optional, Tuple

import torch

StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]


def _enlarge_single_tensor(t, batch_size, num_states, beam_size):
    # shape: (batch_size * beam_size, *)
    _, *last_dims = t.size()
    return (
        t.view(batch_size, 1, 1, *last_dims)
        .expand(batch_size, num_states, beam_size, *last_dims)
        .reshape(-1, *last_dims)
    )


class ConstrainedBeamSearch(object):
    r"""
    Implements Constrained Beam Search for decoding the most likely sequences conditioned on a
    Finite State Machine with specified state transitions.
    """
    def __init__(
        self,
        end_index: int,
        max_steps: int = 20,
        beam_size: int = 5,
        per_node_beam_size: Optional[int] = None,
    ):
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or self.beam_size
        self.init_state = 0

    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: StateType,
        step: StepFunctionType,
        fsm: torch.Tensor,
    ):
        # shape: (batch_size, num_states, num_states, vocab_size)
        batch_size, num_states, _, vocab_size = fsm.size()

        predictions = []
        backpointers = []

        batch_size = start_predictions.size(0)

        start_class_log_probabilities, state = step(start_predictions, start_state)
        num_classes = start_class_log_probabilities.size(-1)

        start_state_predictions = start_class_log_probabilities.view(
            batch_size, 1, num_classes
        ).expand(batch_size, num_states, num_classes)

        start_state_predictions = start_state_predictions.masked_fill(
            1 - fsm[:, self.init_state, :, :], -1e20
        )

        # (batch_size, num_states, beam_size)
        start_top_log_probabilities, start_predicted_classes = start_state_predictions.topk(
            self.beam_size
        )

        # shape: (batch_size, num_states, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.full((1, num_classes), -1e20).to(start_predictions.device)
        log_probs_after_end[:, self._end_index] = 0.0

        state = {
            key: _enlarge_single_tensor(value, batch_size, num_states, self.beam_size)
            for (key, value) in state.items()
        }

        step_state_mask = fsm.view(batch_size, num_states, num_states, 1, num_classes).expand(
            batch_size, num_states, num_states, self.beam_size, num_classes
        )

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size * num_states, )
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size * num_states)

            if (last_predictions == self._end_index).all():
                break

            class_log_probabilities, state = step(last_predictions, state)
            last_predictions_expanded = (
                last_predictions.view(-1)
                .unsqueeze(-1)
                .expand(batch_size * num_states * self.beam_size, num_classes)
            )

            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(
                batch_size, num_states, self.beam_size, num_classes
            )

            restricted_predicted_classes = torch.LongTensor(
                batch_size, num_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_log_probs = torch.FloatTensor(
                batch_size, num_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_indices = torch.LongTensor(batch_size, num_states, self.beam_size).to(
                start_predictions.device
            )

            expanded_last_log_probabilities = last_log_probabilities.view(
                batch_size, num_states, self.beam_size, 1
            ).expand(batch_size, num_states, self.beam_size, self.per_node_beam_size)

            for i in range(num_states):
                # shape (batch_size, num_states, self.beam_size, num_classes)
                state_log_probabilities = cleaned_log_probabilities

                state_log_probabilities = state_log_probabilities.masked_fill(
                    1 - step_state_mask[:, :, i, :, :], -1e20
                )
                top_log_probabilities, predicted_classes = state_log_probabilities.topk(
                    self.per_node_beam_size
                )

                summed_top_log_probabilities = (
                    top_log_probabilities + expanded_last_log_probabilities
                )

                # shape: (batch_size, old_num_states * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_num_states * beam_size * per_node_beam_size)
                reshaped_predicted_classes = predicted_classes.reshape(batch_size, -1)

                # shape (batch_size, beam_size)
                state_beam_log_probs, state_beam_indices = reshaped_summed.topk(self.beam_size)
                # shape (batch_size, beam_size)
                state_predicted_classes = reshaped_predicted_classes.gather(1, state_beam_indices)

                restricted_predicted_classes[:, i, :] = state_predicted_classes
                restricted_beam_indices[:, i, :] = state_beam_indices
                restricted_beam_log_probs[:, i, :] = state_beam_log_probs

            restricted_predicted_classes = restricted_predicted_classes.view(batch_size, -1)

            predictions.append(restricted_predicted_classes)

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, num_states, -1)

            backpointer = restricted_beam_indices / self.per_node_beam_size

            backpointers.append(backpointer.view(batch_size, -1))

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, num_states * self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, num_states * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return (
                    state_tensor.reshape(batch_size, num_states * self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * num_states * self.beam_size, *last_dims)
                )

            state = {key: track_back_state(value) for (key, value) in state.items()}

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_predictions = all_predictions.view(batch_size, num_states, self.beam_size, -1)

        return all_predictions, last_log_probabilities
