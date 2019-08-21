import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERY_NEGATIVE_NUMBER = -1e20

def enlarge_single_tensor(t, batch_size, state_size, beam_size):
    # shape: (batch_size * beam_size, *)
    _, *last_dims = t.size()
    return t.view(batch_size, 1, 1, *last_dims) \
            .expand(batch_size, state_size, beam_size, *last_dims) \
            .reshape(-1, *last_dims)

class ConstraintBeamSearch(nn.Module):
    def __init__(self, end_index, max_steps, beam_size, per_node_beam_size=None):
        super(ConstraintBeamSearch, self).__init__()
        self._end_index = end_index
        self.init_state = 0
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or self.beam_size

        self.select_state_func = None
        self.early_stop = True
        
        self.VERY_NEGATIVE_TENSOR = torch.FloatTensor([VERY_NEGATIVE_NUMBER])

    def update_parameter(self, select_state_func, early_stop=True):
        self.select_state_func = select_state_func
        self.early_stop = early_stop

    def search(self, step_func, image_feature, start_predictions, start_state, state_transform, image_ids):
        assert self.select_state_func is not None, "accept state function should be set"
        assert state_transform.size(1) == state_transform.size(2)
        self.state_size = state_transform.size(1)

        device_id = start_predictions.get_device()
        self.VERY_NEGATIVE_TENSOR = self.VERY_NEGATIVE_TENSOR.to(device_id)
        predictions = []
        backpointers = []

        batch_size = start_predictions.size(0)

        start_class_log_probabilities, state = step_func(image_feature, start_predictions, start_state)
        self.num_classes = start_class_log_probabilities.size(-1)
        state_prediction = start_class_log_probabilities \
                                .view(batch_size, 1, self.num_classes) \
                                .expand(batch_size, self.state_size, self.num_classes)
        
        state_prediction = torch.where(
            state_transform[:, self.init_state, :, :],
            state_prediction,
            self.VERY_NEGATIVE_TENSOR
        )

        # (batch_size, state_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = state_prediction.topk(self.beam_size)

        # shape: (batch_size, state_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.FloatTensor(1, self.num_classes).fill_(VERY_NEGATIVE_NUMBER).to(device_id)
        log_probs_after_end[:, self._end_index] = 0.

        state = {key: enlarge_single_tensor(value, batch_size, self.state_size, self.beam_size) for (key, value) in state.items()}
        image_feature = enlarge_single_tensor(image_feature, batch_size, self.state_size, self.beam_size)

        step_state_mask = state_transform \
                                .view(batch_size, self.state_size, self.state_size, 1, self.num_classes) \
                                .expand(batch_size, self.state_size, self.state_size, self.beam_size, self.num_classes)
        
        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(-1)

            if (last_predictions == self._end_index).all() and self.early_stop:
                break

            class_log_probabilities, state = step_func(image_feature, last_predictions, state)
            last_predictions_expanded = last_predictions.view(-1).unsqueeze(-1)\
                                                        .expand(batch_size * self.state_size * self.beam_size, self.num_classes)

            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(batch_size, self.state_size, self.beam_size, self.num_classes)

            restricted_predicted_classes = torch.LongTensor(batch_size, self.state_size, self.beam_size).to(device_id)
            restricted_beam_log_probs = torch.FloatTensor(batch_size, self.state_size, self.beam_size).to(device_id)
            restricted_beam_indices = torch.LongTensor(batch_size, self.state_size, self.beam_size).to(device_id)

            expanded_last_log_probabilities = last_log_probabilities \
                        .view(batch_size, self.state_size, self.beam_size, 1) \
                        .expand(batch_size, self.state_size, self.beam_size, self.per_node_beam_size)

            for i in range(self.state_size):
                # shape (batch_size, self.state_size, self.beam_size, self.num_classes)
                state_log_probabilities = cleaned_log_probabilities

                state_log_probabilities = torch.where(
                    step_state_mask[:, :, i, :, :],
                    state_log_probabilities,
                    self.VERY_NEGATIVE_TENSOR
                )

                top_log_probabilities, predicted_classes = \
                    state_log_probabilities.topk(self.per_node_beam_size)

                summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

                # shape: (batch_size, old_state_size * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_state_size * beam_size * per_node_beam_size)
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

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, self.state_size, -1)

            backpointer = restricted_beam_indices / self.per_node_beam_size

            backpointers.append(backpointer.view(batch_size, -1))

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                        view(batch_size, self.state_size * self.beam_size, *([1] * len(last_dims))).\
                        expand(batch_size, self.state_size * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return state_tensor.\
                          reshape(batch_size, self.state_size * self.beam_size, *last_dims).\
                          gather(1, expanded_backpointer).\
                          reshape(batch_size * self.state_size * self.beam_size, *last_dims)

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
        all_predictions = all_predictions.view(batch_size, self.state_size, self.beam_size, -1)

        return self.select_state_func(all_predictions, last_log_probabilities, image_ids)
