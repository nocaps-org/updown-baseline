from typing import Callable, Dict, List, Optional, Tuple

import torch

StepFunctionType = Callable[
    [torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]
]


def _enlarge_single_tensor(t, batch_size, num_fsm_states, beam_size):
    # shape: (batch_size * beam_size, *)
    _, *last_dims = t.size()
    return (
        t.view(batch_size, 1, 1, *last_dims)
        .expand(batch_size, num_fsm_states, beam_size, *last_dims)
        .reshape(-1, *last_dims)
    )


class ConstrainedBeamSearch(object):
    r"""
    Implements Constrained Beam Search for decoding the most likely sequences conditioned on a
    Finite State Machine with specified state transitions.

    .. note::

        We keep the method signatures as close to :class:`~allennlp.nn.beam_search.BeamSearch`
        as possible. Most of the docstring is adapted from AllenNLP, so thanks to them!

    Parameters
    ----------
    end_index : int
        The index of the ``@@BOUNDARY@@`` token in the target vocabulary.
    max_steps : int, optional (default = 20)
        The maximum number of decoding steps to take, i.e. the maximum length of the predicted
        sequences.
    beam_size : int, optional (default = 10)
        The width of the beam used for each "main state" in the Finite State Machine.
    per_node_beam_size : int, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
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

    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: Dict[str, torch.Tensor],
        step: StepFunctionType,
        fsm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state, a step function, and an FSM adjacency matrix, apply Constrained
        Beam Search to find most likely target sequences satisfying specified constraints in FSM.

        .. note::

            If your step function returns ``-inf`` for some log probabilities
            (like if you're using a masked log-softmax) then some of the "best"
            sequences returned may also have ``-inf`` log probability. Specifically
            this happens when the beam size is smaller than the number of actions
            with finite log probability (non-zero probability) returned by the step function.
            Therefore if you're using a mask you may want to check the results from ``search``
            and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : torch.Tensor
            A tensor containing the initial predictions with shape ``(batch_size, )``. These are
            usually just ``@@BOUNDARY@@`` token indices.
        start_state : ``Dict[str, torch.Tensor]``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens, given the
            current state and the predictions from the last time step. The function should accept
            two arguments. The first being a tensor of shape ``(group_size,)``, representing the
            index of the predicted tokens from the last time step, and the second being the
            current state. The ``group_size`` will be ``batch_size * beam_size * num_fsm_states``
            except in the initial step, for which it will just be ``batch_size``. The function is
            expected to return a tuple, where the first element is a tensor of shape
            ``(group_size, vocab_size)`` containing the log probabilities of the tokens for the
            next step, and the second element is the updated state. The tensor in the state should
            have shape ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, num_fsm_states, beam_size, max_steps)``
            and ``log_probabilities`` has shape ``(batch_size, num_fsm_states, beam_size)``.
        """
        # shape: (batch_size, num_fsm_states, num_fsm_states, vocab_size)
        batch_size, num_fsm_states, _, vocab_size = fsm.size()

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. None for
        # the first. Stores the index n for the parent prediction.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop because we are going
        # from a single decoder input (the output from the encoder) to the top `beam_size`
        # decoder outputs per FSM state. On the other hand, within the main loop we are going
        # from the `beam_size` elements of the beam (per FSM state) to `beam_size`^2 candidates
        # from which we will select the top `beam_size` elements for the next iteration.

        # shape: start_class_log_probabilities (batch_size, vocab_size)
        # shape: state["h1"], state["c1"]... etc. (batch_size, hidden_size)
        start_class_log_probabilities, state = step(start_predictions, start_state)
        vocab_size = start_class_log_probabilities.size(-1)

        start_state_predictions = start_class_log_probabilities.view(
            batch_size, 1, vocab_size
        ).expand(batch_size, num_fsm_states, vocab_size)

        start_state_predictions = start_state_predictions.masked_fill(
            1 - fsm[:, 0, :, :], float("-inf")
        )

        # (batch_size, num_fsm_states, beam_size)
        start_top_log_probabilities, start_predicted_classes = start_state_predictions.topk(
            self.beam_size
        )
        # shape: (batch_size, num_fsm_states, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.full((1, vocab_size), float("-inf")).to(
            start_predictions.device
        )
        log_probs_after_end[:, self._end_index] = 0.0

        state = {
            key: _enlarge_single_tensor(value, batch_size, num_fsm_states, self.beam_size)
            for (key, value) in state.items()
        }

        step_state_mask = fsm.view(
            batch_size, num_fsm_states, num_fsm_states, 1, vocab_size
        ).expand(batch_size, num_fsm_states, num_fsm_states, self.beam_size, vocab_size)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size * num_fsm_states, )
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size * num_fsm_states
            )

            if (last_predictions == self._end_index).all():
                break

            class_log_probabilities, state = step(last_predictions, state)
            last_predictions_expanded = (
                last_predictions.view(-1)
                .unsqueeze(-1)
                .expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            )

            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, vocab_size
            )

            restricted_predicted_classes = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_log_probs = torch.FloatTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_indices = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)

            expanded_last_log_probabilities = last_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, 1
            ).expand(batch_size, num_fsm_states, self.beam_size, self.per_node_beam_size)

            for i in range(num_fsm_states):
                # shape (batch_size, num_fsm_states, self.beam_size, vocab_size)
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
                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
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

            backpointer = restricted_beam_indices / self.per_node_beam_size
            backpointers.append(backpointer.view(batch_size, -1))

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, num_fsm_states, -1)

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, num_fsm_states * self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, num_fsm_states * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return (
                    state_tensor.reshape(batch_size, num_fsm_states * self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * num_fsm_states * self.beam_size, *last_dims)
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
        all_predictions = all_predictions.view(batch_size, num_fsm_states, self.beam_size, -1)

        return all_predictions, last_log_probabilities
