from functools import partial
from typing import Callable, Tuple, Dict

import torch

from allennlp.nn.beam_search import BeamSearch as AllenNlpBeamSearch


StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[
    [torch.FloatTensor, torch.FloatTensor, StateType], Tuple[torch.Tensor, StateType]
]


class BeamSearch(AllenNlpBeamSearch):
    r"""
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    """

    def __init__(self, end_index: int, max_steps: int = 50, beam_size: int = 10):
        super().__init__(end_index, max_steps, beam_size)

    def search(
        self,
        image_features: torch.FloatTensor,
        start_predictions: torch.LongTensor,
        start_state: StateType,
        step: StepFunctionType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        step_with_features = partial(step, image_features)
        return super().search(start_predictions, start_state, step_with_features)
