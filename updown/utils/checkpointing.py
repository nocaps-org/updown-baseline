import copy
import os
from typing import Any, Dict, Optional, Union, Type

import torch
from torch import nn, optim


class CheckpointManager(object):
    r"""
    A :class:`CheckpointManager` periodically serializes models and optimizer as .pth files during
    training, and keeps track of best performing checkpoint based on an observed metric.

    Extended Summary
    ----------------
    It saves state dicts of models and optimizer as ``.pth`` files in a specified directory. This
    class closely follows the API of PyTorch optimizers and learning rate schedulers.

    Notes
    -----
    For :class:`~torch.nn.DataParallel` objects, ``.module.state_dict()`` is called instead of
    ``.state_dict()``.

    Parameters
    ----------
    models: Dict[str, torch.nn.Module]
        Models which need to be serialized as a checkpoint.
    optimizer: torch.optim.Optimizer
        Optimizer which needs to be serialized as a checkpoint.
    serialization_dir: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default="max")
        One of ``min``, ``max``. In ``min`` mode, best checkpoint will be recorded when metric
        hits a lower value; in `max` mode it will be recorded when metric hits a higher value.
    filename_prefix: str, optional (default="checkpoint")
        Prefix of the to-be-saved checkpoint files.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager({"model": model}, optimizer, "/tmp/ckpt", mode="min")
    >>> num_epochs = 20
    >>> for epoch in range(num_epochs):
    ...     train(model)
    ...     val_loss = validate(model)
    ...     ckpt_manager.step(val_loss, epoch)
    """

    def __init__(
        self,
        models: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Type[optim.Optimizer],
        serialization_dir: str,
        mode: str = "max",
        filename_prefix: str = "checkpoint",
    ):

        # Convert single model to a dict.
        if isinstance(models, nn.Module):
            models = {"model": models}

        for key in models:
            if not isinstance(models[key], nn.Module):
                raise TypeError("{} is not a Module".format(type(models).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self._models = models
        self._optimizer = optimizer
        self._serialization_dir = serialization_dir

        self._mode = mode
        self._filename_prefix = filename_prefix

        # Initialize members to hold state dict of best checkpoint and its performance.
        self._best_metric: Optional[Union[float, torch.Tensor]] = None
        self._best_ckpt: Dict[str, Any] = {}

    def step(self, metric: Union[float, torch.Tensor], epoch_or_iteration: int):
        r"""Serialize checkpoint and update best checkpoint based on metric and mode."""

        # Update best checkpoint based on metric and metric mode.
        if not self._best_metric:
            self._best_metric = metric

        models_state_dict: Dict[str, Any] = {}
        for key in self._models:
            if isinstance(self._models[key], nn.DataParallel):
                models_state_dict[key] = self._models[key].module.state_dict()
            else:
                models_state_dict[key] = self._models[key].state_dict()

        if (self._mode == "min" and metric < self._best_metric) or (
            self._mode == "max" and metric > self._best_metric
        ):
            self._best_metric = metric
            self._best_ckpt = copy.copy(models_state_dict)

        # Serialize checkpoint corresponding to current epoch (or iteration).
        torch.save(
            {**models_state_dict, "optimizer": self._optimizer.state_dict()},
            os.path.join(
                self._serialization_dir, f"{self._filename_prefix}_{epoch_or_iteration}.pth"
            ),
        )
        # Serialize best performing checkpoint observed so far.
        torch.save(
            self._best_ckpt,
            os.path.join(self._serialization_dir, f"{self._filename_prefix}_best.pth"),
        )
