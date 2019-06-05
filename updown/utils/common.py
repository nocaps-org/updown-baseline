from typing import Dict, Generator

import torch
from torch.utils.data import DataLoader


def cycle(
    dataloader: DataLoader, device: torch.device
) -> Generator[Dict[str, torch.Tensor], None, None]:
    r"""
    A generator which yields a random batch from dataloader perpetually. This generator is
    used in the constructor.

    Extended Summary
    ----------------
    This is done so because we train for a fixed number of iterations, and do not have the
    notion of 'epochs'. Using ``itertools.cycle`` with dataloader is harmful and may cause
    unexpeced memory leaks.
    """
    while True:
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)
            yield batch
