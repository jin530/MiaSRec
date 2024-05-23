# the source is copied from TorchRua
# https://github.com/speedcell4/torchrua
import torch

from torch import Tensor
from torch.types import Device
from torch.nn.utils.rnn import PackedSequence

from typing import Tuple, Optional


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    acc_sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    acc_sizes[0] = 0
    return acc_sizes

@torch.no_grad()
def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = torch.repeat_interleave(repeats=sizes)

    major_ptr = torch.repeat_interleave(accumulate_sizes(sizes), repeats=sizes)
    major_ptr = torch.arange(major_ptr.size()[0], device=major_ptr.device) - major_ptr

    return major_ptr, minor_ptr

@torch.no_grad()
def reverse_packed_indices(batch_sizes: Tensor, device: Device = None) -> Tensor:
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    _, token_sizes = torch.unique(batch_ptr, sorted=True, return_counts=True)
    token_ptr = token_sizes[batch_ptr] - token_ptr - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(sequence: PackedSequence) -> PackedSequence:
    indices = reverse_packed_indices(batch_sizes=sequence.batch_sizes, device=sequence.data.device)

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )