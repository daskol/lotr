from functools import partial
from typing import Callable, Literal, Sequence

import torch as T

from lotr.lotr import LoTR, LoTRLinear
from lotr.optim import Tucker, tucker2

# Literal types for allowed initialization modes.
CoreInit = Literal['trivial', 'normal', 'svd']
FactorInit = Literal['trivial', 'neutral', 'svd']

LoTRInitializer = Callable[[Sequence[LoTRLinear]], None]


def assign_(ten: T.Tensor, val: T.Tensor, dtype=None, ix=None) -> T.Tensor:
    with T.no_grad():
        # Indexer `ix` is a type of tuple.
        if ix:
            val = val[ix]
        ten.data = val.type(dtype or val.dtype)
        return ten


def make_factor_init(mode: FactorInit, decomp: Tucker, last=False):
    ix = 2 * int(last)  # First (0) and last factors (2).
    match mode:
        case 'trivial':
            return T.nn.init.zeros_
        case 'normal':
            return T.nn.init.normal_
        case 'svd' | 'tucker':
            val = decomp.modes[ix]
            if not last:
                val = val.T
            return partial(assign_, val=val)
        case _:
            raise ValueError(
                f'Unknown initialization mode for a factor {mode}.')


def make_core_init(mode: CoreInit, decomp: Tucker):
    def middle_fn_svd(ix: int, ten: T.Tensor):
        return assign_(ten, decomp.core[:, ix, :])
    match mode:
        case 'trivial':
            return lambda ix, ten: T.nn.init.zeros_(ten)
        case 'neutral':
            return lambda ix, ten: T.nn.init.eye_(ten)
        case 'svd' | 'tucker':
            return middle_fn_svd
        case _:
            raise ValueError(f'Unknown initialization mode for a core {mode}.')


def init_lotr(layers: Sequence[LoTRLinear], left: FactorInit, middle: CoreInit,
              right: FactorInit) -> None:
    # If there is not layer in a sequence then do nothing. ¯\_(ツ)_/¯
    layers = [*layers]
    if not layers:
        return

    # Validate module types.
    rank: int | None = None
    for ix, layer in enumerate(layers):
        if not isinstance(layer, LoTRLinear):
            raise ValueError(
                'Each layer in a sequence must be of LoTRLinear type or '
                f'derived from it but actual type of layer #{ix} '
                f'is {type(layer)}.')

        if rank is None:
            rank = layer.rank
        elif rank != layer.rank:
            raise ValueError(
                'Each layer in a sequence must have the same Tucker rank '
                f'but layer #{ix} has rank {layer.rank} instead of {rank}.')

    # If there is at least one `svd` initializer mode then we have to make SVD
    # decomposition.
    decomp: Tucker | None = None
    if 'svd' in (left, middle, right):
        ten = T.stack([el.linear.weight for el in layers], axis=1)
        decomp = tucker2(ten, rank, axis=1)

    # Create container for a correction.
    layer: LoTRLinear = layers[0]
    dtype = layer.linear.weight.dtype
    device = layer.linear.weight.device
    lotr = LoTR(layer.in_features, layer.out_features, rank, device, dtype)

    # Prepare and apply initializers for left and right factors.
    left_fn = make_factor_init(left, decomp)
    left_fn(lotr.lhs)
    right_fn = make_factor_init(right, decomp, last=True)
    right_fn(lotr.rhs)

    # Prepare initializer for core and apply to core of each LOTR-linear layer.
    middle_fn = make_core_init(middle, decomp)
    for ix, layer in enumerate(layers):
        layer.lotr = LoTR.from_lotr(lotr)
        middle_fn(ix, layer.lotr.mid)


def make_lotr_init(left: FactorInit = 'normal',
                   middle: CoreInit = 'trivial',
                   right: FactorInit = 'normal') -> LoTRInitializer:
    """Construct initializer for a sequence of LoTR affine layers.

    There are several initialization modes. They are

    - trivial -- initialization with zeros;
    - neutral -- initialization with identiy matrix;
    - normal -- draw elements from normal distribution;
    - svd -- initialization with factors of Tucker decomposition.
    """
    return partial(init_lotr, left=left, middle=middle, right=right)
