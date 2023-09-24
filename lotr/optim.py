"""Module `optim` defines a series routine and optimizers for optimization over
low-dimentsional manifold.

Optimizer :py:class:`LowRankMixin` defines an optimization step for a single
matrix-like parameter. It is intended to use with LoRA-like adapters.

Optimizer :py:class:`LoTRMixin` defines an optimization step for a group of
matrix-like parameters which are projected on low-dimentsional manifold with
small tensor rank (aka Tucker decomposition). It is intended to use with LoTR
layers.

All mixins are used behind generic optimizers like :py:`torch.optim.Adam` as a
post-step. In order to use them in common pipelines, for example with
:py:'transformers.Trainer`, one should combine multiple optimizers and
schedulers with :py:class:`OptimizerList` and :py:class:`LRSchedulerList`.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable

try:
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:
    from numpy.core.multiarray import normalize_axis_index

import torch as T
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LRScheduler

__all__ = ('LRSchedulerList', 'LowRankAdam', 'LowRankAdamW', 'LowRankMixin',
           'LowRankSGD', 'OptimizerList')


class LowRankMixin(T.optim.Optimizer):
    """A mix-in type to project weights on low-rank matrix manifold as a
    post-step of a generic optimizer.

    .. code-block:: python

       layer = T.nn.Linear(emb_size, emb_size)
       layer.bias.requires_grad_(False)  # Not a matrix.
       opt = LowRankAdam(layer.parameters(), rank=2)
       ...
       opt.step()
       opt.zero_grad()
    """

    def __init__(self, params, *args, rank=1, **kwargs):
        if not isinstance(rank, int):
            raise ValueError(f'Rank `rank` must be of int type: {type(rank)}.')
        if rank <= 0:
            raise ValueError(f'Rank `rank` must be a positive: {rank}.')

        # TODO: Recursively sanitize parameter shapes.

        # Forward object construction to the next class in MRO chain. In this
        # way we can construct a reference optimizer (e.g. torch.optim.Adam or
        # torch.optim.SGD).
        super().__init__(params, *args, **kwargs)

        # As soon as base optimizer is constructed, we should add rank
        # hyperparameter to each param group as a defaults in order to be
        # consistent with semantic of torch.optim.Optimizer.
        for group in self.param_groups:
            group.setdefault('rank', rank)

    @T.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            rank = group['rank']
            for param in group['params']:
                self.project(param, rank)
        return loss

    def project(self, param: Parameter, rank: int):
        """Project every matrix on low-dimentsional manifold of a specified
        rank `rank`.
        """
        if param.grad is None:
            return

        assert param.ndim == 2, \
            f'Only matrix weights are supported: shape={param.shape}.'

        # Access to low-rank representation of the weight matrix.
        state = self.state[param]

        if (factors := state.get('low-rank')) is None:
            # Create factors with truncated SVD decomposition of given rank.
            lhs, svals, rhs = T.svd_lowrank(param, rank)
            scaler = T.sqrt(svals)
            lhs *= scaler
            rhs *= scaler
        else:
            # Common step consists of the following operations.
            #
            #   U_new = QR(\delta W_new V)
            #   V_new = \delta W_new^top U_new
            lhs, _ = T.linalg.qr(param @ factors[1])
            rhs = param.T @ lhs

        # Update factors in param state.
        state['low-rank'] = (lhs, rhs)

        # Restore weight matrix from factors.
        param = lhs * rhs


class LowRankAdam(LowRankMixin, T.optim.Adam):
    """A variant of Adam optimizer with an additional step for projection on
    low-rank matrix manifold.
    """

    def __init__(self, params, *, rank=1, **kwargs):
        super().__init__(params, rank=rank, **kwargs)


class LowRankAdamW(LowRankMixin, T.optim.AdamW):
    """A variant of AdamW optimizer with an additional step for projection on
    low-rank matrix manifold.
    """

    def __init__(self, params, *, rank=1, **kwargs):
        super().__init__(params, rank=rank, **kwargs)


class LowRankSGD(LowRankMixin, T.optim.SGD):
    """A variant of SGD optimizer with an additional step for projection on
    low-rank matrix manifold.
    """

    def __init__(self, params, *args, rank=1, **kwargs):
        super().__init__(params, *args, rank=rank, **kwargs)


@dataclass
class Tucker:
    """A simple dataclass to represent Tucker and Tucker2 decompositions.

    In case of Tucker2 decomposition, an instance of this type has non-trivial
    `axis` and `size` attributes which refer to identity mode and its size.
    """

    rank: tuple[int, ...]

    core: T.Tensor

    modes: tuple[T.Tensor, ...]

    axis: None | int = None

    size: None | int = None

    def __post_init__(self):
        if self.axis is not None:
            if self.size is None:
                raise ValueError('Both `axis` and `size` must be given '
                                 'simulteneously.')

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, ...]:
        shape = []
        for it, mode in enumerate(self.modes):
            if it == self.axis:
                shape.append(self.size)
            else:
                shape.append(mode.shape[0])
        return tuple(shape)


def tucker2(arr: T.Tensor, rank: int | tuple[int, ...], axis=-1) -> Tucker:
    """A native implementation of Tucker2 algorithm for building corresponding
    tensor decomposition.
    """
    assert arr.ndim == 3, 'Only 3-tensors are accepted.'
    axis = normalize_axis_index(axis, 3)
    if isinstance(rank, int):
        ranks = (rank, ) * 3
    elif isinstance(rank, tuple):
        ranks = rank
    ranks = ranks[:axis] + (arr.shape[axis], ) + ranks[axis + 1:]

    # Build mode matricies iteratively.
    modes = []
    for it, size, rank in zip(range(3), arr.shape, ranks):
        # Along axis `axis` identity core is assumed (skip).
        if it == axis:
            modes.append(None)
            continue
        ten: T.Tensor = T.moveaxis(arr, it, 0)
        mat = ten.reshape(size, -1)
        mode, *_ = T.svd_lowrank(mat, rank)
        modes.append(mode)

    # Build a contraction operation to compute core tensor.
    operands = [arr, (0, 1, 2)]
    output_ix = []
    for it, mode in zip(range(3), modes):
        if mode is None:
            # If there is not mode to contract with then add current axis to
            # the spec of output.
            output_ix.append(it)
        else:
            # Calculate output index which should be greater then 2 (i.e. 3 or
            # 4) and it can be computed out of number of operands.
            jt = 2 + len(operands) // 2
            operands.extend([mode, (it, jt)])
            output_ix.append(jt)
    core = T.einsum(*operands, output_ix)

    # Collate factors and cores to a single output type.
    return Tucker(ranks, core, tuple(modes), axis, arr.shape[axis])


def filter_trainable(params: Iterable, trainable=True):
    """Filter trainable parameters out of params or param group.

    Args:
        trainable: filter trainable or non-trainable.
    """

    def pred(param: T.Tensor):
        return ((not trainable and not param.requires_grad)
                or (trainable and param.requires_grad))

    def filter_out(params: Iterable):
        return [param for param in params if pred(param)]

    if isinstance(params, dict):
        param_group = params
        param_group['params'] = filter_out(param_group['params'])
        return param_group
    else:
        return filter_out(params)


def sanitize_param_group(params: Iterable) -> dict[str, list[Any]]:
    """Check that parameters in parameter group has the same shape."""
    if isinstance(params, dict):
        param_group = params
    else:
        param_group = {'params': list(params)}

    params = [*param_group['params']]
    if len(params) == 0:
        return param_group

    shape = params[0].shape
    if any(shape != param.shape for param in params):
        raise ValueError(f'All params must have the same shape: {shape}.')

    return param_group


class LoTRMixin(LowRankMixin):
    """A mixin type to make projection step onto low-rank tensor manifold after
    gradient step.

    This mixin type is designed in a way to be composable with any generic
    optimizer like `toch.optim.Adam`. Composition is done through subclassing
    this mixing `LoTRMixin` and an optimizer type. Meanwhile, `LoTRMixin` has
    to precede to an optimizer since the transfer gradient computation and
    application to the next type in MRO hierarchy.

    .. code-block:: python

       # In case of model with regular structure, one should collect all
       # similar weights (weight matriced of linear layers as an example) to a
       # single parameter list of parameter group and pass it to optimizer on
       # construction or add it later with `add_param_group` method of
       # `torch.optim.Optizer` class.
       model = T.nn.Sequential(T.nn.Linear(emb_size, emb_size),
                               T.nn.Linear(emb_size, emb_size))
       params = [linear.weight for linear in model]

       opt = LoTRAdam(params, rank=2)
       ...
       opt.step()
       opt.zero_grad()
    """

    def add_param_group(self, param_group):
        param_group = sanitize_param_group(param_group)
        return super().add_param_group(param_group)

    @T.no_grad()
    def step(self, closure=None):
        # Call parent method of parent type.
        loss = super(LowRankMixin, self).step(closure)
        for group in self.param_groups:
            # All parameters in group are required for projection on low-rank.
            params = group['params']
            # If all one parameters have gradients then apply projection
            # on low-rank manifold to entire parameter group.
            if all([param.grad is not None for param in params]):
                self.project(params, group['rank'])
        return loss

    def project(self, params: list[Parameter], rank: int):
        """Project every matrix on low-dimentsional manifold of a specified
        rank `rank`.
        """
        # Build up a 3-tensor from parameter list and apply factorization.
        # Group state are stored at the state of the first parameter.
        state = self.state[params[0]]
        ten = T.stack([param.data for param in params], axis=1)
        rows, depth, cols = ten.shape

        # Obtain shared factors U, V used in decomposition δW = U S V^top.
        #
        # TODO(@bershatsky): Find out the best axis from computational
        # efficacy perspetctive (axis=0).
        if (decomp := state.get('decomp')) is None:
            # If there is not any factors then initialize them.
            decomp = tucker2(ten, rank, axis=1)
        else:
            # Update factors like (Tucker2 ALS step.
            mat = T.einsum('ijk,kc->ijc', decomp.modes[2]).reshape(rows, -1)
            mode0, *_ = T.svd_lowrank(mat, rank)
            mat = T.einsum('ijk,ia->kaj', mode0).reshape(cols, -1)
            mode2, *_ = T.svd_lowrank(mat, rank)
            decomp.modes = (mode0, None, mode2)

        # Update Tucker factors of parameter group.
        state['decomp'] = decomp

        # Apply projection procedure with shared factors and given rank.
        modes = (decomp.modes[0], decomp.modes[2])
        core = T.einsum('ijk,ia,kc->ajc', ten, *modes)
        for ix, _ in enumerate(params):
            params[ix] = T.einsum('ac,ia,jc->ij', core[:, ix, :], *modes)


class LoTRAdam(LoTRMixin, T.optim.Adam):
    """A variant of Adam optimizer with an additional step for projection on
    low-rank tensor manifold for use with LoTR adapter.
    """

    def __init__(self, params, *, rank=1, **kwargs):
        super().__init__(params, rank=rank, **kwargs)


class LoTRAdamW(LoTRMixin, T.optim.AdamW):
    """A variant of AdamE optimizer with an additional step for projection on
    low-rank tensor manifold for use with LoTR adapter.
    """

    def __init__(self, params, *, rank=1, **kwargs):
        super().__init__(params, rank=rank, **kwargs)


class LoTRSGD(LoTRMixin, T.optim.SGD):
    """A variant of SGD optimizer with an additional step for projection on
    low-rank tensor manifold for use with LoTR adapter.
    """

    def __init__(self, params, *, rank=1, **kwargs):
        super().__init__(params, rank=rank, **kwargs)


class OptimizerList(T.optim.Optimizer):

    def __init__(self, optimizers: list[T.optim.Optimizer]):
        # Do not call __init__ of parent type. We just need to mock the
        # optimizer methods and attributes.
        self.defaults = {}
        self.param_groups = {}
        self.state = defaultdict(dict)
        self.optimizers = list(optimizers)

    def step(self, closure=None):
        assert closure is None, 'Not implemented suppor for closure.'
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state: list[Any]):
        for opt, opt_state in zip(self.optimizers, state):
            opt.load_state_dict(opt_state)

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none)


class LRSchedulerList(LRScheduler):

    def __init__(self, schedulers: list[LRScheduler]):
        self.schedulers = schedulers

    def get_last_lr(self):
        lrs = chain.from_iterable(scheduler.get_last_lr()
                                  for scheduler in self.schedulers)
        return [*lrs]

    def step(self, epoch=None):
        for scheduler in self.schedulers:
            scheduler.step(epoch)
