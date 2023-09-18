from collections import defaultdict
from math import sqrt
from itertools import chain
from typing import Any

import torch as T
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LRScheduler

__all__ = ('LowRankLinear',)


class LowRankLinear(T.nn.Module):
    """A simple representation for linear layer and its small low-rank
    correction.

    At the moment this implementation stores correction as a full matrix with
    out any factors. It is assumed that special low-rank optimizer based on
    general ones (see :py:class:`LowRankAdam` or :py:class:`LowRankSGD`) is
    used for optimization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.linear = T.nn.Linear(in_features, out_features, bias, device,
                                  dtype)
        self.linear.requires_grad_(False)
        self.correction = Parameter(T.empty_like(self.linear.weight))
        self.reset_parameters(False)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def reset_parameters(self, recursively=True) -> None:
        if recursively:
            self.linear.reset_parameters()
        T.nn.init.kaiming_uniform_(self.correction, a=sqrt(5))

    def forward(self, input: T.Tensor) -> T.Tensor:
        assert not self.linear.weight.requires_grad, \
            'Weights of kernel must be freezed.'
        return self.linear(input) + self.scale * (input @ self.correction)

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, **kwargs):
        """Creates linear layer with additive lowr-rank term from linear layer.
        """
        bias = linear.bias is not None
        self = cls(linear.in_features, linear.out_features, bias, **kwargs)
        self.linear = linear.requires_grad_(False)
        return self

    def to_linear(self) -> T.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError


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


class OptimizerList(T.optim.Optimizer):

    def __init__(self, optimizers: list[T.optim.Optimizer]):
        # Do not call __init__ of parent type. We just need to mock the
        # optimizer methods and attributes.
        self.defaults = {}
        self.param_groups = {}
        self.stete = defaultdict(dict)
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
