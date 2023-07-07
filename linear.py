from contextlib import contextmanager
from enum import Enum
from sys import version_info
from typing import Optional

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import torch as T
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from optimizer_KLS.Linear import Linear as KLSLinear


class DLRTLinear(KLSLinear):
    """Class DLRTLinear is a thin wrapper over original implementation of
    linear layer trained with KLS-scheme.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, rank=None, adaptive=True):
        super().__init__(in_features, out_features, bias, device, dtype, rank,
                         not adaptive)


class Mode(Enum):

    K = 'K-step'

    L = 'L-step'

    S = 'S-step'


@contextmanager
def layer_mode(layer: DLRTLinear, mode: Mode):
    step = layer.step
    layer.step = mode.value[:1]
    yield layer
    layer.step = step


class AdaptiveLoRaLinear(T.nn.Module):
    """Class AdaptiveLoRaLinear implements a factorized analog similar to
    common linear layer. It splits all weights to frozed and factorized
    components which. These components act additively. Factorized component
    represented as $U S V^\\top$ instead of widely-used $U V$ factorization in
    LoRA.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, rank: Optional[int] = None,
                 adaptive: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = Mode.S

        # Initialize main weights (kernel + optinal bias). By default, these
        # weights are not trainable.
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(
            data=T.empty((out_features, in_features), **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = Parameter(T.empty(out_features, **factory_kwargs),
                                  requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize additive low-rank term (kernel only).
        self.adaptive = adaptive
        if self.adaptive:
            max_rank = min([self.in_features, self.out_features]) // 2
            self.rank = min(rank, max_rank) if rank is not None else None
        else:
            self.rank = min(rank, self.in_features, self.out_features)
        self.correction = DLRTLinear(in_features=self.in_features,
                                     out_features=self.out_features,
                                     bias=False,
                                     device=device,
                                     dtype=dtype,
                                     rank=self.rank,
                                     adaptive=self.adaptive)

        # NOTE Since we use DLRT as an addtive low-rank correction term, we
        # need to adjust initialization in such a way that correction does not
        # impact at the beginning.
        self.correction.K.data = T.zeros_like(self.correction.K)
        self.correction.L.data = T.zeros_like(self.correction.L)
        self.correction.S_hat.data = T.zeros_like(self.correction.S_hat)

    def forward(self, xs: T.Tensor, mode: Optional[Mode] = None) -> T.Tensor:
        ys = F.linear(xs, self.weight, self.bias)
        if mode is None:
            mode = self.mode
        with layer_mode(self.correction, mode) as layer:
            return ys + layer(xs)

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, rank: Optional[int] = None,
                    adaptive: bool = True):
        device = linear.weight.device
        dtype = linear.weight.dtype
        module = cls(linear.in_features, linear.out_features,
                     linear.bias is not None, device, dtype, rank, adaptive)
        module.weight.data = linear.weight
        if module.bias is not None:
            module.bias.data = linear.bias
        return module


class LoTR(T.nn.Module):
    """A container for storing common factors and private factors of an
    addivitve term in Tucker-like form.

    >>> layer = LoTR(3, 4, 2)
    >>> clone = LoTR.from_lotr(layer)
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # LoTR acts on a single input `x` as `L M R x` but since neural
        # networks are usually applied to a batched input `X` the layer acts as
        # `X (L M R)^T`. In other words, we store transposed `L`, `M`, and `R`.
        self.lhs = Parameter(
            T.empty((rank, out_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.mid = Parameter(
            T.empty((rank, rank), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.rhs = Parameter(
            T.empty((in_features, rank), device=device, dtype=dtype),
            requires_grad=False,
        )

        # By default, there is not impact on output.
        T.nn.init.normal_(self.lhs)
        T.nn.init.zeros_(self.mid)
        T.nn.init.normal_(self.rhs)

    def __repr__(self) -> str:
        args = ', '.join([
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'rank={self.rank}',
        ])
        return f'{self.__class__.__name__}({args})'

    @property
    def device(self):
        return self.mid.device

    @property
    def dtype(self):
        return self.mid.dtype

    def forward(self, input: T.Tensor) -> T.Tensor:
        hidden = input @ self.rhs
        hidden = hidden @ self.mid
        return hidden @ self.lhs

    @classmethod
    def from_lotr(cls, other: 'LoTR') -> Self:
        """Construct a new container from existing one with left and right
        factors reused.
        """
        self = cls(other.in_features, other.out_features, other.rank,
                   other.device, other.dtype)
        self.lhs = other.lhs
        self.rhs = other.rhs
        return self


class LoTRLinear(T.nn.Module):
    """A variant of linear layer with a trainable additive term in form of
    tucker-like factorization and shared across multiple layer instances.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 1,
                 bias: bool = True, device=None, dtype=None,
                 scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.linear = T.nn.Linear(in_features, out_features, bias, device,
                                  dtype)
        self.lotr = LoTR(in_features, out_features, rank, device, dtype)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def rank(self) -> int:
        return self.lotr.rank

    def forward(self, input: T.Tensor) -> T.Tensor:
        with T.no_grad():
            intermediate = self.linear(input)
        return intermediate + self.scale * self.lotr(input)

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, lotr: 'LoTRLinear', **kwargs):
        """Creates linear layer with additive LoTR term from linear layer and
        LoTR container.

        It clones centeral core of LoTR container and shares left and right
        matrices. Linear layer is not cloned and stays the same with the
        exception that it is excluded from gradient computation graph.
        """
        bias = linear.bias is not None
        self = cls(linear.in_features, linear.out_features, lotr.rank, bias,
                   **kwargs)
        self.linear = linear.requires_grad_(False)
        self.lotr = LoTR.from_lotr(lotr)
        return self

    def to_linear(self) -> T.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError
