from typing import Optional
from contextlib import contextmanager

import torch as T
import torch.nn.functional as F
from enum import Enum
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
                                     adaptive=not self.adaptive)

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
