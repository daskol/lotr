from math import sqrt

import torch as T
from torch.nn.parameter import Parameter

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
