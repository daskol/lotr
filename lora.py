from math import sqrt

import torch as T
from torch.nn import Parameter

__all__ = ('LoRALinear', )


class LoRALinear(T.nn.Module):
    r"""A reimplementation of LoRA linear for parameter efficient training.

    .. note::
       In the original paper author initialize the first factor with zeros and
       the second one from Gaussian distribution. We use uniform Kaiming
       initialization for the second factor instead.

    .. note::

       Factorized representation of additive correction has order unlike the
       one reported in the original paper. Factors are stored as list of
       :math:`A`, :math:`B^\top` to make application to input more convenient.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True, device=None, dtype=None,
                 scale: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scale = scale / rank

        opts = {'dtype': dtype, 'device': device}

        # Create frozen linear layer.
        self.linear = T.nn.Linear(in_features, out_features, bias, **opts)
        self.linear.requires_grad_(False)

        # Create trainable factorized coorection.
        self.factors = T.nn.ParameterList([
            Parameter(T.empty((in_features, rank), **opts)),
            Parameter(T.empty((out_features, rank), **opts)),
        ])

        # Initialize only correction.
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
        T.nn.init.kaiming_uniform_(self.factors[0], a=sqrt(5))
        T.nn.init.zeros_(self.factors[1])

    def forward(self, input: T.Tensor) -> T.Tensor:
        assert not self.linear.weight.requires_grad, \
            'Weights of kernel must be freezed.'
        output = self.linear(input)
        correction = (input @ self.factors[0]) @ self.factors[1].T
        return output + self.scale * correction

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, rank: int, **kwargs):
        """Creates linear layer with additive lowr-rank term from linear layer.
        """
        kwargs['bias'] = kwargs.get('bias', linear.bias is not None)
        self = cls(linear.in_features, linear.out_features, rank, **kwargs)
        self.linear = linear.requires_grad_(False)
        return self

    def to_linear(self) -> T.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError
