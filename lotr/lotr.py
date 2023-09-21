from sys import version_info

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import torch as T
from torch.nn.parameter import Parameter

__all__ = ('LoTR', 'LoTRLinear')


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

    Almost all arguments of the :py:`LoTRLinear` layer corresponds to vanila
    :py:`torch.nn.Linear` counterpart except `rank` and `scale` parameters.

    Args:
      rank: rank of LoTR addvive term.
      scale: a factor before LoTR term which is similar to alpha in LoRA.

    In example below we load pretrained RoBERTa model with all weights freezed.
    We unfreeze classification head. Then we create a shared LoTR container
    which is used later in the loop to substitue default :py:`torch.nn.Linear`
    layers with :py:`LoTRLinear` variant.

    .. code-block:: python

       from lotr import LoTR, LoTRLinear
       from transformers import AutoModelForSequenceClassification

       model = AutoModelForSequenceClassification \
           .from_pretrained('roberta-base') \
           .requires_grad_(False)
       model.classifier.requires_grad_(True)

       # number of train parameters: 592130
       print('number of train parameters:', model.num_parameters(True))

       lotr = LoTR(model.config.hidden_size, model.config.hidden_size, rank=2)
       for layer in model.roberta.encoder.layer:
           layer.attention.self.query = LoTRLinear.from_linear(
               linear=layer.attention.self.query,
               lotr=lotr,
               scale=1.0,
           )
           layer.attention.self.value = LoTRLinear.from_linear(
               linear=layer.attention.self.value,
               lotr=lotr,
               scale=1.0,
           )

       # number of train parameters: 595298
       # total number of parameters: 124650338
       print('number of train parameters:', model.num_parameters(True))
       print('total number of parameters:', model.num_parameters())
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 1,
                 bias: bool = True, device=None, dtype=None,
                 scale: float = 1.0):
        super().__init__()
        self.scale = scale

        opts = {'device': device, 'dtype': dtype}
        self.linear = T.nn.Linear(in_features, out_features, bias, **opts)
        self.linear.weight.requires_grad_(False)
        self.lotr = LoTR(in_features, out_features, rank, **opts)

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
        assert not self.linear.weight.requires_grad, \
            'Weights of kernel must be freezed.'
        return self.linear(input) + self.scale * self.lotr(input)

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, lotr: 'LoTRLinear', **kwargs):
        """Creates linear layer with additive LoTR term from linear layer and
        LoTR container.

        It clones centeral core of LoTR container and shares left and right
        matrices. Linear layer is not cloned and stays the same with the
        exception that it is excluded from gradient computation graph. Original
        linear layer is freezed while LoTR term is forced to require gradients.
        """
        bias = linear.bias is not None
        self = cls(linear.in_features, linear.out_features, lotr.rank, bias,
                   **kwargs)
        self.linear = linear.requires_grad_(False)
        self.lotr = LoTR.from_lotr(lotr).requires_grad_()
        return self

    def to_linear(self) -> T.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError
