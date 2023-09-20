import torch as T

from lora import LoRALinear


class TestLoRALinear:

    def test_init(self, in_features=10, out_features=20, rank=2):
        layer = LoRALinear(in_features, out_features, rank)
        assert layer.in_features == layer.linear.in_features
        assert layer.out_features == layer.linear.out_features
        assert layer.rank == rank
        assert len(layer.factors) == 2
        assert layer.factors[0].shape == (in_features, rank)
        assert layer.factors[1].shape == (out_features, rank)

    def test_forward(self):
        original = T.nn.Linear(10, 10)
        linear = LoRALinear.from_linear(original, 10)
        xs = T.arange(6 * 10, dtype=T.float32) \
            .requires_grad_(True) \
            .reshape(2, 3, 10)
        ys = linear(xs)
        assert xs.shape == ys.shape

    def test_backward(self, in_features=4, out_features=5, rank=2):
        layer = LoRALinear(in_features, out_features, rank)
        xs = T.ones((2, 3, 4), requires_grad=True)
        ys = layer(xs)
        ys.backward(T.ones_like(ys))

        # Check gradients on input tensor.
        assert xs.grad is not None

        trainable = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable) == 2

        # Order by the first axis of param shape.
        fst, snd = sorted(trainable, key=lambda p: p.shape[0])
        assert fst.grad is not None
        assert fst.grad.shape == (in_features, rank)
        assert snd.grad is not None
        assert snd.grad.shape == (out_features, rank)
