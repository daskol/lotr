import torch as T

from linear import AdaptiveLoRaLinear, LoTR, Mode, LoTRLinear


def test_linear():
    inp = T.ones((32, 8))
    module = AdaptiveLoRaLinear(8, 8)
    module(inp, Mode.K)
    module(inp, Mode.L)
    module(inp, Mode.S)


class TestLoTR:

    def test_init(self):
        layer = LoTR(3, 4, 2)
        assert layer.in_features == 3
        assert layer.out_features == 4
        assert layer.rank == 2
        assert len([*layer.parameters()]) == 3

    def test_from_lotr(self):
        layer = LoTR(3, 4, 2)
        clone = LoTR.from_lotr(layer)
        assert clone.in_features == 3
        assert clone.out_features == 4
        assert clone.rank == 2
        assert len([*clone.parameters()]) == 3
        assert clone.lhs.data_ptr() == layer.lhs.data_ptr()
        assert clone.mid.data_ptr() != layer.mid.data_ptr()
        assert clone.rhs.data_ptr() == layer.rhs.data_ptr()

    def test_forward_backward(self):
        layer = LoTR(3, 4, 2)
        layer.requires_grad_()
        # Make gradient step.
        opt = T.optim.SGD(layer.parameters(), 1e-3)
        opt.zero_grad()
        xs = T.arange(6, dtype=T.float32).reshape(2, 3)
        ys = layer(xs)
        loss = (ys.sum() - 1) ** 2
        loss.backward()
        opt.step()


class TestLoTRLinear:

    def test_init(self):
        layer = LoTRLinear(3, 4, 2)
        assert layer.in_features == 3
        assert layer.out_features == 4
        assert layer.rank == 2

    def test_from_linear(self):
        lotr = LoTR(2, 3, 4)
        linear = T.nn.Linear(3, 4)
        layer = LoTRLinear.from_linear(linear, lotr)
        assert layer.in_features == linear.in_features
        assert layer.out_features == linear.out_features
        assert layer.rank == lotr.rank
        assert layer.lotr.lhs.data.data_ptr() == lotr.lhs.data.data_ptr()
        assert layer.lotr.rhs.data.data_ptr() == lotr.rhs.data.data_ptr()

    def test_forward_backward(self):
        layer = LoTRLinear(3, 4, 2)
        layer.requires_grad_()
        # Make gradient step.
        opt = T.optim.SGD(layer.parameters(), 1e-3)
        opt.zero_grad()
        xs = T.arange(6, dtype=T.float32).reshape(2, 3)
        ys = layer(xs)
        loss = (ys.sum() - 1) ** 2
        loss.backward()
        opt.step()
