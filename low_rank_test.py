import pytest
import torch as T

from low_rank import LowRankAdam, LowRankLinear, LowRankSGD


class TestLowRankLinear:

    def test_forward(self):
        linear = LowRankLinear(10, 10)
        xs = T.arange(6 * 10, dtype=T.float32) \
            .requires_grad_(True) \
            .reshape(2, 3, 10)
        ys = linear(xs)
        assert xs.shape == ys.shape


class TestLowRankMixin:

    RANK = 2

    @pytest.fixture
    def layer(self, emb_size=8):
        layer = T.nn.Linear(emb_size, emb_size)
        layer.bias.requires_grad_(False)
        return layer

    def make_step(self, layer: T.nn.Linear, opt: T.optim.Optimizer):
        emb_size = layer.in_features
        param = layer.weight
        for _ in range(2):
            xs = T.arange(3 * emb_size, dtype=T.float32) \
                .requires_grad_(True) \
                .reshape((3, emb_size))
            ys = layer(xs)
            ys.backward(T.ones_like(ys))

            opt.step()
            opt.zero_grad()

            # Verify that factors exists and check their shapes.
            state = opt.state[param]
            assert state != {}
            assert state['low-rank'][0].shape == (emb_size, self.RANK)
            assert state['low-rank'][1].shape == (emb_size, self.RANK)

    def test_adam(self, layer):
        opt = LowRankAdam(layer.parameters(), rank=self.RANK)
        self.make_step(layer, opt)

    def test_sgd(self, layer):
        opt = LowRankSGD(layer.parameters(), 1e-3, rank=self.RANK)
        self.make_step(layer, opt)
