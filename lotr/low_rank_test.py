import torch as T

from lotr.low_rank import LowRankLinear


class TestLowRankLinear:

    def test_forward(self):
        linear = LowRankLinear(10, 10)
        xs = T.arange(6 * 10, dtype=T.float32) \
            .requires_grad_(True) \
            .reshape(2, 3, 10)
        ys = linear(xs)
        assert xs.shape == ys.shape
