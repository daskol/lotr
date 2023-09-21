import numpy as np
import pytest
import torch as T
from numpy.testing import assert_array_equal, assert_array_less
from transformers import get_polynomial_decay_schedule_with_warmup

from lotr.optim import LowRankAdam, LowRankSGD, LRSchedulerList, OptimizerList


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


class TestOptimizerList:

    def test_init(self, emb_size=8):
        layer = T.nn.Linear(emb_size, emb_size)
        opt_adam = T.optim.Adam([layer.bias])
        opt_lr = LowRankAdam([layer.weight])
        opt = OptimizerList([opt_adam, opt_lr])

        warmup_steps = 10
        total_steps = 100
        scheduler_adam = get_polynomial_decay_schedule_with_warmup(
            opt_adam, warmup_steps, total_steps)
        scheduler_lr = get_polynomial_decay_schedule_with_warmup(
            opt_lr, warmup_steps, total_steps)
        scheduler = LRSchedulerList([scheduler_adam, scheduler_lr])

        xs = T.arange(3 * emb_size, dtype=T.float32) \
            .requires_grad_(True) \
            .reshape((3, emb_size))
        ys = layer(xs)
        ys.backward(T.ones_like(ys))

        opt.step()
        opt.zero_grad()
        for optimizer in opt.optimizers:
            lrs = np.array([x['lr'] for x in optimizer.param_groups])
            assert_array_equal(lrs, 0)

        scheduler.step()
        for optimizer in opt.optimizers:
            lrs = np.array([x['lr'] for x in optimizer.param_groups])
            assert_array_less(0, lrs)

        # HuggingFace's transformers requires this method for logging learning
        # rate at the end of epoch.
        lrs = scheduler.get_last_lr()
        assert isinstance(lrs, list)
        assert len(lrs) > 0
