import numpy as np
import pytest
import torch as T
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_less)
from transformers import get_polynomial_decay_schedule_with_warmup

from lotr.optim import (LoTRAdam, LoTRAdamW, LoTRSGD, LowRankAdam, LowRankSGD,
                        LRSchedulerList, OptimizerList, Tucker,
                        filter_trainable, sanitize_param_group, tucker2)


@pytest.fixture
def layer(emb_size=8):
    layer = T.nn.Linear(emb_size, emb_size)
    layer.bias.requires_grad_(False)
    return layer


@pytest.fixture
def layers(emb_size=8, depth=2):
    layers = []
    for _ in range(depth):
        layer = T.nn.Linear(emb_size, emb_size)
        layer.bias.requires_grad_(False)
        layers.append(layer)
    return T.nn.Sequential(*layers)


class TestLowRankMixin:

    RANK = 2

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


def test_filter_trainable_list(layer: T.nn.Linear):
    # Check that layer is "semi-trainable".
    assert layer.weight.requires_grad
    assert not layer.bias.requires_grad
    inp_params = [*layer.parameters()]
    out_params = filter_trainable(inp_params)
    assert isinstance(out_params, list)
    assert len(inp_params) != len(out_params)
    assert len(out_params) == 1


def test_filter_trainable_dict(layer: T.nn.Linear):
    # Check that layer is "semi-trainable".
    assert layer.weight.requires_grad
    assert not layer.bias.requires_grad
    inp_params = [*layer.parameters()]
    param_group = filter_trainable({'params': inp_params})
    assert isinstance(param_group, dict)
    assert 'params' in param_group
    out_params = param_group['params']
    assert len(inp_params) != len(out_params)
    assert len(out_params) == 1


class TestSanitizeParamGroup:

    def test_empty(self, layers: T.nn.Sequential):
        sanitize_param_group([])  # No exception is raised.

    def test_correct(self, layers: T.nn.Sequential):
        assert len(layers) == 2, 'Expected a model of depth 2.'
        params = filter_trainable(layers.parameters())
        param_group = sanitize_param_group(params)
        assert 'params' in param_group
        assert len(param_group['params']) == len(layers)

    def test_invalid(self, layers: T.nn.Sequential):
        emb_size = layers[0].in_features
        layers.append(T.nn.Linear(2 * emb_size, 2 * emb_size))
        with pytest.raises(ValueError):
            sanitize_param_group(layers.parameters())


class TestLoTRMixin:

    RANK = 2

    def make_step(self, model: T.nn.Sequential, opt: T.optim.Optimizer):
        layer = model[0]
        emb_size = layer.in_features
        param = layer.weight
        for _ in range(2):
            xs = T.arange(3 * emb_size, dtype=T.float32) \
                .requires_grad_(True) \
                .reshape((3, emb_size))
            ys = model(xs)
            ys.backward(T.ones_like(ys))

            opt.step()
            opt.zero_grad()

            # Verify that factors exists and check their shapes.
            state = opt.state[param]
            assert state != {}
            # Factors are stored as an model of Tucker2 decomposition.
            decomp = state['decomp']
            assert isinstance(decomp, Tucker)
            assert decomp.shape == (emb_size, len(model), emb_size)
            assert decomp.modes[0].shape == (emb_size, self.RANK)
            assert decomp.modes[1] is None
            assert decomp.modes[2].shape == (emb_size, self.RANK)

    def test_adam(self, layers):
        params = filter_trainable(layers.parameters())
        opt = LoTRAdam(params, rank=self.RANK)
        self.make_step(layers, opt)

    def test_adamw(self, layers):
        params = filter_trainable(layers.parameters())
        opt = LoTRAdamW(params, rank=self.RANK)
        self.make_step(layers, opt)

    def test_sgd(self, layers):
        params = filter_trainable(layers.parameters())
        opt = LoTRSGD(params, lr=1e-3, rank=self.RANK)
        self.make_step(layers, opt)


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


def test_tucker2():
    inp = T.arange(2 * 3 * 4, dtype=T.float64).reshape(2, 3, 4)
    out = tucker2(inp, 2, axis=-2)

    assert out.ndim == 3
    assert out.shape == inp.shape
    assert out.axis == 1
    assert out.size == 3
    assert out.rank == (2, 3, 2)
    assert out.core.shape == out.rank

    assert len(out.modes) == 3
    assert out.modes[0].shape == (2, 2)
    assert out.modes[1] is None
    assert out.modes[2].shape == (4, 2)

    ten = T.einsum('ajb,ia,kb->ijk', out.core, out.modes[0], out.modes[2])
    assert_allclose(inp, ten, atol=1e-6, rtol=1e0)
