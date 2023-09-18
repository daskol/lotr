import numpy as np
import torch as T
from numpy.testing import assert_array_equal, assert_array_less
from transformers.modeling_outputs import SequenceClassifierOutput

from lotr_glue import instantiate_model, instantiate_optimizer, train


def test_instantiate_model():
    instantiate_model()


def test_instantiate_optimizer():
    num_samples = 8551
    num_epoches = 20
    batch_size = 16

    model = instantiate_model()
    opt, scheduler = instantiate_optimizer(model, 1e-5, 8, batch_size,
                                           num_epoches, num_samples)


def test_step():
    num_samples = 8551
    num_epoches = 20
    batch_size = 16

    model = instantiate_model()
    opt, scheduler = instantiate_optimizer(model, 1e-5, 8, batch_size,
                                           num_epoches, num_samples)

    # Apply model to synthetic batch.
    input_ids = T.arange(2 * 128).reshape((2, 128))
    labels = T.arange(2)
    output: SequenceClassifierOutput = model(input_ids, labels=labels)
    output.loss.backward()

    opt.step()
    opt.zero_grad()

    for optimizer in opt.optimizers:
        lrs = np.array([x['lr'] for x in optimizer.param_groups])
        assert_array_equal(lrs, 0)

    scheduler.step()

    for optimizer in opt.optimizers:
        lrs = np.array([x['lr'] for x in optimizer.param_groups])
        assert_array_less(0, lrs)


def test_train():
    train(task='cola',
          batch_size=8,
          num_epoches=1,
          enable_lotr=True,
          rank=2,
          lr=1e-3)
