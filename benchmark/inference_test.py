import logging
import re
from copy import deepcopy
from functools import partial
from gzip import open as gzopen
from json import load
from pathlib import Path

import numpy as np
import pytest
import torch as T
from transformers import (AutoTokenizer, PreTrainedModel,
                          RobertaForSequenceClassification)

from lotr import LoRALinear, LoTR, LoTRLinear, map_module

# Silence info messages and warning like weigths are not initialized.
logging.getLogger('transformers').setLevel(logging.ERROR)

# We use this global device toggle in order to make benchmark simpler. In order
# to run benchmark on CPU just `export CUDA_VISIBLE_DEVICES=`.
if T.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def load_sample(path: Path) -> tuple[list[str], list[str]]:
    with gzopen(path) as fin:
        json = load(fin)
    return json['question'], json['sentence']


def load_pretokenized_input(path: Path, tokenizer_name: str) -> np.ndarray:
    textual_data = load_sample(path)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    inputs = tokenizer(*textual_data, max_length=256, padding='max_length',
                       return_tensors='np')
    input_ids = inputs['input_ids']
    assert input_ids.shape == (64, 256)
    tokens = np.tile(input_ids, (16, 2))
    assert tokens.shape == (1024, 512)
    return tokens


@pytest.fixture(scope='module')
def tokens():
    data_path = Path(__file__).parent / 'testdata/sample.64.json.gz'
    sample = load_pretokenized_input(data_path, 'roberta-base')
    assert sample.shape == (1024, 512)
    return T.tensor(sample)


@pytest.fixture(scope='module')
def model():
    return RobertaForSequenceClassification \
        .from_pretrained('roberta-base', num_labels=2) \
        .requires_grad_(False)


@pytest.mark.benchmark(
    group='inference',
    min_rounds=20,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.slow
@pytest.mark.usefixtures('benchmark', 'model', 'tokens')
class LoTRBenchmark:

    RE_CORRECTION = re.compile(r'/roberta/encoder/layer/\d+'
                               r'/attention/self/(value|query)/correction/.*')

    RE_HEAD = re.compile(r'/classifier/.*')

    RE_PATTERNS = (RE_CORRECTION, RE_HEAD)

    @staticmethod
    def convert_lotr(module: T.nn.Module, path: str, lotr: LoTR, **kwargs):
        if not isinstance(module, T.nn.Linear):
            raise ValueError('Only linear layer can be converted: '
                             f'type={type(module)}.')
        return LoTRLinear.from_linear(module, lotr, **kwargs)

    @staticmethod
    def convert_model(model, lotr: LoTR, **kwargs):
        return map_module(
            model, partial(LoTRBenchmark.convert_lotr, lotr=lotr, **kwargs),
            r'/roberta/encoder/layer/\d+/attention/self/(value|query)')

    @staticmethod
    def mask_weights(model: T.nn.Module):
        for name, param in model.named_parameters():
            name = '/' + name.replace('.', '/')
            for pattern in LoTRBenchmark.RE_PATTERNS:
                if pattern.match(name) is not None:
                    param.requires_grad = True
                    break
            else:
                param.requires_grad = False
        return model

    @pytest.mark.parametrize('rank', [8, 32, 64, 96, 128])
    @pytest.mark.parametrize('seq_len', [32, 128, 512])
    @pytest.mark.parametrize('batch_size', [16, 128])
    def test_lotr(self, benchmark, model: PreTrainedModel, tokens: T.Tensor,
                  batch_size: int, seq_len: int, rank: int):
        # Make contingues array of input batch.
        batch = tokens[:batch_size, :seq_len].contiguous().to(device)
        mask = T.ones_like(batch)
        pos_ids = T.arange(seq_len, device=device)

        # Clone model and add adapters.
        model = deepcopy(model)
        lotr = LoTR(model.config.hidden_size, model.config.hidden_size, rank)
        model = self.convert_model(model, lotr)
        model = self.mask_weights(model).to(device)

        with T.no_grad():
            benchmark(model, input_ids=batch, attention_mask=mask,
                      position_ids=pos_ids)


@pytest.mark.benchmark(
    group='inference',
    min_rounds=20,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.slow
@pytest.mark.usefixtures('benchmark', 'model', 'tokens')
class LoRABenchmark:

    RE_CORRECTION = re.compile(
        r'/roberta/encoder/layer/\d+/attention/self/(value|query)/factors/.*')

    RE_HEAD = re.compile(r'/classifier/.*')

    RE_PATTERNS = (RE_CORRECTION, RE_HEAD)

    @staticmethod
    def convert_low_rank(rank: int, module: T.nn.Module, path: str):
        if not isinstance(module, T.nn.Linear):
            raise ValueError('Only linear layer can be converted: '
                             f'type={type(module)}.')
        return LoRALinear.from_linear(module, rank)

    @staticmethod
    def convert_model(model, rank: int):
        return map_module(
            model, partial(LoRABenchmark.convert_low_rank, rank),
            r'/roberta/encoder/layer/\d+/attention/self/(value|query)')

    @staticmethod
    def mask_weights(model: T.nn.Module):
        for name, param in model.named_parameters():
            name = '/' + name.replace('.', '/')
            for pattern in LoRABenchmark.RE_PATTERNS:
                if pattern.match(name) is not None:
                    param.requires_grad = True
                    break
            else:
                param.requires_grad = False
        return model

    @pytest.mark.parametrize('rank', [8, 32, 64, 96, 128])
    @pytest.mark.parametrize('seq_len', [32, 128, 512])
    @pytest.mark.parametrize('batch_size', [16, 128])
    def test_lora(self, benchmark, model: PreTrainedModel, tokens: T.Tensor,
                  batch_size: int, seq_len: int, rank: int):
        # Make contingues array of input batch.
        batch = tokens[:batch_size, :seq_len].contiguous().to(device)
        mask = T.ones_like(batch)
        pos_ids = T.arange(seq_len, device=device)

        # Clone model and add adapters.
        model = deepcopy(model)
        model = self.convert_model(model, rank)
        model = self.mask_weights(model).to(device)

        with T.no_grad():
            benchmark(model, input_ids=batch, attention_mask=mask,
                      position_ids=pos_ids)
