import re
from operator import attrgetter
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal
from transformers import RobertaForSequenceClassification

from lotr.init import make_lotr_init
from lotr.lotr import LoTR, LoTRLinear

RE_FILTER = re.compile(
    r'roberta.encoder.layer.\d+.attention.self.(value|query)')


def attrsetter(obj: Any, attr: str, val: Any):
    prefix, name = attr.rsplit('.', 1)
    getter = attrgetter(prefix)
    parent = getter(obj)
    return setattr(parent, name, val)


def test_make_lotr_init(rank: int = 80):
    # Load pretrained RoBERTa model..
    model = RobertaForSequenceClassification \
        .from_pretrained('roberta-base', num_labels=2) \
        .requires_grad_(False)
    modules = {k: m for k, m in model.named_modules() if RE_FILTER.match(k)}

    # Replace default Linear layers with LoTRLinear layers.
    layers = []
    lotr = LoTR(model.config.hidden_size, model.config.hidden_size, rank)
    for path, module in modules.items():
        layer = LoTRLinear.from_linear(module, lotr)
        layers.append(layer)
        attrsetter(model, path, layer)

    # Apply initializer.
    initializer = make_lotr_init('trivial', 'neutral', 'svd')
    initializer(layers)

    # Verify that everything initialized as expected.
    attn1 = model.roberta.encoder.layer[1].attention.self
    attn2 = model.roberta.encoder.layer[2].attention.self
    assert isinstance(attn1.query, LoTRLinear)
    assert isinstance(attn2.query, LoTRLinear)

    # In this test, Left factors are zeros, middle ones are ones, and right are
    # factors of Tucker decomposition.
    query1: LoTRLinear = attn1.query
    query2: LoTRLinear = attn2.query
    assert query1.lotr.lhs is query2.lotr.lhs
    assert query1.lotr.rhs is query2.lotr.rhs
    assert_array_equal(query1.lotr.lhs, 0)
    assert_array_equal(query1.lotr.mid, np.eye(rank))
    assert_array_equal(query2.lotr.mid, np.eye(rank))
