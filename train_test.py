from transformers import RobertaForSequenceClassification

from train import convert_model, mask_weights, train

MODEL_PATH = 'roberta-base'


def test_convert_model():
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model = convert_model(model, rank=1, adaptive=True)
    assert model is not None  # Quite useless check.


def test_mask_weights():
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model = convert_model(model, rank=1, adaptive=True)
    model = mask_weights(model)
    trainable = sum(p.requires_grad for p in model.parameters())
    total = len([*model.parameters()])
    assert trainable != total


def test_train():
    train(task='cola', rank=1)
