from transformers import RobertaForSequenceClassification

from train import convert_model, train


def test_convert_model():
    model_path = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model = convert_model(model, rank=1, adaptive=True)
    print(model)


def test_train():
    train(task='cola', rank=1)
