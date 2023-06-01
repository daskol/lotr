import torch as T

from linear import AdaptiveLoRaLinear, Mode


def test_linear():
    inp = T.ones((32, 8))
    module = AdaptiveLoRaLinear(8, 8)
    module(inp, Mode.K)
    module(inp, Mode.L)
    module(inp, Mode.S)
