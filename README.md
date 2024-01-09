# LoTR: Low Tensor Rank Adaptation of Large Language Models

*Parameter efficient fine-tuning*

## Overview

An attempt to introduce dynamic rank in [LoRa][1] as it did in \[[2][2]\] and
improve optimization algorithm in general.

![Comparison of LoRA against LoTR][3]

[1]: https://arxiv.org/abs/2106.09685
[2]: https://arxiv.org/abs/2205.13571
[3]: ./doc/iclr2024/fig/parameter-efficiency.png

## Experiments

### Logging Files

We assume that all raw experiment results (i.e. logging files, first of all)
are located in `log` directory. This directory's high-level structure should
reflect experimental setup. So the path relative to this directory should have
structure as follows.

```
<dataset>/<model>/<method>/<param1>/<param2>/.../<seed>/<tfevents-file>
```

The model segment preceeds the method path segment since number of differnt
models usually are smaller that number of methods and training pipeline usually
parameterized by model and then by method. All floating point parameters should
be used in scientific notation to ensure that no significant digits are lost.
The lat directory is random seed used to run an experiment.

Note that the requirements above are involuntary since there is no
full-featured machine learning experiment management software.
