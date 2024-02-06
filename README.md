# LoTR: Low Tensor Rank Adaptation of Large Language Models

*Low Tensor Rank adaptation of large language models*

## Overview

This repository is the original implementation of LoTR ([arXiv:2402.01376][4]),
a novel approach for parameter-efficient fine-tuning of LLMs which represents a
gradient update to parameters in a form of tensor decomposition. Low-rank
adapter for each layer is constructed as a product of three matrices, and
tensor structure arises from sharing left and right multipliers of this product
among layers. Simultaneous compression of a sequence of layers with low-rank
tensor representation allows LoTR to archive even better parameter efficiency
then LoRA especially for deep models. Moreover, the core tensor does not depend
on original weight dimension and can be made arbitrary small, which allows for
extremely cheap and fast downstream fine-tuning.

```bibtex
@misc{bershatsky2024lotr,
  title         = {{LoTR}: Low Tensor Rank Weight Adaptation},
  author        = {Daniel Bershatsky and Daria Cherniuk and Talgat Daulbaev and Aleksandr Mikhalev and Ivan Oseledets},
  year          = {2024},
  eprint        = {2402.01376},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

[1]: https://arxiv.org/abs/2106.09685
[2]: https://arxiv.org/abs/2205.13571
[3]: ./doc/iclr2024/fig/parameter-efficiency.png
[4]: https://arxiv.org/abs/2402.01376

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

### Convertion to Arrow Parquet

TensorBoard `tfvents`-file are quite large files which take noticably long time
to read and load. So we convert `tfevents`-files to `parquet`-files with the
following command.

```shell
python -m lotr.tb2parquet log/glue data/glue.parquet \
    --names model method task lr rank seed \
```

Now, one can read a single `parquet`-file with all time series as follows.

```python
import pandas as pd
df = pd.read_parquet('data/glue.parquet')
```

To be more specific, 20Mb of `tfevents`-file are converted to 200Kb of
`parquet`-file.
