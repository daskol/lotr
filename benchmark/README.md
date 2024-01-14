# LoTR: Benchmarks

## Overview

One needs to install `pytest` and `pytest-benchmark` first. We assume that
current working directory is the repository root directory. In order to run all benchmark, one can execute the following command.

```shell
pytest -m slow --benchmark-json=inference.json benchmark/inference_test.py
```

The command above saves benchmark results to `inference.json` with all timings
in the current directory (repository root). If we want to inspect benchmark
data later or save them in CSV-formatted file, then one should run the
following.

```shell
pytest-benchmark compare inference.json --csv=inference.csv
```

This command prints tabular data and save them to `inference.csv` file.
