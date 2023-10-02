FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace/lotr

ADD pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/pip \
    export GIST=https://gist.githubusercontent.com/daskol/5513ff9c5b8a2d6b2a0e78f522dd2800; \
    wget $GIST/raw/4e7b80e5f9d49c2e39cf8aa4e6b6b8b951724730/peds.py && \
    python peds.py -e dev -e exp -i . && \
    rm -rf peds.py && \
    pip install build setuptools 'setuptools[toml]>=7' wheel
