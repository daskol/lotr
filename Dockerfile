FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace/lotr

ADD pyproject.toml .

# First of all we need to update a `pip` since there is an issue with resolving
# of immediate dependencies because of transitive ones.
RUN --mount=type=cache,target=/root/.cache/pip \
    export GIST=https://gist.githubusercontent.com/daskol/5513ff9c5b8a2d6b2a0e78f522dd2800; \
    wget $GIST/raw/4e7b80e5f9d49c2e39cf8aa4e6b6b8b951724730/peds.py && \
    pip install -U build pip setuptools 'setuptools[toml]>=7' wheel && \
    python peds.py -e dev -e exp -i . && \
    rm -rf peds.py
