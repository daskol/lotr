FROM ubuntu:20.04

RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -y --no-install-recommends \
        python-is-python3 \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install six tensorboard==2.11.0  # Fix version for Ubuntu 20.04.

WORKDIR /data

CMD ["/usr/local/bin/tensorboard", "serve", "--host", "0.0.0.0", \
     "--load_fast", "true", "--logdir", "/data/tensorboard", \
     "--reload_interval", "2.5"]
