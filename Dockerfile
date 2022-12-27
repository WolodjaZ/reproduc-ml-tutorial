FROM ubuntu:20.04
LABEL author="Vladimir Zaigrajew"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            python3-dev \
            curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY train_jax.py /workspace/train_jax.py
COPY train_pytorch.py /workspace/train_pytorch.py
COPY train_tensorflow.py /workspace/train_tensorflow.py

COPY mlcube.yaml /mlcube/mlcube.yaml
COPY workspace/train.yaml /mlcube/workspace/train.yaml

RUN chmod +x /workspace/run.sh

ENTRYPOINT ["/bin/bash", "/workspace/run.sh"]