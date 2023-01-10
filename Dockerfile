FROM python:3.9-slim
LABEL author="Vladimir Zaigrajew"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            python3-dev \
            curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY mnist_jax.py /workspace/mnist_jax.py
COPY mnist_pytorch.py /workspace/mnist_pytorch.py
COPY mnist_tensorflow.py /workspace/mnist_tensorflow.py
COPY run.sh /workspace/run.sh

COPY mlcube.yaml /mlcube/mlcube.yaml
COPY workspace/train.yaml /mlcube/workspace/train.yaml

RUN chmod +x /workspace/run.sh

ENTRYPOINT ["/bin/bash", "/workspace/run.sh"]