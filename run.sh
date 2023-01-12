#!/bin/bash

# Set workspace env to use in configs
workspace_path=${2##*=}
export WORKSPACE_PATH=$workspace_path

# Case statement which file with dedicated framework to run
case "$1" in
  pytorch)
    python /workspace/mnist_pytorch.py ;;
  tensorflow)
    python /workspace/mnist_tensorflow.py ;;
  jax)
    python /workspace/mnist_jax.py ;;
  *)
    echo "Sorry, you need to give one of this string `pytorch`, `tensorflow`, `jax` not $1" ;;
esac