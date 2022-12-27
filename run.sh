#!/bin/bash

case "$1" in
  pytorch)
    python /workspace/train_pytorch.py ;;
  tensorflow)
    python /workspace/train_tensorflow.py ;;
  jax)
    python /workspace/train_jax.py ;;
  *)
    echo "Sorry, you need to give one of this string `pytorch`, `tensorflow`, `jax` not $1" ;;
esac