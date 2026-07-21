#!/bin/sh
set -e

BASE="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"
MODEL="model.onnx"
TOKENIZER="tokenizer.json"

echo "Creating models directory."
mkdir -p models

echo "Downloading ${MODEL}."
curl -L --fail -o "models/${MODEL}" "${BASE}/onnx/${MODEL}"

echo "Downloading ${TOKENIZER}."
curl -L --fail -o "models/${TOKENIZER}" "${BASE}/${TOKENIZER}"

echo "Model files are in models/"
