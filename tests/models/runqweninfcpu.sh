#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright IBM Corp.
# SPDX-License-Identifier: Apache-2.0


TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
MODEL_NAME="qwen3/Qwen3-1.7B"
MODEL_PATH="/home/kurtis/tmp/models/qwen3-1.7B-old"
TOKENIZER=${MODEL_PATH}
OUT_DIR="${HOME}/tmp/aiu/${MODEL_NAME}/${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
RC=$?
if (( RC != 0 )); then
  echo "Error creating directory, aboring script"
  exit 1
fi
OUT_FILE="${OUT_DIR}/run_output.txt"

#    --max_new_tokens=8 
#   --no_early_termination 
#   --batch_size=1 \
# time PYTHONUNBUFFERED="1" TORCH_LOGS="dynamo,graph_breaks" python3 /tmp/aiu-fms-testing-utils/scripts/inference.py \
# time PYTHONUNBUFFERED="1" TORCH_LOGS="dynamo,graph_breaks" python3 ./aiu-fms-inference.py \
python3 ./cpufmsinf.py \
    --architecture=qwen3 \
    --variant=1.7b \
    --model_path="${MODEL_PATH}" \
    --model_source=hf \
    --tokenizer="${TOKENIZER}" \
    --min_pad_length=64 \
    --device_type=cpu \
    --compile \
    --default_dtype=fp16 \
    --unfuse_weights \
    > "${OUT_DIR}/run_output.txt" 2>&1
    RC=$?
    echo "Output is in file '${OUT_FILE}'"
    echo "RC=${RC}"

