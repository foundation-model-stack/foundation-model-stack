#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

#export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1
#export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,autopilot=1,kvcacheopt=1,progshareopt=1
###unset DT_OPT
export HOST_ENDIAN_FORMAT=big

export SENCORELETS=2
export SENCORES=32
# export DEE_DUMP_GRAPHS=granite8b
export TOKENIZERS_PARALLELISM=false
# export DATA_PREC=fp16
export SENLIB_DEVEL_CONFIG_FILE=/root/.senlib.json
export AIU_CONFIG_FILE_0=/root/.senlib.json
# export FLEX_RDMA_PCI_BUS_ADDR_0="0000:00:00.0"
export AIU_WORLD_RANK_0="0000:00:00.0"
export DTLOG_LEVEL=error
### export TORCH_SENDNN_LOG=DEBUG
### unset TORCH_SENDNN_LOG
export DT_DEEPRT_VERBOSE=-1
# export COMPILATION_MODE=offline_decoder
# export DTCOMPILER_KEEP_EXPORT=true
export FLEX_COMPUTE=SENTIENT
export FLEX_DEVICE=VFIO
# export FLEX_OVERWRITE_NMB_FRAME=1
# export FLEX_MONITOR_SLEEP=0
# unset FMS_SKIP_TP_EMBEDDING

TRAIN_DATA=$1
INPUT_MODEL=$2
OUTPUT_MODEL=$3
TOKENIZER=$4
BACKEND=$5

echo ${TRAIN_DATA}
echo ${INPUT_MODEL}
echo ${OUTPUT_MODEL}
echo ${TOKENIZER}
echo ${BACKEND}

DEE_DUMP_GRAPHS=bert DTCOMPILER_KEEP_EXPORT=1 PYTHONUNBUFFERED=1 TORCH_SENDNN_TRAIN=1 TORCH_LOGS=dynamo \
python3 -u ./train_classification.py \
	--architecture=bert_classification \
	--variant=base \
	--num_classes=2 \
	--checkpoint_format=hf \
	--model_path=${INPUT_MODEL} \
	--tokenizer=${TOKENIZER} \
	--device_type=cpu \
	--dataset_path=${TRAIN_DATA} \
	--head_only \
	--batch_size=1 \
	--compile \
	--compile_backend=${BACKEND} \
	--dataset_style=aml \
	--unfuse_weights \
	--batch_size=4 \
	--default_dtype="fp32" \
	--output_path=${OUTPUT_MODEL}
