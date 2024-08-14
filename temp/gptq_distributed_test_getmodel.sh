export MASTER_ADDR="localhost"
export MASTER_PORT="29501"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

NUM_PROC=4  # number of devices for Tensor Parallel

if [ $NUM_PROC -eq 1 ]; then
    export GPTQ_DISTRIBUTED=0
else
    export GPTQ_DISTRIBUTED=1
fi

torchrun --nproc-per-node $NUM_PROC \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    gptq_distributed_test_getmodel.py #&> 20240628_llama-70b_TP${NUM_PROC}_gpu_debug.log
