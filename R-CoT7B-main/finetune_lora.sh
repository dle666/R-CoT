#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# export MODEL="internlm/internlm-xcomposer-vl-7b"
export MODEL="pretrain_weight_path"
# export DATA="data.txt"
export DATA="data.txt"

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --given_num True \
    --fp16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --output_dir output/finetune_rcot7b_lora \
    --num_train_epochs 1 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps=2 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 4096 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing False
