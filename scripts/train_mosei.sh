#!bin/bash
NUM_PROC=$1
CONFIG=$2
LOG_DIR='./output/'
curr_time=$(date "+%Y_%m_%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 12349 train.py \
--config ${CONFIG} --data mosei --exp layers53 \
2>&1 | tee -a ${LOG_DIR}/${curr_time}.log

#  cp -r `ls . | grep -v  | xargs` dir-B
