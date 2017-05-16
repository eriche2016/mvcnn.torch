#!/bin/bash

if [ 1 -eq 0 ]; then 
    th prepare_data.lua
fi

if [ 1 -eq 0 ]; then 
    th -i test_dataloader.lua
fi
# predict view score 
if [ 1 -eq 0 ]; then 
    CUDA_VISIBLE_DEVICES=3 th -i train_view_score_prediction_net.lua
fi 

# get view score for Training and testting set 
if [ 1 -eq 0 ]; then 
    CUDA_VISIBLE_DEVICES=1 th -i view_score_getter.lua
fi 

# train multiview cnn 
if [ 1 -eq 0 ]; then 
    CUDA_VISIBLE_DEVICES=3 th -i train_mv.lua
fi

# train multiview cnn with LBP 
if [ 1 -eq 1 ]; then 
    CUDA_VISIBLE_DEVICES=3 th -i train_mv_CircularConv.lua
fi 
