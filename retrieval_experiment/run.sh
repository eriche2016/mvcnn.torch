#!/bin/bash

if [ 1 -eq 1 ]; then  
    CUDA_VISIBLE_DEVICES=0 th -i generate_features.lua
fi

# compute retrieval metrics
if [ 1 -eq 1 ]; then 
    CUDA_VISIBLE_DEVICES=0 th -i map.lua
fi 
