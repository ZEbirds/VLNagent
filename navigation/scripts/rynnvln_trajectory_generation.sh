#!/bin/bash


OUTPUT_PATH = './rynn_data'
mkdir -p ${OUTPUT_PATH}

python3 src/rynnvln_trajectory_generation.py  \
        --data-root data/trajectory_data/R2R/ \
        --ann-path data/trajectory_data/R2R/annotations.json \
        --output-file  ${OUTPUT_PATH}/r2r_annotations_sft.json \
        --num-workers 32\ 

