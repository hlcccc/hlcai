# !/bin/bash
# This script computes UMPIRE and evaluates the results.
# Change the paths as necessary.
SPLIT="okvqa_val2014"
CKPT="llava-1.5-13b-hf"
generation_file="output_dir/${SPLIT}/generation_embedding/${CKPT}.pkl"
output_dir="output_dir/${SPLIT}/results"

# Compute UMPIRE and evaluate
python pipeline/compute_umpire_and_evaluate.py \
        --generation_file=$generation_file \
        --output_dir=$output_dir