#!/bin/bash
# This script computes UMPIRE with uncertainty quantification and early warning mechanism.
# Based on INSIDE paper approach for hallucination detection

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-1.5-13b-hf"
MODEL_PATH='llava-hf/llava-1.5-13b-hf'
SPLIT="okvqa_val2014"
IMG_DIR='/ai/teacher/ssz/all_data/mqa/OKVQA/val2014'
QUES_FILE='data/okvqa/okvqa_processed.jsonl'
OUTDIR="output_dir"

IDX=0
CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python pipeline/generate_with_uncertainty.py \
        --model_path $MODEL_PATH \
        --question_file $QUES_FILE \
        --image_folder $IMG_DIR \
        --outdir $OUTDIR/$SPLIT/generation_embedding/$CKPT/${CHUNKS}_${IDX} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 1 \
        --top_p '0.9' \
        --num_generations_per_prompt 5 \
        --fraction_of_data_to_use 0.01 \
        --enable_early_warning \
        --entropy_threshold 0.7 \
        --variance_threshold 0.5 \
        --early_stop_consecutive 2 \
        --record_uncertainty \
        --eval_all_layers

wait

python pipeline/merge_generation.py \
        --generation_dir $OUTDIR/$SPLIT/generation_embedding/$CKPT

python pipeline/evaluate_uncertainty.py \
        --generation_file=$OUTDIR/$SPLIT/generation_embedding/$CKPT/${CHUNKS}_${IDX}/generations_with_uncertainty.pkl \
        --output_dir=$OUTDIR/$SPLIT/results \
        --uncertainty_weight 0.5
