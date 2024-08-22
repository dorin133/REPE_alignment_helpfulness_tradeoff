#!/bin/bash
export HF_DATASETS_CACHE='/export/work/dshteyma/.cache/huggingface/datasets'
export HF_HOME='/export/work/dshteyma/.cache/huggingface'

TARGET=microsoft/Phi-3-mini-4k-instruct
OUTPUT_DIR=~/test_output
MAX_NEW_TOKENS=256
NUM_ASST_TOKENS=3
ITERS=100
TAIL=
TASK=
SAMPLING=
TARGET_GEN_CONFIG=
METRICS="inference_latency acceptance_rate output_recorder"

while test $# -gt 0; do
    case "$1" in
    --target_name*)
        shift
        TARGET=$1
        shift
        ;;
    --output_dir)
        shift
        OUTPUT_DIR=$1
        shift
        ;;
    --max_new_tokens)
        shift
        MAX_NEW_TOKENS=$1
        shift
        ;;
    --num_assistant_tokens)
        shift
        NUM_ASST_TOKENS=$1
        shift
        ;;
    --iter)
        shift
        ITERS=$1
        shift
        ;;
    --task)
        shift
        TASK=$1
        shift
        ;;
    --metric_names)
        # add metrics until you reach the next flag
        shift
        METRICS=$1
        shift
        while test $# -gt 0; do
            case "$1" in
            --*)
                break
                ;;
            *)
                METRICS+=' '$1
                shift
                ;;
            esac
        done
        ;;
    --do_sample)
        # add sampling arguments until you reach the next flag
        SAMPLING+='do_sample=True'
        TARGET_GEN_CONFIG+='--target_generation'
        shift
        while test $# -gt 0; do
            case "$1" in
            --*)
                break
                ;;
            *)
                SAMPLING+=' '$1
                shift
                ;;
            esac
        done
        ;;
    *)
        TAIL+=' '$1
        shift
        ;;
    esac
done

DATASET=
case $TASK in
cnn-dm)
    DATASET+='--dataset_name abisee/cnn_dailymail:3.0.0 --dataset_input article --prompt_name cnn-dm --apply_chat_template'
    ;;
tinystories)
    DATASET+='--dataset_name roneneldan/TinyStories --dataset_input text --prompt_name default --add_bos_token --random_prefix'
    ;;
dollyalpaca)
    DATASET+='--dataset_name databricks/databricks-dolly-15k --dataset_input context instruction --prompt_name dolly_alpaca_prompt --apply_chat_template'
    ;;
dolly)
    DATASET+='--dataset_name databricks/databricks-dolly-15k --dataset_input context instruction --prompt_name dolly --apply_chat_template'
    ;;
*)
    echo "Unknown task: $TASK, using provided/default dataset_name, dataset_input, and prompt_name"
    ;;
esac

CMD="python -m fastdraft.evaluation
    --target_name $TARGET 
    --metric_names $METRICS 
    $DATASET 
    --draft_generation num_assistant_tokens=$NUM_ASST_TOKENS num_assistant_tokens_schedule=const $SAMPLING
    $TARGET_GEN_CONFIG $SAMPLING
    --target_dtype float16 
    --draft_dtype float16 
    --device cuda 
    --max_new_tokens $MAX_NEW_TOKENS 
    --iter $ITERS 
    --output_dir $OUTPUT_DIR 
    $TAIL 
    --overwrite"
echo $CMD

$CMD
