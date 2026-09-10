#!/bin/bash
# The two PCR Benchmark tasks released at qfchou/HOMIE-PCR.
#
#   hf download qfchou/HOMIE-PCR --repo-type dataset --local-dir ./pcr
#   ARCH_IMAGES=/path/to/book_set/images bash run_pcr.sh
#
# ARCH images are not redistributed: https://warwick.ac.uk/fac/cross_fac/tia/data/arch
set -e

MODEL="${MODEL:-qfchou/HOMIE-Qwen3-VL-2B}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-$MODEL}"
ARCH_IMAGES="${ARCH_IMAGES:?set ARCH_IMAGES to the ARCH Bookset images folder}"
PCR="${PCR:-./pcr}"
GPUS="${GPUS:-0}"
PORT="${PORT:-29505}"
CSV="${CSV:-./pcr.csv}"

EVAL="$(dirname "$0")/eval/eval_zeroshot/eval_path.py"

run() { CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --main_process_port "$PORT" "$EVAL" "$@"; }

# --- (qi,qi,...) -> ct : multi-image query -> caption ----------------------------
run \
    --image_data_path "$ARCH_IMAGES" \
    --data_path "$PCR/multi_image_to_text.json" \
    --original_model_id "$ORIGINAL_MODEL" --model_id "$MODEL" \
    --task multire --batch_size 1 \
    --csv_output "$CSV"

# --- (qi,qt) -> ci : image + relational modifier -> target image -----------------
run \
    --image_data_path "$ARCH_IMAGES" \
    --data_path "$PCR/image_text_to_image.json" \
    --original_model_id "$ORIGINAL_MODEL" --model_id "$MODEL" \
    --task composedretrieval --batch_size 4 \
    --csv_output "$CSV"
