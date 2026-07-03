#!/usr/bin/env bash
# Launch the five vanilla OPD baselines sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    "$SCRIPT_DIR/run_qwen3vl_32b_to_2b_fsdp.sh"
    "$SCRIPT_DIR/run_qwen3vl_32b_to_4b_fsdp.sh"
    "$SCRIPT_DIR/run_qwen3vl_8b_to_2b_fsdp.sh"
    "$SCRIPT_DIR/run_qwen3vl_8b_to_4b_fsdp.sh"
    "$SCRIPT_DIR/run_qwen3vl_4b_to_2b_fsdp.sh"
)

start_index=${START_INDEX:-0}
end_index=${END_INDEX:-4}

for i in "${!scripts[@]}"; do
    if (( i < start_index || i > end_index )); then
        continue
    fi
    echo "===== Running vanilla OPD baseline $((i + 1))/5: ${scripts[$i]} ====="
    "${scripts[$i]}" "$@"
done
