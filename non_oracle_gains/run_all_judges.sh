#!/bin/bash
# Usage: bash run_all_judges.sh [B]   (default B=1000)
set -u
cd "$(dirname "$0")"

B="${1:-1000}"
PY="${PYTHON:-python3}"
OUT=../reports/non_oracle_gains
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

JUDGES=(gemma2_9b gemma3_27b qwen3_4b mistral_7b llama31_8b llama31_70b \
        gemini2_flash gemini3_flash gemini31_pro gpt54 opus47)

echo "[$(date '+%T')] running ${#JUDGES[@]} judges sequentially, B=$B"
fail=0
for j in "${JUDGES[@]}"; do
  echo "[$(date '+%T')] >>> $j"
  if $PY -u calibration_sweep.py --judge "$j" --B "$B" > "$LOGDIR/$j.log" 2>&1; then
    echo "[$(date '+%T')] DONE $j"
  else
    echo "[$(date '+%T')] FAILED $j (see $LOGDIR/$j.log)"; fail=1
  fi
done

echo "[$(date '+%T')] === verifying outputs ==="
for j in "${JUDGES[@]}"; do
  if [ -f "$OUT/per_judge/$j.json" ]; then echo "  ok   per_judge/$j.json"; else echo "  MISS per_judge/$j.json"; fail=1; fi
done
echo "[$(date '+%T')] all finished (fail=$fail)"
exit $fail
