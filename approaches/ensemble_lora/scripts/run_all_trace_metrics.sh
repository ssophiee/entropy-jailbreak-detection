#!/bin/bash
# Run the 20 entropy-trace aggregation metrics for ensemble LoRA.
#
# Usage:
#   bash approaches/ensemble_lora/scripts/run_all_trace_metrics.sh <ensemble_dir>
#
# Example:
#   bash approaches/ensemble_lora/scripts/run_all_trace_metrics.sh saved_models/ensemble_lora_20250601

set -euo pipefail

ENSEMBLE_DIR="${1:?Usage: $0 <ensemble_dir>}"
OUTDIR="results/ensemble_lora_trace/all_keys"

KEYS=(
  mean
  std
  min
  max
  median
  p10
  p90
  trimmed_mean
  first_mean
  last_mean
  auc
  slope
  frac_above_q
  range
  delta_end
  delta_seg
  spearman_rho
  total_variation
  monotonicity_up
  peak_pos
)

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "key,auroc,auroc_flipped,average_precision,tpr@1%fpr" > "$SUMMARY"

for KEY in "${KEYS[@]}"; do
  echo "===== Running key: $KEY ====="

  OUTPUT=$(CUDA_VISIBLE_DEVICES=0 python -m approaches.ensemble_lora.scripts.compute_ensemble_trace_metrics \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --score_key "$KEY" \
    --output_dir "$OUTDIR/$KEY")

  AUROC=$(echo "$OUTPUT" | grep -E "^\s*auroc\s" | awk '{print $NF}')
  AUROC_FLIPPED=$(echo "$OUTPUT" | grep "auroc_flipped" | awk '{print $NF}')
  AP=$(echo "$OUTPUT" | grep "average_precision" | awk '{print $NF}')
  TPR1=$(echo "$OUTPUT" | grep "tpr@1%fpr" | awk '{print $NF}')

  echo "$KEY,$AUROC,$AUROC_FLIPPED,$AP,$TPR1" >> "$SUMMARY"
done

echo ""
echo "Done. Summary saved to $SUMMARY"
echo ""
echo "Alternatively, run with --run_all for a single invocation:"
echo "  python -m approaches.ensemble_lora.scripts.compute_ensemble_trace_metrics \\"
echo "    --ensemble_dir $ENSEMBLE_DIR --run_all"
