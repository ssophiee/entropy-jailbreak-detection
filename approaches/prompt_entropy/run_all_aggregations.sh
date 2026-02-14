#!/bin/bash

MODEL="Qwen/Qwen2.5-3B-Instruct"
OUTDIR="results/prompt_entropy/${MODEL}_all_keys"
SUMMARY="$OUTDIR/summary.csv"

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

mkdir -p $OUTDIR
echo "key,auroc,auroc_flipped" > $SUMMARY

for KEY in "${KEYS[@]}"; do
  echo "Running key: $KEY"

  OUTPUT=$(CUDA_VISIBLE_DEVICES=0 python -m approaches.prompt_entropy.compute_prompt_entropy \
    --model_name $MODEL \
    --score_key $KEY \
    --output_dir $OUTDIR/$KEY)

  AUROC=$(echo "$OUTPUT" | grep "auroc " | awk '{print $2}')
  AUROC_FLIPPED=$(echo "$OUTPUT" | grep "auroc_flipped" | awk '{print $2}')

  echo "$KEY,$AUROC,$AUROC_FLIPPED" >> $SUMMARY
done

echo "Done. Summary saved to $SUMMARY. This may not have saved successfully. If so, launch the summary.sh script"
