#!/usr/bin/env bash
set -euo pipefail

# Summarise per-key results produced by run_all_trace_metrics.sh
# OR from a single --run_all JSON.
#
# Usage:
#   bash approaches/laplace_approx/scripts/summary.sh [results_dir] [summary.csv]
#
# Mode 1 – per-key dirs (default):
#   results_dir = results/laplace_trace/all_keys
#   Expects: results_dir/<key>/laplace_trace_<key>_*.json
#
# Mode 2 – single --run_all JSON:
#   bash approaches/laplace_approx/scripts/summary.sh path/to/laplace_trace_all_keys_*.json

ROOT="${1:-results/laplace_trace/all_keys}"
SUMMARY="${2:-$ROOT/summary.csv}"

KEYS=(
  mean std min max median
  p10 p90 trimmed_mean
  first_mean last_mean
  auc slope
  frac_above_q range
  delta_end delta_seg spearman_rho
  total_variation monotonicity_up
  peak_pos
)

# Detect mode: single JSON file vs directory of per-key results
if [[ -f "$ROOT" && "$ROOT" == *.json ]]; then
  # Mode 2: single --run_all JSON
  JSON_FILE="$ROOT"
  SUMMARY="${2:-$(dirname "$JSON_FILE")/summary.csv}"

  echo "key,accuracy,auroc,auroc_flipped,average_precision,tpr@1%fpr,tpr@0.1%fpr,ece,brier" > "$SUMMARY"

  for KEY in "${KEYS[@]}"; do
    LINE="$(python3 - <<'PY' "$JSON_FILE" "$KEY"
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    d = json.load(f)
m = d.get("detection_metrics", {}).get(key, {})
def g(name):
    v = m.get(name, "")
    return f"{v:.6g}" if isinstance(v, float) else str(v)
print(",".join([key, g("accuracy"), g("auroc"), g("auroc_flipped"),
                g("average_precision"), g("tpr@1%fpr"), g("tpr@0.1%fpr"),
                g("ece"), g("brier")]))
PY
)"
    echo "$LINE" >> "$SUMMARY"
  done

else
  # Mode 1: per-key directories
  echo "key,accuracy,auroc,auroc_flipped,average_precision,tpr@1%fpr,tpr@0.1%fpr,ece,brier,json_path" > "$SUMMARY"

  for KEY in "${KEYS[@]}"; do
    DIR="$ROOT/$KEY"
    JSON_PATH="$(ls -t "$DIR"/*.json 2>/dev/null | head -n 1 || true)"

    if [[ -z "$JSON_PATH" ]]; then
      echo "WARN: no json found for key=$KEY in $DIR" >&2
      echo "$KEY,,,,,,,,," >> "$SUMMARY"
      continue
    fi

    LINE="$(python3 - <<'PY' "$JSON_PATH" "$KEY"
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    d = json.load(f)
m = d.get("detection_metrics", {}).get(key, {})
def g(name):
    v = m.get(name, "")
    return f"{v:.6g}" if isinstance(v, float) else str(v)
print(",".join([key, g("accuracy"), g("auroc"), g("auroc_flipped"),
                g("average_precision"), g("tpr@1%fpr"), g("tpr@0.1%fpr"),
                g("ece"), g("brier"), path]))
PY
)"
    echo "$LINE" >> "$SUMMARY"
  done
fi

echo "✓ Wrote $SUMMARY"
