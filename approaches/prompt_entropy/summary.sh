#!/usr/bin/env bash
set -euo pipefail

# Where your per-key outputs live, e.g.:
# results/prompt_entropy/gpt2_all_keys/mean/prompt_entropy_mean_*.json
ROOT="${1:-results/prompt_entropy/Qwen/Qwen2.5-3B-Instruct_all_keys}"
SUMMARY="${2:-$ROOT/summary.csv}"

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
)

# CSV header
echo "key,accuracy,auroc,auroc_flipped,average_precision,tpr@1%fpr,tpr@0.1%fpr,ece,brier,prob_transform,json_path" > "$SUMMARY"

for KEY in "${KEYS[@]}"; do
  DIR="$ROOT/$KEY"

  # Find newest json for this key
  JSON_PATH="$(ls -t "$DIR"/*.json 2>/dev/null | head -n 1 || true)"

  if [[ -z "${JSON_PATH}" ]]; then
    echo "WARN: no json found for key=$KEY in $DIR" >&2
    echo "$KEY,,,,,,,,,,," >> "$SUMMARY"
    continue
  fi

  # Extract metrics from JSON robustly (no jq dependency)
  LINE="$(python - <<'PY' "$JSON_PATH" "$KEY"
import json, sys
path = sys.argv[1]
key = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

# metrics are stored under payload["metrics"] in my compute script
m = d.get("metrics", {})

def get(name, default=""):
    v = m.get(name, default)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

fields = [
    key,
    get("accuracy"),
    get("auroc"),
    get("auroc_flipped"),
    get("average_precision"),
    get("tpr@1%fpr"),
    get("tpr@0.1%fpr"),
    get("ece"),
    get("brier"),
    get("prob_transform"),
    path
]
# Escape commas in prob_transform if any (unlikely); wrap that field in quotes
fields[9] = '"' + fields[9].replace('"', '""') + '"'
print(",".join(fields))
PY
)"

  echo "$LINE" >> "$SUMMARY"
done

echo "✓ Wrote $SUMMARY"
