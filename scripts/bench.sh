#!/usr/bin/env bash
# scripts/bench.sh — sparkyard's thin benchmark runner.
#   MODE=quality (default) → tool-eval-bench  (tool-calling quality, /100)
#   MODE=speed             → llama-benchy      (throughput: pp/tg)
# Discovers models from the gateway, warms each, runs the tool, writes results
# under test-results/. The benchmark engines are external, user-installed tools:
#   tool-eval-bench — https://github.com/SeraphimSerapis/tool-eval-bench
#   llama-benchy    — https://github.com/eugr/llama-benchy
# If the tool for the chosen MODE is absent, print an install hint and exit 0.
set -uo pipefail   # per-model tolerance; deliberately NOT -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$REPO_ROOT/scripts/lib.sh"

MODE="${MODE:-quality}"
BASE_URL="${BASE_URL:-http://0.0.0.0:28080}"

case "$MODE" in
  quality) TOOL=tool-eval-bench ;;
  speed)   TOOL=llama-benchy ;;
  *) echo "${RED}unknown MODE='$MODE' (use quality|speed)${NC}" >&2; exit 2 ;;
esac

# Tool presence — advisory if absent (speed may run via `uvx llama-benchy`).
tool_present() {
  command -v "$TOOL" >/dev/null 2>&1 && return 0
  [ "$MODE" = speed ] && command -v uvx >/dev/null 2>&1 && uvx llama-benchy --help >/dev/null 2>&1 && return 0
  return 1
}
if ! tool_present; then
  echo "${YELLOW}$TOOL not found — skipping $MODE bench (advisory).${NC}"
  if [ "$MODE" = quality ]; then
    echo "Install: uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git"
  else
    echo "Install: uvx llama-benchy   (https://github.com/eugr/llama-benchy)"
  fi
  exit 0
fi

MODELS=$(curl -fsS "$BASE_URL/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json; print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))' 2>/dev/null)
if [ -z "$MODELS" ]; then
  echo "${RED}no models discovered at $BASE_URL/v1/models — is the stack up?${NC}" >&2
  exit 1
fi

OUT_DIR="$REPO_ROOT/test-results/bench-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/SUMMARY.txt"; : > "$SUMMARY"
echo "${BOLD}sparkyard bench — MODE=$MODE  BASE_URL=$BASE_URL${NC}"
echo "models:"; echo "$MODELS" | sed 's/^/  - /'; echo

BENCHY="llama-benchy"; command -v llama-benchy >/dev/null 2>&1 || BENCHY="uvx llama-benchy"

while IFS= read -r M; do
  [ -z "$M" ] && continue
  echo "${BLUE}=== [$M] ===${NC}"
  safe="${M//\//_}"
  log="$OUT_DIR/${safe}.${MODE}.log"
  # Warm up: force llama-swap to load + JIT (tolerate long L-tier loads).
  payload=$(M="$M" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"messages":[{"role":"user","content":"warmup"}],"max_tokens":4}))')
  if ! curl -fsS --max-time 1800 "$BASE_URL/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$payload" -o /dev/null; then
    echo "  ${RED}warmup failed — skipping${NC}"
    printf '%-60s %s\n' "$M" "WARMUP_FAILED" >> "$SUMMARY"; continue
  fi
  if [ "$MODE" = quality ]; then
    tool-eval-bench --base-url "$BASE_URL" --model "$M" --short --no-live > "$log" 2>&1
    score=$(grep -oE 'Score:[[:space:]]+[0-9]+ / 100' "$log" | tail -1)
    echo "  ${score:-Score: ?}"
    printf '%-60s %s\n' "$M" "${score:-?}" >> "$SUMMARY"
  else
    $BENCHY --base-url "$BASE_URL/v1" --model "$M" \
      --pp 512 --tg 128 --depth 0 --runs 3 --no-warmup --skip-coherence \
      --save-result "$OUT_DIR/${safe}.md" --format md > "$log" 2>&1
    echo "  done → ${safe}.md"
    printf '%-60s %s\n' "$M" "see ${safe}.md" >> "$SUMMARY"
  fi
done <<< "$MODELS"

echo; echo "${BOLD}=== SUMMARY ($MODE) ===${NC}"; cat "$SUMMARY"
echo "results: $OUT_DIR"
