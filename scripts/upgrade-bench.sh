#!/usr/bin/env bash
# scripts/upgrade-bench.sh — before/after benchmark for a component upgrade.
#   usage: upgrade-bench.sh <label> <model> [model ...]
# Per model: unload everything, time a cold load, then run llama-benchy at
# pp512/pp8192 x tg128, concurrency 1 and 4, 3 runs. Results land in
# test-results/upgrade-<label>-<timestamp>/ (<model>.json + SUMMARY.txt).
# Run it once as `before` on the old images and once as `after` on the new
# ones, then compare the two folders with scripts/compare-bench.py.
#
# llama-benchy is pinned (BENCHY_VERSION) so the tool cannot change between
# the two runs. A memory watchdog unloads the model and kills vllm-* containers
# if MemAvailable drops below WATCHDOG_MIN_GIB: a runaway load on GB10 unified
# memory can hang the whole box.
set -uo pipefail   # per-model tolerance; deliberately NOT -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$REPO_ROOT/scripts/lib.sh"

BASE_URL="${BASE_URL:-http://127.0.0.1:28080}"
BENCHY_VERSION="${BENCHY_VERSION:-0.4.0}"
WATCHDOG_MIN_GIB="${WATCHDOG_MIN_GIB:-20}"
MEMINFO="${MEMINFO:-/proc/meminfo}"
UNLOAD_SETTLE="${UNLOAD_SETTLE:-15}"

if [ $# -lt 2 ]; then
  echo "usage: upgrade-bench.sh <label> <model> [model ...]" >&2
  exit 2
fi
LABEL="$1"; shift

BENCHY="llama-benchy"
if ! command -v llama-benchy >/dev/null 2>&1; then
  if command -v uvx >/dev/null 2>&1; then
    BENCHY="uvx llama-benchy@$BENCHY_VERSION"
  else
    echo "${RED}llama-benchy not found. Install: uvx llama-benchy   (https://github.com/eugr/llama-benchy)${NC}" >&2
    exit 1
  fi
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/test-results/upgrade-$LABEL-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/SUMMARY.txt"
echo "label=$LABEL out=$OUT_DIR" | tee "$SUMMARY"

avail_gib() { awk '/^MemAvailable:/{print int($2/1048576)}' "$MEMINFO"; }

MINFILE="$OUT_DIR/.minavail"; avail_gib > "$MINFILE"
WDLOG="$OUT_DIR/watchdog.log"
(
  while true; do
    a=$(avail_gib); m=$(cat "$MINFILE")
    [ "$a" -lt "$m" ] && echo "$a" > "$MINFILE"
    if [ "$a" -lt "$WATCHDOG_MIN_GIB" ]; then
      echo "$(date +%T) avail ${a}G < ${WATCHDOG_MIN_GIB}G, unloading" >> "$WDLOG"
      curl -s "$BASE_URL/unload" -o /dev/null
      docker ps --format '{{.Names}}' 2>/dev/null | grep '^vllm-' | xargs -r docker kill >/dev/null 2>&1
    fi
    sleep 1
  done
) &
WD_PID=$!
trap 'kill "$WD_PID" 2>/dev/null' EXIT

for M in "$@"; do
  curl -fsS "$BASE_URL/unload" -o /dev/null; sleep "$UNLOAD_SETTLE"
  avail_gib > "$MINFILE"
  payload=$(M="$M" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"messages":[{"role":"user","content":"hi"}],"max_tokens":4}))')
  t0=$(date +%s)
  code=$(curl -sS --max-time 1800 -o /dev/null -w '%{http_code}' "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$payload")
  load=$(( $(date +%s) - t0 ))
  if [ "$code" != 200 ]; then
    printf '%-50s LOAD_FAILED http=%s after %ss minavail=%sG\n' "$M" "$code" "$load" "$(cat "$MINFILE")" | tee -a "$SUMMARY"
    continue
  fi
  $BENCHY --base-url "$BASE_URL/v1" --model "$M" --pp 512 8192 --tg 128 --depth 0 \
    --runs 3 --concurrency 1 4 --skip-coherence --save-result "$OUT_DIR/${M//\//_}.json" --format json \
    > "$OUT_DIR/${M//\//_}.log" 2>&1
  rc=$?
  printf '%-50s load=%ss bench_rc=%s minavail=%sG\n' "$M" "$load" "$rc" "$(cat "$MINFILE")" | tee -a "$SUMMARY"
done
curl -fsS "$BASE_URL/unload" -o /dev/null
[ -s "$WDLOG" ] && { echo "${YELLOW}watchdog fired:${NC}"; cat "$WDLOG"; }
echo DONE | tee -a "$SUMMARY"
