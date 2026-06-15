#!/usr/bin/env bash
# Test harness for scripts/bench.sh — syntax + advisory-absent + bad-mode paths.
set -uo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bench.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

[ -f "$BENCH" ] || fail "bench.sh not found at $BENCH"
bash -n "$BENCH" || fail "bash -n"

# Tools absent (restricted PATH) → quality + speed are advisory (exit 0 + hint).
out=$(MODE=quality PATH="/usr/bin:/bin" bash "$BENCH" </dev/null 2>&1); rc=$?
[ "$rc" = 0 ] || fail "quality should exit 0 when tool absent (got $rc): $out"
echo "$out" | grep -qi "tool-eval-bench not found" || fail "quality should print an install hint"

out=$(MODE=speed PATH="/usr/bin:/bin" bash "$BENCH" </dev/null 2>&1); rc=$?
[ "$rc" = 0 ] || fail "speed should exit 0 when tool absent (got $rc): $out"

# Unknown MODE → exit 2.
MODE=bogus bash "$BENCH" </dev/null >/dev/null 2>&1; rc=$?
[ "$rc" = 2 ] || fail "unknown MODE should exit 2 (got $rc)"

echo "PASS: bench.sh"
