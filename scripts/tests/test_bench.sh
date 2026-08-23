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

# MODELS= filter: restrict the sweep to named models. Unscoped, bench.sh walks
# every id from /v1/models — on a 128GB unified-memory box that can mean loading
# a 78GB model, so scoping must be possible without editing the script.
FAKEBIN=$(mktemp -d); trap 'rm -rf "$FAKEBIN"' EXIT
cat > "$FAKEBIN/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in */v1/models) echo '{"data":[{"id":"alpha"},{"id":"beta"},{"id":"gamma"}]}'; exit 0;; esac; done
exit 0
EOF
cat > "$FAKEBIN/llama-benchy" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$FAKEBIN/curl" "$FAKEBIN/llama-benchy"

out=$(MODE=speed MODELS="beta" OUT_DIR="$FAKEBIN/out" PATH="$FAKEBIN:/usr/bin:/bin" bash "$BENCH" </dev/null 2>&1)
echo "$out" | grep -q "beta"   || fail "MODELS filter should keep the named model: $out"
echo "$out" | grep -q -- "- alpha" && fail "MODELS filter should drop unnamed models: $out"
echo "$out" | grep -q -- "- gamma" && fail "MODELS filter should drop unnamed models: $out"

# Unknown name is fail-closed rather than a silent no-op sweep.
MODE=speed MODELS="nope" OUT_DIR="$FAKEBIN/out2" PATH="$FAKEBIN:/usr/bin:/bin" bash "$BENCH" </dev/null >/dev/null 2>&1; rc=$?
[ "$rc" = 2 ] || fail "unknown MODELS name should exit 2 (got $rc)"

echo "PASS: bench.sh"
