#!/usr/bin/env bash
# Test harness for scripts/upgrade-bench.sh + scripts/compare-bench.py —
# usage, a faked end-to-end run, the watchdog, and the comparison table.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$DIR/upgrade-bench.sh"
COMPARE="$DIR/compare-bench.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

[ -f "$BENCH" ] || fail "upgrade-bench.sh not found at $BENCH"
bash -n "$BENCH" || fail "bash -n"

# No models → usage, exit 2 (an unscoped sweep would load every model).
bash "$BENCH" before </dev/null >/dev/null 2>&1; rc=$?
[ "$rc" = 2 ] || fail "missing models should exit 2 (got $rc)"

FAKE=$(mktemp -d); trap 'rm -rf "$FAKE"' EXIT
mkdir -p "$FAKE/bin"
cat > "$FAKE/bin/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in */unload) echo unload >> "$FAKE_CALLS"; exit 0;; esac; done
for a in "$@"; do [ "$a" = "%{http_code}" ] && { printf 200; exit 0; }; done
exit 0
EOF
# Fake llama-benchy: write a minimal result to the --save-result path.
cat > "$FAKE/bin/llama-benchy" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do [ "$1" = --save-result ] && out="$2"; shift; done
cat > "$out" <<J
{"benchmarks":[{"prompt_size":512,"concurrency":1,"pp_throughput":{"mean":${FAKE_PP}},"tg_throughput":{"mean":${FAKE_TG}}}]}
J
EOF
cat > "$FAKE/bin/docker" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$FAKE/bin/"*
printf 'MemTotal: 127535748 kB\nMemAvailable: 94371840 kB\n' > "$FAKE/meminfo"

run() {  # run <label> <pp> <tg> <outdir> <models...>
  local label=$1 pp=$2 tg=$3 out=$4; shift 4
  FAKE_CALLS="$FAKE/calls" FAKE_PP=$pp FAKE_TG=$tg OUT_DIR="$out" UNLOAD_SETTLE=0 \
    MEMINFO="$FAKE/meminfo" PATH="$FAKE/bin:/usr/bin:/bin" bash "$BENCH" "$label" "$@" </dev/null 2>&1
}

out=$(run before 1000 50 "$FAKE/before" alpha beta)
[ -f "$FAKE/before/alpha.json" ] || fail "should save alpha.json: $out"
[ -f "$FAKE/before/beta.json" ]  || fail "should save beta.json: $out"
grep -q "^alpha .*load=.*bench_rc=0 minavail=90G" "$FAKE/before/SUMMARY.txt" || fail "summary row: $(cat "$FAKE/before/SUMMARY.txt")"
grep -q "^DONE" "$FAKE/before/SUMMARY.txt" || fail "summary should end with DONE"
[ -s "$FAKE/before/watchdog.log" ] && fail "watchdog should stay quiet at 90G"

# Watchdog: MemAvailable below the floor → it unloads and logs.
: > "$FAKE/calls"
printf 'MemAvailable: 10485760 kB\n' > "$FAKE/meminfo"
UNLOAD_SETTLE=2 run low 1000 50 "$FAKE/low" alpha >/dev/null
grep -q "avail 10G < 20G" "$FAKE/low/watchdog.log" 2>/dev/null || fail "watchdog should fire at 10G"
printf 'MemAvailable: 94371840 kB\n' > "$FAKE/meminfo"

# compare-bench.py: +10% prompt, +20% gen, and a model only in one run.
run after 1100 60 "$FAKE/after" alpha gamma >/dev/null
table=$(python3 -I "$COMPARE" "$FAKE/before" "$FAKE/after") || fail "compare-bench.py failed"
echo "$table" | grep -q "| alpha |.*| pp512 c1 | 1,000 → 1,100 (+10%) | 50.0 → 60.0 (+20%) |" || fail "alpha row: $table"
echo "$table" | grep -q "| beta | not run after" || fail "beta should be 'not run after': $table"
echo "$table" | grep -q "| gamma | not run before" || fail "gamma should be 'not run before': $table"

echo "PASS: upgrade-bench.sh"
