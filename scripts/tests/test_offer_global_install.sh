#!/usr/bin/env bash
# Test harness for offer-global-install.sh. Runs in a temp dir with stubbed uv
# and/or pipx on a controlled PATH; asserts each branch. The interactive [y/N]
# TTY branch is not exercised here (no pty); its install action is covered via
# SPARKYARD_INSTALL_YES.
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/offer-global-install.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

mkdir -p "$TMP/tools"
: > "$TMP/tools/pyproject.toml"

# Controlled PATH: the externals used by the script and its stubs (grep, cat,
# dirname) plus bash to invoke the script — so tool presence is deterministic
# regardless of the host's real PATH.
BIN="$TMP/bin"; mkdir -p "$BIN"
for t in grep cat dirname bash; do ln -s "$(command -v "$t")" "$BIN/$t"; done

LOG="$TMP/cmd.log"; export LOG

# Stub for uv: logs argv; `tool list` emits sparkyard only when UV_HAS=1.
make_uv() {
  cat > "$BIN/uv" <<'STUB'
#!/usr/bin/env bash
echo "uv $*" >> "$LOG"
[ "$1 $2" = "tool list" ] && [ "${UV_HAS:-0}" = "1" ] && echo "sparkyard v1.0.0"
exit 0
STUB
  chmod +x "$BIN/uv"
}
# Stub for pipx: logs argv; `list --short` emits sparkyard only when PIPX_HAS=1.
make_pipx() {
  cat > "$BIN/pipx" <<'STUB'
#!/usr/bin/env bash
echo "pipx $*" >> "$LOG"
case "$1" in
  list) [ "${PIPX_HAS:-0}" = "1" ] && echo "sparkyard 1.0.0" ;;
esac
exit 0
STUB
  chmod +x "$BIN/pipx"
}

run() {
  ( cd "$TMP" && PATH="$BIN" UV_HAS="${UV_HAS:-0}" PIPX_HAS="${PIPX_HAS:-0}" \
      SPARKYARD_INSTALL_YES="${SPARKYARD_INSTALL_YES:-0}" \
      bash "$SCRIPT" </dev/null ) 2>&1
}

# 1. neither uv nor pipx → silent, no install.
rm -f "$BIN/uv" "$BIN/pipx"; : > "$LOG"
out="$(run)" || fail "case1: nonzero exit"
echo "$out" | grep -q "detected" && fail "case1: should be silent (no installer)"
[ -s "$LOG" ] && fail "case1: no installer should be called"

# 2. uv present, already installed → note, no install.
make_uv; rm -f "$BIN/pipx"; : > "$LOG"
out="$(UV_HAS=1 run)" || fail "case2: nonzero exit"
echo "$out" | grep -q "already installed via uv" || fail "case2: expected uv already-installed note"
grep -qF "uv tool install ./tools" "$LOG" && fail "case2: must not install when present"

# 3. uv present, not installed, SPARKYARD_INSTALL_YES=1 → uv tool install.
make_uv; rm -f "$BIN/pipx"; : > "$LOG"
out="$(UV_HAS=0 SPARKYARD_INSTALL_YES=1 run)" || fail "case3: nonzero exit"
grep -qF "uv tool install ./tools" "$LOG" || fail "case3: expected 'uv tool install ./tools'"

# 4. BOTH present, not installed, INSTALL_YES=1 → uv chosen, pipx NOT used.
make_uv; make_pipx; : > "$LOG"
out="$(UV_HAS=0 PIPX_HAS=0 SPARKYARD_INSTALL_YES=1 run)" || fail "case4: nonzero exit"
grep -qF "uv tool install ./tools" "$LOG" || fail "case4: expected uv chosen over pipx"
grep -qF "pipx install ./tools" "$LOG" && fail "case4: pipx must not be used when uv present"

# 5. uv absent, pipx present, not installed, non-interactive → tip names pipx, no install.
rm -f "$BIN/uv"; make_pipx; : > "$LOG"
out="$(PIPX_HAS=0 run)" || fail "case5: nonzero exit"
echo "$out" | grep -qF "pipx install ./tools" || fail "case5: expected pipx tip"
grep -qF "pipx install ./tools" "$LOG" && fail "case5: must not install non-interactively"

# 6. uv present, not installed, non-interactive → tip names uv, no install.
make_uv; rm -f "$BIN/pipx"; : > "$LOG"
out="$(UV_HAS=0 run)" || fail "case6: nonzero exit"
echo "$out" | grep -qF "uv tool install ./tools" || fail "case6: expected uv tip"
grep -qF "uv tool install ./tools" "$LOG" && fail "case6: must not install non-interactively"

echo "PASS: offer-global-install.sh"
