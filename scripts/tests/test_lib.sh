#!/usr/bin/env bash
# Test harness for scripts/lib.sh. Sources the lib; asserts load_env + colors.
set -euo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

[ -f "$LIB" ] || fail "lib.sh not found at $LIB"
# shellcheck source=/dev/null
. "$LIB"

# 1. load_env exports KEY=val pairs from a file into the environment.
printf 'SPARKYARD_TEST_VAR=hello\nSPARKYARD_TEST_TWO=2\n' > "$TMP/env"
load_env "$TMP/env"
[ "${SPARKYARD_TEST_VAR:-}" = "hello" ] || fail "load_env did not export SPARKYARD_TEST_VAR"
[ "${SPARKYARD_TEST_TWO:-}" = "2" ]     || fail "load_env did not export SPARKYARD_TEST_TWO"

# 2. load_env on a missing file is a silent no-op (must not error under set -e).
( set -e; load_env "$TMP/does-not-exist" ) || fail "load_env errored on a missing file"

# 3. Every color var is DEFINED (set, though possibly empty on a non-TTY).
for c in RED GREEN YELLOW BLUE BOLD NC; do
  [ -n "${!c+x}" ] || fail "color var $c is not defined"
done

echo "PASS: lib.sh"
