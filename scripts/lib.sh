#!/usr/bin/env bash
# scripts/lib.sh — shared shell helpers for sparkyard operator scripts.
#
# SOURCE this file (do not execute it):  . "$REPO_ROOT/scripts/lib.sh"
#
# Provides:
#   load_env <file>  — export every KEY=val from <file> into the environment.
#                      Existence-guarded: a missing file is a silent no-op, so
#                      callers that REQUIRE a file keep their own check. This is
#                      the robust idiom (set -a; . file; set +a) that P5
#                      converged on, replacing the fragile export $(grep|xargs).
#   Color vars RED GREEN YELLOW BLUE BOLD NC — set to ANSI codes when stdout is
#                      a TTY, empty otherwise (so pipes/logs stay free of escape
#                      codes). Always defined, so they are safe under `set -u`.

load_env() {
  local file="$1"
  if [ -f "$file" ]; then
    # Preserve the caller's allexport (-a) state: only turn it back off if it
    # wasn't already set when we were called.
    local _had_allexport=0
    case "$-" in *a*) _had_allexport=1 ;; esac
    set -a
    # shellcheck source=/dev/null
    . "$file"
    [ "$_had_allexport" = 1 ] || set +a
  fi
}

if [ -t 1 ]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[1;33m'
  BLUE=$'\033[0;34m'
  BOLD=$'\033[1m'
  NC=$'\033[0m'
else
  RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi
