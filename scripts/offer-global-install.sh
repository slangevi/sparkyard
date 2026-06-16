#!/usr/bin/env bash
# scripts/offer-global-install.sh — if a Python tool installer (uv or pipx) is
# present, offer to install a global `sparkyard` command. Advisory + idempotent;
# MUST never fail `make init` (always exits 0).
set -uo pipefail   # NOT -e: a failed install must not abort init.

# repo-root sanity: same cwd requirement as gen-secrets.sh — called from the
# repo root by `make init`.
[ -f tools/pyproject.toml ] || exit 0

# shellcheck source=/dev/null
. "$(dirname "$0")/lib.sh" 2>/dev/null || true   # TTY-aware colors (optional)

# Pick a tool installer: prefer uv, fall back to pipx; neither → silent no-op.
if command -v uv >/dev/null 2>&1; then
  TOOL=uv
  installed() { uv tool list 2>/dev/null | grep -q '^sparkyard '; }
  do_install() { uv tool install ./tools; }
  INSTALL_CMD="uv tool install ./tools"
  UPDATE_HINT="uv tool install --reinstall ./tools"
elif command -v pipx >/dev/null 2>&1; then
  TOOL=pipx
  installed() { pipx list --short 2>/dev/null | grep -q '^sparkyard '; }
  do_install() { pipx install ./tools; }
  INSTALL_CMD="pipx install ./tools"
  UPDATE_HINT="pipx install --force ./tools"
else
  exit 0
fi

# already installed? → idempotent note, no prompt. If the list command itself
# fails (e.g. a corrupted tool env), `installed` is false and we fall through to
# offering the install — a safe fallback (the installer reports an already-
# installed package itself).
if installed; then
  echo "${GREEN:-}✓ global \`sparkyard\` already installed via ${TOOL}.${NC:-}"
  exit 0
fi

cat <<EOF

${BOLD:-}${TOOL} detected.${NC:-} A global \`sparkyard\` command lets you run
render / validate / add-model from anywhere — and is handy for agents/scripts —
without the \`tools/.venv/bin/\` prefix. It installs the same package into an
isolated ${TOOL} environment; the in-repo \`tools/.venv\` editable install stays
your dev copy (refresh the global copy later with \`${UPDATE_HINT}\`).
EOF

run_install() {
  if do_install; then
    echo "${GREEN:-}✓ installed — run \`sparkyard --help\`.${NC:-}"
  else
    echo "${YELLOW:-}${TOOL} install failed — retry later: ${INSTALL_CMD}${NC:-}"
  fi
}

if [ "${SPARKYARD_INSTALL_YES:-0}" = "1" ]; then
  run_install                                    # automation / test opt-in
elif [ -t 0 ]; then
  printf 'Install it globally now with %s? [y/N] ' "$TOOL"
  read -r ans || ans=""
  case "$ans" in
    [yY]*) run_install ;;
    *)     echo "Skipped. Later: ${INSTALL_CMD}" ;;
  esac
else
  echo "tip: \`${INSTALL_CMD}\` for a global \`sparkyard\`."
fi
exit 0
