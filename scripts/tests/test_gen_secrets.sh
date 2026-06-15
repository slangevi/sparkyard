#!/usr/bin/env bash
# Test harness for gen-secrets.sh. Runs in a temp dir; asserts behavior.
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gen-secrets.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$(dirname "$SCRIPT")/../secrets.env.example" "$TMP/secrets.env.example"

fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. Fresh run creates secrets.env with non-empty generated secrets.
( cd "$TMP" && bash "$SCRIPT" )
[ -f "$TMP/secrets.env" ] || fail "secrets.env not created"
grep -q '^LITELLM_MASTER_KEY=sk-..*' "$TMP/secrets.env" || fail "master key not generated"
grep -Eq '^POSTGRES_PASSWORD=.{16,}$' "$TMP/secrets.env" || fail "postgres pw not generated"
grep -Eq '^LITELLM_UI_PASSWORD=.{16,}$' "$TMP/secrets.env" || fail "ui pw not generated"

# 2. User-supplied secrets remain blank (not fabricated).
grep -q '^HF_TOKEN=$' "$TMP/secrets.env" || fail "HF_TOKEN should remain blank"

# 2b. Derived/extra secrets are produced.
grep -Eq '^WEBUI_SECRET_KEY=.{16,}$' "$TMP/secrets.env" || fail "WEBUI_SECRET_KEY not generated"
grep -q '^DATABASE_URL=postgresql://litellm_admin:..*@litellm-db:5432/litellm$' "$TMP/secrets.env" || fail "DATABASE_URL not constructed"
mk="$(grep '^LITELLM_MASTER_KEY=' "$TMP/secrets.env" | cut -d= -f2-)"
oak="$(grep '^OPENAI_API_KEY=' "$TMP/secrets.env" | cut -d= -f2-)"
[ -n "$mk" ] && [ "$mk" = "$oak" ] || fail "OPENAI_API_KEY must equal LITELLM_MASTER_KEY"

# 3. Idempotent: re-run does NOT change existing values.
before="$(cat "$TMP/secrets.env")"
( cd "$TMP" && bash "$SCRIPT" )
after="$(cat "$TMP/secrets.env")"
[ "$before" = "$after" ] || fail "re-run changed existing secrets (not idempotent)"

# 4. File mode is 600.
mode="$(stat -c '%a' "$TMP/secrets.env")"
[ "$mode" = "600" ] || fail "secrets.env mode is $mode, expected 600"

# 5. Reconcile: a pre-existing secrets.env missing a newly-added example key
#    gets that key appended + filled, deriving from the file's EXISTING values.
TMP2="$(mktemp -d)"
cp "$(dirname "$SCRIPT")/../secrets.env.example" "$TMP2/secrets.env.example"
# Simulate an OLD secrets.env (predates WEBUI_SECRET_KEY/DATABASE_URL/OPENAI_API_KEY)
# with the base secrets already set:
grep -vE '^(WEBUI_SECRET_KEY|DATABASE_URL|OPENAI_API_KEY)=' "$TMP2/secrets.env.example" > "$TMP2/secrets.env"
sed -i 's|^LITELLM_MASTER_KEY=$|LITELLM_MASTER_KEY=sk-EXISTINGKEY12345678|' "$TMP2/secrets.env"
sed -i 's|^POSTGRES_PASSWORD=$|POSTGRES_PASSWORD=EXISTINGPGPW1234567890|' "$TMP2/secrets.env"
( cd "$TMP2" && bash "$SCRIPT" )
grep -Eq '^WEBUI_SECRET_KEY=.{16,}$' "$TMP2/secrets.env" || fail "reconcile: WEBUI_SECRET_KEY not appended/filled"
grep -q '^OPENAI_API_KEY=sk-EXISTINGKEY12345678$' "$TMP2/secrets.env" || fail "reconcile: OPENAI_API_KEY not mirrored from existing master key"
grep -q '^DATABASE_URL=postgresql://litellm_admin:EXISTINGPGPW1234567890@litellm-db:5432/litellm$' "$TMP2/secrets.env" || fail "reconcile: DATABASE_URL not built from existing values"
rm -rf "$TMP2"

# 6. set_if_blank handles sed/shell-special chars in a value (e.g. a custom
#    POSTGRES_PASSWORD with | & \ flowing into the derived DATABASE_URL).
TMP3="$(mktemp -d)"
cp "$(dirname "$SCRIPT")/../secrets.env.example" "$TMP3/secrets.env.example"
grep -v '^POSTGRES_PASSWORD=' "$TMP3/secrets.env.example" > "$TMP3/secrets.env"
printf 'POSTGRES_PASSWORD=%s\n' 'p|a&s\s' >> "$TMP3/secrets.env"
( cd "$TMP3" && bash "$SCRIPT" )   # must NOT crash
grep -qF 'DATABASE_URL=postgresql://litellm_admin:p|a&s\s@litellm-db:5432/litellm' "$TMP3/secrets.env" \
  || fail "special-char password not preserved verbatim in DATABASE_URL"
rm -rf "$TMP3"

# 7. A minimal secrets.webui.env (only the 2 keys Open WebUI needs) is written, mode 600.
[ -f "$TMP/secrets.webui.env" ] || fail "secrets.webui.env not created"
grep -Eq '^OPENAI_API_KEY=.+$' "$TMP/secrets.webui.env" || fail "secrets.webui.env missing OPENAI_API_KEY"
grep -Eq '^WEBUI_SECRET_KEY=.+$' "$TMP/secrets.webui.env" || fail "secrets.webui.env missing WEBUI_SECRET_KEY"
[ "$(grep -cE '^[A-Z_]+=' "$TMP/secrets.webui.env")" = "2" ] || fail "secrets.webui.env should hold exactly 2 keys"
wmode="$(stat -c '%a' "$TMP/secrets.webui.env")"; [ "$wmode" = "600" ] || fail "secrets.webui.env mode is $wmode"

# 8. Least-privilege subset files for litellm-db + litellm: ONLY the keys each
#    service needs; NEVER HF_TOKEN / REGISTRY_TOKEN (no secret bleed into the
#    stock postgres / gateway containers). mode 600.
[ -f "$TMP/secrets.db.env" ] || fail "secrets.db.env not created"
grep -q '^POSTGRES_PASSWORD=' "$TMP/secrets.db.env" || fail "secrets.db.env missing POSTGRES_PASSWORD"
if grep -Eq '^(HF_TOKEN|REGISTRY_TOKEN|LITELLM_MASTER_KEY)=' "$TMP/secrets.db.env"; then fail "secrets.db.env leaks a non-postgres secret"; fi
[ "$(grep -cE '^[A-Z_]+=' "$TMP/secrets.db.env")" = "3" ] || fail "secrets.db.env should hold exactly 3 keys"
dbmode="$(stat -c '%a' "$TMP/secrets.db.env")"; [ "$dbmode" = "600" ] || fail "secrets.db.env mode is $dbmode"

[ -f "$TMP/secrets.litellm.env" ] || fail "secrets.litellm.env not created"
grep -q '^LITELLM_MASTER_KEY=' "$TMP/secrets.litellm.env" || fail "secrets.litellm.env missing master key"
grep -q '^DATABASE_URL=' "$TMP/secrets.litellm.env" || fail "secrets.litellm.env missing DATABASE_URL"
if grep -Eq '^(HF_TOKEN|REGISTRY_TOKEN|POSTGRES_PASSWORD)=' "$TMP/secrets.litellm.env"; then fail "secrets.litellm.env leaks a non-litellm secret"; fi
[ "$(grep -cE '^[A-Z_]+=' "$TMP/secrets.litellm.env")" = "4" ] || fail "secrets.litellm.env should hold exactly 4 keys"
lmode="$(stat -c '%a' "$TMP/secrets.litellm.env")"; [ "$lmode" = "600" ] || fail "secrets.litellm.env mode is $lmode"

echo "PASS: gen-secrets.sh"
