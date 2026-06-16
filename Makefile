# Sparkyard operator commands.
VENV := tools/.venv
PY   := $(VENV)/bin/python

.PHONY: venv secrets validate render doctor test test-sh add-model download bench init lint build vllm-node

venv: $(VENV)/.installed

$(VENV)/.installed: tools/pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -e "./tools[dev]"
	touch $@

secrets:
	bash scripts/gen-secrets.sh

validate: venv
	tools/.venv/bin/sparkyard validate

# Build the locally-customized images (llama-cpp + llama-swap) on this box.
build:
	docker compose build

render: venv
	tools/.venv/bin/sparkyard render

doctor: venv
	tools/.venv/bin/sparkyard doctor

test: venv
	cd tools && .venv/bin/python -m pytest -q
	@$(MAKE) test-sh

# Run the shell behavioral tests (lib, bench, gen-secrets).
test-sh:
	@for t in scripts/tests/*.sh; do echo "# $$t"; bash "$$t" || exit 1; done

add-model: venv
	tools/.venv/bin/sparkyard add-model "$(HF_REPO)" $(ADDARGS)

download: venv
	tools/.venv/bin/sparkyard download $(if $(MODEL),--model "$(MODEL)",)

# Clone + build the externally-sourced vLLM serving image(s) for SM121 (GB10).
# Default builds vllm-node + vllm-node-tf5 at the settings pin (~30 min).
#   make vllm-node                       # base + tf5
#   make vllm-node VARIANT=mxfp4         # GPT-OSS-120B variant (opt-in)
#   make vllm-node VLLMARGS="--print"    # dry-run the plan
#   make vllm-node VLLMARGS="--vllm-ref abc1234"
vllm-node: venv
	tools/.venv/bin/sparkyard vllm-node \
	  $(if $(VARIANT),--variant $(VARIANT),) $(VLLMARGS)

# Thin benchmark over the live gateway. MODE=quality (default) | speed.
bench:
	MODE=$(MODE) BASE_URL=$(BASE_URL) bash scripts/bench.sh

# One-command onboarding for the SSOT flow. Idempotent — safe to re-run.
init:
	@test -f settings.local.yaml || { cp settings.example.yaml settings.local.yaml; \
	  echo "→ created settings.local.yaml (edit it to set your paths)"; }
	@test -f models.yaml || { cp models.example.yaml models.yaml; \
	  echo "→ created models.yaml from models.example.yaml (edit it / add your models)"; }
	@$(MAKE) secrets
	@$(MAKE) venv
	@bash scripts/offer-global-install.sh
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit settings.local.yaml (paths) + models.yaml (your models) + secrets.env (HF_TOKEN)"
	@echo "  2. make render          # generate .env + llama-swap/LiteLLM configs"
	@echo "  3. make build           # build the local llama-cpp + llama-swap images"
	@echo "  4. docker compose up -d"
	@echo ""
	@echo "The generator is installed at tools/.venv/bin/sparkyard (or run it via make)."

# Shellcheck every tracked *.sh when shellcheck is present; advisory no-op otherwise.
lint:
	@if command -v shellcheck >/dev/null 2>&1; then \
	  git ls-files '*.sh' | xargs -r shellcheck && echo "✓ shellcheck clean"; \
	else \
	  echo "shellcheck not installed — skipping (advisory)."; \
	  echo "Install: https://github.com/koalaman/shellcheck#installing"; \
	fi
