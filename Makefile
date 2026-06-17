# Sparkyard operator commands.
VENV := tools/.venv
PY   := $(VENV)/bin/python

.PHONY: venv secrets validate render doctor test test-sh add-model download bench init lint build vllm-node update start stop

venv: $(VENV)/.installed

$(VENV)/.installed: tools/pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -e "./tools[dev]"
	touch $@

secrets: venv
	tools/.venv/bin/sparkyard secrets

validate: venv
	tools/.venv/bin/sparkyard validate

# Build the locally-customized images (llama-cpp + llama-swap) on this box.
build: venv
	tools/.venv/bin/sparkyard build

start: venv
	tools/.venv/bin/sparkyard start

stop: venv
	tools/.venv/bin/sparkyard stop

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

# Check for + apply upstream component updates (bumps pins, pulls/builds; leaves a diff).
#   make update                       # apply
#   make update UPDATEARGS=--check    # dry-run report
update: venv
	tools/.venv/bin/sparkyard update $(UPDATEARGS)

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
bench: venv
	tools/.venv/bin/sparkyard bench $(if $(MODE),--mode $(MODE),) $(if $(BASE_URL),--base-url $(BASE_URL),)

# One-command onboarding. `sparkyard init` seeds configs + secrets; the make
# path also builds the venv and offers a global install.
init: venv
	@tools/.venv/bin/sparkyard init
	@bash scripts/offer-global-install.sh

# Shellcheck every tracked *.sh when shellcheck is present; advisory no-op otherwise.
lint:
	@if command -v shellcheck >/dev/null 2>&1; then \
	  git ls-files '*.sh' | xargs -r shellcheck && echo "✓ shellcheck clean"; \
	else \
	  echo "shellcheck not installed — skipping (advisory)."; \
	  echo "Install: https://github.com/koalaman/shellcheck#installing"; \
	fi
