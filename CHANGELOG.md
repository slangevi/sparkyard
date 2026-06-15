# Changelog

All notable changes to sparkyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

sparkyard is an independent reimplementation; see [`CREDITS.md`](CREDITS.md) for
the prior work that inspired it.

## [Unreleased]

### Added
- `make add-model` accepts GGUF repos: detects them, picks a quant
  (`--gguf-file <pattern>` or an interactive menu), emits a `llamacpp` entry, and
  can download the chosen quant (shard-aware) via `--download`.
- `make download` fetches GGUF entries (previously skipped), including multi-part
  shard families; `ctx_size` is inferred from `config.json` when available.

## [1.0.0] - 2026-06-15

First public release: a single-source-of-truth, multi-engine LLM stack for the
NVIDIA DGX Spark (GB10).

### Added

- **SSOT generator** (`tools/sparkyard/`): a committed `models.example.yaml`
  (seeded to a gitignored `models.yaml` by `make init`) + machine-local
  `settings.local.yaml` generate the live `llama-swap/config.yaml`,
  `LiteLLM/config.yaml`, and compose `.env` via `make render`. Validation is
  fail-closed; writes are atomic.
- **`make` operator flow**: `init`, `secrets`, `validate`, `render`, `doctor`,
  `add-model`, `download`, `bench`, `test`, `lint`.
- **One secrets home, least-privilege delivery**: a single gitignored
  `secrets.env` (`make secrets` scaffolds + auto-generates); the generated LiteLLM
  config references the master key by env, never inlining secrets. `make secrets`
  projects per-service least-privilege subsets (`secrets.db.env`,
  `secrets.litellm.env`, `secrets.webui.env`) so no container receives a secret it
  does not need — `HF_TOKEN` never reaches a runtime container.
- **LiteLLM gateway** (`:14000`): OpenAI-compatible plus an Anthropic
  `/v1/messages` endpoint (for Claude Code / agents); master-key auth enforced.
- **Open WebUI** (`:3000`) for browser chat over the gateway.
- **Adaptive vLLM launcher** (`llama-swap/scripts/launch.py`): stdlib Python,
  sizes `--gpu-memory-utilization` from `/proc/meminfo` with a GB10 crash-guard;
  `--print` dry-run mode.
- **`make add-model`**: introspect a HF repo → propose + append a vLLM entry →
  render → optional download. **`make download`**: fetch weights for entries with
  `hf_repo`. **`make bench`**: quality (tool-eval-bench) / speed (llama-benchy).
- **Build-local images**: the custom `llama-cpp` + `llama-swap` images build on
  the box (`make build`, digest-pinned base layers); Ollama and LiteLLM reference
  pinned upstream digests directly; the `vllm-node` build refs are recorded in
  `vllm/VLLM_NODE_PROVENANCE.md`. No registry round-trip.
- **Hardened network exposure**: only the authenticated LiteLLM gateway (`:14000`,
  master key) and Open WebUI (`:3000`, login) are published off-box; the inference
  engines bind `127.0.0.1` and Postgres has no host port. Service healthchecks +
  `service_healthy` startup ordering.

[1.0.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.0.0
