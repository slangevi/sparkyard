# Changelog

All notable changes to sparkyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

sparkyard is an independent reimplementation; see [`CREDITS.md`](CREDITS.md) for
the prior work that inspired it.

## [Unreleased]

## [1.4.0] - 2026-06-22

This release makes `sparkyard update` *apply* updates for the two source-built
components — `vllm-node` and `llama-cpp` — which were previously report-only. Each
resolves its upstream default-branch HEAD, rebuilds at that ref, and persists the
pin only on a successful build. Additive and backward-compatible: the heavy builds
fire only when a component is named explicitly.

### Added

- **`sparkyard update vllm-node` / `update llama-cpp` now apply.** Each resolves
  the upstream default-branch HEAD (`vllm-project/vllm@main`,
  `ggml-org/llama.cpp@master`), reports how many commits behind it is, and — when
  named explicitly — rebuilds at that ref. `--check` / `--notes` show a real
  commit-diff; both fail-soft to a static note when GitHub is unreachable.
- **Explicit-naming gate.** A bare `sparkyard update` (no component args) reports
  the source-built components but does not trigger their (~30-min) builds; name
  one explicitly to opt into the build.
- **Build-then-persist.** A failed build writes nothing. On success, `vllm-node`
  syncs all four ref locations — `settings.local.yaml`, `settings.py`
  `DEFAULT_VLLM_REF`, and `vllm/VLLM_NODE_PROVENANCE.md` (including its
  reproduce-command refs) — sourced from the clone's recorded build artifacts.

### Changed

- **llama.cpp is now pinned.** `llama-cpp/llama-cpp.Dockerfile` gained an
  `ARG LLAMA_CPP_REF`; it previously cloned llama.cpp HEAD on every build, which
  contradicted the stack's "never floats" rule. `sparkyard update llama-cpp`
  rebuilds via `--build-arg` and bumps the ARG on a successful build.

### Fixed

- **The `llama-cpp` update component now targets the right compose service.** It
  builds the `llama-server` service; the previous note pointed at a non-existent
  `llama-cpp` service.
- **`vllm-node` build on a fresh clone.** The build no longer runs
  `git checkout <vllm_ref>` inside the `spark-vllm-docker` tooling clone — that ref
  is a vLLM commit absent from the tooling repo, so a fresh clone aborted with
  `pathspec did not match`. `build-and-copy.sh` already checks vLLM out via
  `--vllm-ref`. Verified on a DGX Spark (GB10): the base build completes and the
  resulting image loads vLLM.

## [1.3.0] - 2026-06-22

This release makes `sparkyard update` selective: scope a check or apply to one or
more named components instead of the whole stack. Additive and
backward-compatible — running `sparkyard update` with no arguments behaves exactly
as before.

### Added

- **Per-component `sparkyard update [COMPONENT]...`** — name one or more
  components to scope the check, apply, and `--notes` to just those; with no names
  it still processes everything. Valid names: `ollama`, `litellm`, `litellm-db`,
  `open-webui`, `llama-swap`, plus the report-only `llama-cpp` and `vllm-node`.
  Examples: `sparkyard update litellm --check`,
  `sparkyard update litellm open-webui`, `sparkyard update vllm-node --notes`.
  `make update UPDATEARGS="litellm --check"` forwards the same arguments.
- **Fail-closed component validation** — an unknown component name aborts with a
  clear message naming the valid set and a non-zero exit, before any registry or
  network call.

## [1.2.0] - 2026-06-21

This release enriches `sparkyard update`'s preview: `--notes` now explains what a
pending update provides and recommends whether to apply it — summarized by your
own local LiteLLM gateway, with graceful fallbacks. Additive and backward-compatible.

### Added

- **`sparkyard update --check --notes`** — summarize what each pending update
  provides, via your local LiteLLM gateway (`:14000`; no external API; stdlib only).
  Falls back to raw notes when the gateway/model is unavailable, and never changes
  `update`'s exit code.
- **Apply recommendations** — each summarized component ends with an advisory
  `Recommendation: Apply` or `Recommendation: Review first` (breaking changes,
  auth/default-behavior changes, and deprecations lean "Review first").
- **Commit-diff summaries for source-tracked components** — beyond llama-swap's
  release notes, `--notes` summarizes the commits behind a bump: litellm and
  open-webui via their OCI image revision labels (old/new digest → GitHub compare),
  and vllm-node via the pinned vLLM ref → `vllm-project/vllm@main` (with a
  large-jump caveat). Images without provenance (ollama, postgres) show a
  digest-delta + changelog one-liner.
- **`sparkyard update --model <name>`** — choose the gateway model for the
  `--notes` summary (defaults to the first model the gateway lists).

## [1.1.0] - 2026-06-17

This release turns the `make`-driven flow into a first-class `sparkyard` CLI and
broadens the engine + component tooling. Every `make` target remains as a thin
alias and nothing was removed, so upgrading is drop-in.

### Added

- **`sparkyard` CLI** — a first-class console entry point installed by `make venv`
  (or globally via `uv tool install ./tools`), covering the full operator flow:
  `init`, `secrets`, `render`, `build`, `start`/`stop`, `update`, `doctor`,
  `download`, `add-model`, `vllm-node`, `bench`, and `validate`. It autodiscovers
  the repo root by walking up to the committed `models.example.yaml` marker, so it
  runs from any subdirectory (explicit `--models`/`--settings` override). The
  `make` targets are now thin aliases over it.
- **`sparkyard update`** — preview pending upstream component updates with
  `--check`, or apply them: bump the pinned image digests and pull/build the
  stack. Pins are never floated.
- **`make vllm-node` / `sparkyard vllm-node`** — clone and build the vLLM serving
  image(s) for GB10, recording the build refs in `vllm/VLLM_NODE_PROVENANCE.md`.
- **GGUF support in `add-model` / `download`** — `add-model` detects GGUF repos,
  picks a quant (`--gguf-file <pattern>` or an interactive menu), emits a
  `llamacpp` entry, and can download the chosen quant (shard-aware) via
  `--download`. `make download` now fetches GGUF entries (previously skipped),
  including multi-part shard families; `ctx_size` is inferred from `config.json`
  when available.

### Changed

- **CLI-first documentation** — the README is rewritten around the `sparkyard`
  CLI; the `make` targets are demoted to an aliases footnote.
- **CLI internals** — the command layer is reimplemented on top of
  [click](https://click.palletsprojects.com/) (pinned `click==8.4.1`) for polished,
  consistent `--help` output and argument handling, replacing the hand-rolled
  argparse setup.

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

[1.4.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.4.0
[1.3.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.3.0
[1.2.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.2.0
[1.1.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.1.0
[1.0.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.0.0
