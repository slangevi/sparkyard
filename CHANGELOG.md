# Changelog

All notable changes to sparkyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

sparkyard is an independent reimplementation; see [`CREDITS.md`](CREDITS.md) for
the prior work that inspired it.

## [Unreleased]

## [1.9.0] - 2026-08-24

The theme the last release kept hitting — steps that report success without
having done the thing — turns out to have applied to the configuration itself.
Refreshing this box's model lineup meant trying to actually load models that had
only ever been *configured*, and three knobs turned out to be decorative: a
per-model `ready_timeout:` that llama-swap has no key for and silently ignored,
a KV-cache estimate that counted Mamba blocks as attention layers, and no way at
all to say "keep this small model loaded". A 50 GB model failed by 88 seconds
against a ceiling nothing in `models.yaml` could raise.

### Added

- **`groups:` in `models.yaml`** — an optional top-level key, mirroring
  llama-swap's `routing.router.settings.groups` one-for-one, that says which
  models may stay loaded together. Without it llama-swap swaps one model at a
  time, which makes an always-on model impossible: every embedding call from
  Open WebUI's RAG would evict the loaded generation model and pay a full
  reload. Membership is validated fail-closed against model names (llama-swap
  silently ignores a group naming an unknown member, which presents as "my
  model never stays loaded"), aliases are rejected, and a model may belong to at
  most one group. Absent the key the rendered config is unchanged.

### Fixed

- **Launcher KV sizing for `layers_block_type` models.** `launch.py` understood
  the hybrid layouts spelled as `layer_types`, `full_attention_interval`, and
  `hybrid_override_pattern`, but not the per-block list form
  (`layers_block_type: [mamba, moe, attention, ...]`) that Nemotron-3.5 and
  Nemotron-3-Puzzle ship. Every block counted as an attention layer, sizing the
  KV cache ~8x too large (Lightning-30B: 26 GiB estimated vs ~3 GiB actual) and
  inflating `--gpu-memory-utilization` accordingly.
- **A model slower than 900 s to load could never start.** llama-swap v251 has
  exactly one ready-wait control — the *global* `healthCheckTimeout` — and no
  `readyTimeout` key at either level, so the per-model `ready_timeout:` in
  `models.yaml` was rendered into a key llama-swap ignores and every model was
  really gated by a hardcoded global 900. A 50 GB NVFP4 model with
  `ready_timeout: 1800` died mid-load with
  `health check timed out after 15m0s`, which reads as a broken model rather
  than a too-short timeout. `healthCheckTimeout` is now derived from the slowest
  model in the set, the dead `readyTimeout` keys are gone, and LiteLLM's
  `request_timeout` is derived from the same figure plus headroom so the gateway
  cannot give up while a model is still loading.
- **`nemotron_h_puzzle` configs could not launch at all.** Puzzle (NAS-derived)
  configs carry no `num_hidden_layers` — the block list is the only layer count
  present — so `parse_config` bailed out with
  `FATAL: could not parse layers/heads/head_dim`. The layer count is now derived
  from `layers_block_type` when the scalar is absent.

## [1.8.0] - 2026-08-24

Ergonomics, all of it earned the hard way: each command here replaces a
workaround that was typed by hand repeatedly during one GB10 bring-up. The
running theme is the same one the previous two releases kept hitting — steps
that reported success without having done the thing.

### Added

- **`sparkyard reload`** — re-render, then restart the services that read the
  generated config, waiting for *all* of them to report healthy. `start` is
  `docker compose up -d`, which cannot see a bind-mounted config file change, so
  a render was never live until llama-swap and litellm were restarted by hand —
  and waiting on llama-swap alone races LiteLLM into "connection reset by peer".
- **`sparkyard models`** — list what is configured: engine, context, gmem range,
  whether weights are on disk, and the aliases callers actually address.
  `--json`, `--on-disk`, `--engine`. Previously this required reading
  `models.yaml` by hand; `doctor` only reported it as a side effect.
- **`sparkyard explain <model>`** — dry-run a model's launcher and print the
  gmem plan (`attn_layers`, weights, KV, resolved `gpu_memory_utilization`)
  without loading anything. `--argv` adds the docker command. Fail-closed on an
  unknown name, and explicit that llamacpp entries have no launcher to dry-run.
- **`sparkyard bench --model`** (repeatable) replaces reaching for the `MODELS=`
  environment variable, which was the odd one out among the flags.

### Changed

- **`sparkyard update` no longer implies its pins are live.** It bumps pins and
  pulls/builds but never recreates containers, so the stack keeps running the old
  images; the success message now says so and points at `sparkyard reload`.

## [1.7.1] - 2026-08-24

Maintenance: every component pin brought current, and two fixes that came out of
doing it — one in our own error reporting, one an upstream deprecation that would
have failed silently on GB10.

### Fixed

- **llama.cpp: `--no-mmap` replaced with `--load-mode none`.** Upstream deprecated
  `--mmap`/`--no-mmap` in favour of `--load-mode`. GB10 requires non-mmap loading
  (mmap is unsafe on unified memory), so the flag could not simply be dropped.
  Measured first: with `--load-mode auto` the model loads and the GGUF is *not*
  mapped (0 `.gguf` entries in `/proc/<pid>/maps` against 95 `.so`), so `auto`
  does detect the device correctly — but the requirement is pinned explicitly
  rather than left to upstream detection, since a regression there would be
  silent. The generator emitted the deprecated flag for **every** llamacpp model,
  so the fix spans the Jinja template as well as `docker-compose.yml`; README and
  CLAUDE.md, which both documented `--no-mmap` as required, are corrected.

### Changed

- **`sparkyard update` reports why a digest resolution failed.** The resolver
  exception already carried the registry's explanation; `plan_image_updates`
  discarded it and printed the bare status `error`. An expired ghcr.io token in
  the operator's docker config (offered where no credential was needed, answered
  with 403) left two components on stale pins for two releases while the table
  said nothing more than `error`. The reason is now printed under the row,
  collapsed to one line and length-capped so a multi-line registry error cannot
  wreck the table.

### Dependencies

- `ollama` `05b6fe51` → `57d60e68`; `litellm` `8402d237` → `5ead13ed`;
  `litellm-db` `df7bca00` → `fe0737ba` (postgres 15.19); `open-webui`
  `7f1b0a1a` → `6a773e5c`; `llama-swap` v224 → **v251** (27 releases).
- `llama-cpp` rebuilt at `c060ca97` — 840 commits, now level with master. All ten
  flags the compose passes were checked against the new binary before adopting,
  and both on-disk GGUF models verified serving.

## [1.7.0] - 2026-08-24

Two corrections that only surfaced once a second, larger model was exercised on
GB10 — one to guidance that had been over-generalised from a single measurement,
one to a build variant that upstream had quietly turned into a no-op.

### Removed

- **The `tf5` build variant.** It existed to layer Transformers v5 on top of the
  base image. Upstream made v5 the default and reduced `--tf5` to a tag alias
  that, in its own words, "no longer alter[s] dependency resolution" — so it
  built a byte-equivalent duplicate of `base` at the cost of a full Rust
  toolchain compile (~30 min, versus ~11 for base). Verified identical before
  removal: same vLLM commit, torch 2.13.0, transformers 5.15.1, flashinfer
  0.6.18. `DEFAULT_VARIANTS` is now `["base"]`, so the default build no longer
  produces the duplicate, and `--variant tf5` is rejected. Point any entry
  pinned to `vllm-node-tf5:latest` at `vllm-node:latest`.

### Fixed

- **The README's peak-memory formula is labelled as a steady-state estimate.** It
  was fitted on one 27B model varying only `gmem`, and presented as general
  guidance. A second model contradicts it: 34.9GiB of weights at `gmem 0.40`
  settled at 58.7GB (close to the predicted 65GB) but peaked at **95GB during
  load** — ~36GB above its own steady state, where the 21.8GiB model showed
  almost no spike. The load transient is what actually crashes a unified-memory
  box, so the section now separates resident footprint from load peak and says
  the transient has not been isolated.

## [1.6.0] - 2026-08-24

Fallout from a GB10 bring-up that cost two machine crashes. The through-line is
things that reported success, did nothing, or described state they no longer
matched — so most of this is about making the stack tell the truth about itself.

### Added

- **`sparkyard doctor` warns on `--load-format fastsafetensors`.** It buffers the
  whole weight set while CUDA reserves the gmem budget; measured +32GiB peak
  (86 vs 54) on a 21.8GiB model with one flag changed. On unified memory both
  allocations come out of the same pool, so this can cross the crash threshold
  mid-load on a model that otherwise fits.
- **`sparkyard doctor` warns when a model's weights exceed its gmem budget.**
  `gmem.max` caps what vLLM may reserve and vLLM reserves the whole fraction, so
  weights that do not fit inside it cannot load — cheap to catch on disk rather
  than ten minutes into a cold start.
- **`sparkyard update vllm-node --use-wheels`.** Rebuilds from the published
  wheel set (~10 min) instead of compiling at upstream HEAD (~30 min). Because
  wheels install whatever version they carry rather than a requested ref, the
  build cannot land on HEAD — so the message says so, and the pin is taken from
  the *built* artifacts rather than the resolved head. Without that, the three
  synced ref locations would record a commit the image does not contain. The
  `--check` gate and the explicit-naming gate are unchanged.

### Changed

- **README documents the unified-memory reservation model.**
  `gpu_memory_utilization` is a reservation, not a ceiling: peak RAM tracks it
  linearly (`~20.5GB + gmem x 111.51GB`, measured on a 128GB GB10). Lowering a
  model's KV estimate does not lower its footprint — it only changes which gmem
  the launcher picks.
- **`--use-wheels` and `MODELS=` are documented** in the README command table and
  CLAUDE.md, including their tradeoffs.

### Fixed

- **`vllm/VLLM_NODE_PROVENANCE.md` records what the images actually contain.** It
  described a 2026-06-11 source build that no longer ships, and claimed the two
  images shared a vLLM ref. They had diverged by four vLLM minor versions and a
  PyTorch major, since only `base` was rebuilt. Each image now has its own ref
  table and the drift is stated up front. `DEFAULT_VLLM_REF` follows the image
  that runs.

## [1.5.0] - 2026-08-23

Three bugs surfaced while bringing a new hybrid-attention model up on a DGX Spark
(GB10), plus the two capabilities their diagnosis showed were missing. The theme
is the same in each case: something reported success, or did nothing at all,
while the caller believed work had happened.

### Fixed

- **KV cache sized from attention layers, not every layer.** `auto-gmem` computed
  the cache as if all `num_hidden_layers` held a per-token KV cache. Hybrid models
  interleave linear-attention layers (Gated DeltaNet / Mamba) carrying a small
  fixed recurrent state instead, so the estimate ran high by the attention
  interval — 4x on a 64-layer/16-attending model, inflating `need` to 89.81GiB
  against a true 41.81GiB and pinning `gpu_memory_utilization` to `GMEM_MAX`.
  `parse_config` now detects the attending-layer count from `layer_types`,
  `full_attention_interval`, or `hybrid_override_pattern` (nemotron_h), falling
  back conservatively to every layer for dense models. Corroborated against
  vLLM's own accounting: it reports 4.57GiB to serve one 131072-token sequence
  where the corrected formula predicts 4.29GiB and the old one predicted 17.2GiB.
- **The `vllm-node` tooling clone now advances after fetching.** `build_plan` ran
  `git fetch` on the `spark-vllm-docker` clone and never used it, so every build
  ran from whatever HEAD the clone was created with. A clone found 101 commits
  behind upstream was missing an upstream flashinfer build fix; the build
  compiled kernels for ~17 minutes and then died on an unsatisfiable
  `flashinfer-python` / `nvidia-cutlass-dsl` resolve. A `git merge --ff-only`
  step now runs between fetch and build; `--ff-only` keeps it fail-closed so a
  diverged or dirty clone stops the build rather than silently building
  something else.
- **`bench.sh` no longer sweeps every discovered model unconditionally.** It
  benchmarked every id from `/v1/models` with no way to narrow the run. On a
  unified-memory box that is unsafe rather than merely slow: llama-swap pages in
  each model in turn, so a bench run on a 21-model gateway loads every model on
  disk one after another. It also burned a warmup timeout on each model with no
  weights on disk.

### Added

- **`sparkyard vllm-node --use-wheels`.** Builds only the runner image from
  prebuilt, pre-resolved vLLM and FlashInfer wheels instead of compiling vLLM
  from source: 11:15 against ~30 min, and it sidesteps the dependency conflict
  that fails the source path, because the published wheels are already resolved
  against each other. Not a faster route to an arbitrary `--vllm-ref` — it
  installs whatever version the wheels carry, which is the combination upstream
  has tested on this hardware.
- **`MODELS=` scoping for `sparkyard bench`.** `MODELS="a b" sparkyard bench`
  restricts the sweep to named models. Unknown names exit 2 and list what is
  available, so a typo cannot silently widen the run to everything; omitting
  `MODELS` keeps the previous behaviour.
- **`--print` reports `attn_layers`** alongside `layers`, so the sizing decision
  is inspectable without a load.

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

[1.8.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.8.0
[1.7.1]: https://github.com/slangevi/sparkyard/releases/tag/v1.7.1
[1.7.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.7.0
[1.6.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.6.0
[1.5.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.5.0
[1.4.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.4.0
[1.3.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.3.0
[1.2.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.2.0
[1.1.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.1.0
[1.0.0]: https://github.com/slangevi/sparkyard/releases/tag/v1.0.0
