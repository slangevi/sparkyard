# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this is

sparkyard is a Docker-Compose orchestration stack plus a Python generator for
running multiple LLMs on a single **NVIDIA DGX Spark (GB10, 128 GB unified
memory, SM121 / CUDA 13.1)**. It is not a typical application: "editing code"
here means editing `models.yaml`, `settings`, Jinja templates, Dockerfiles, and
shell/Python tooling. There is no separate runtime test suite — the generator
has a pytest suite (`make test`); the stack is verified by `make render` +
`docker compose`.

## Single source of truth (read this first)

The files that actually run are **generated and gitignored**; the inputs are
committed:

| Committed template | Gitignored working copy | Generated (gitignored) output |
|---|---|---|
| `models.example.yaml` | `models.yaml` (the model SSOT; seeded by `make init`) | `llama-swap/config.yaml` |
| `settings.example.yaml` | `settings.local.yaml` (machine paths; seeded by `make init`) | `LiteLLM/config.yaml` |
| `secrets.env.example` | `secrets.env` (gitignored; `make secrets` scaffolds it) | `.env` (compose vars) |

`docker-compose.yml` is **tracked** and parameterized by the generated `.env`.
Never hand-edit the generated `llama-swap/config.yaml` / `LiteLLM/config.yaml` /
`.env` — change `models.yaml` (or `settings.local.yaml`) and run `make render`.

## The generator (`tools/sparkyard/`)

Pipeline: `placeholders` → `settings` → `model` → `validate` → `render` → `cli`,
with Jinja templates in `tools/sparkyard/templates/`. Driven by the root
`Makefile`; the venv is at `tools/.venv` (`make venv`).

- `models.yaml` entries have an `engine:` of `vllm` or `llamacpp` (the only two —
  validation is fail-closed). Each carries an optional `hf_repo` so `make
  download` / `make add-model` can fetch it. `make add-model` supports both vLLM
  (safetensors) and GGUF repos: for a GGUF repo it detects the type, presents a
  quant picker (`--gguf-file <pattern>` or an interactive numbered menu), emits a
  `llamacpp` entry with GB10 flags, and infers `ctx_size` from `config.json`.
  `make download` fetches weights for both engines.
- `{placeholder}` tokens (e.g. `{llm_root}`, `{repo_path}`) are resolved from
  settings at render time; shell `${VAR}` macros are preserved verbatim
  (negative-lookbehind in the resolver).
- vLLM model blocks render a folded `cmd: >` that calls
  `python3 /app/scripts/launch.py` with `env VAR=…` parameters; the launcher
  sizes `--gpu-memory-utilization` adaptively (see below).

## Common commands

```bash
make init        # first-run onboarding (settings + models.yaml + secrets)
make secrets     # scaffold/auto-generate secrets.env
make validate    # structural validation (fail-closed)
make render      # regenerate the live configs
make build       # build the local llama-cpp + llama-swap images
make doctor      # advisory on-disk model report
make add-model HF_REPO=<org/model> [ADDARGS=--download]
make download [MODEL=<name>]
make bench [MODE=speed]
make test        # pytest (tools/)
make lint        # shellcheck (advisory)
docker compose up -d
```

## Conventions / gotchas

- **`launch.py`** (`llama-swap/scripts/launch.py`) is the vLLM launcher: stdlib
  Python, runs inside the (minimal) llama-swap container, reads `/proc/meminfo`,
  and picks `--gpu-memory-utilization` in `[GMEM_MIN, GMEM_MAX]`. It enforces a
  `SYSTEM_RAM_CEILING_GIB` crash-guard because ~126.5 GB used RAM crashes the
  box. `--print` dry-runs it (prints the gmem + docker argv without launching).
- **Folded-`cmd:` blank-line trap:** a blank line in a `cmd: >` block becomes a
  literal newline and breaks the `env`-prefixed launcher invocation → silent OOM.
  The template structurally avoids this; don't introduce blank lines into
  generated `cmd:` blocks.
- **Container networking:** model containers spawn with
  `--network container:llama-swap` so they share llama-swap's namespace and bind
  `localhost:PORT` inside it. Never give them `network_mode: host`.
- **llama.cpp on GB10:** needs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `--no-mmap`.
- **Secrets:** one gitignored `secrets.env` is the only home for secret values;
  the generated `LiteLLM/config.yaml` references `os.environ/LITELLM_MASTER_KEY`,
  never inlining a secret.
- **Validation is fail-closed:** `make render` won't write if `models.yaml` is
  invalid (atomic writes).
- **Images:** llama-cpp and llama-swap are built locally via `make build`
  (`docker compose build`). Ollama and LiteLLM reference pinned upstream digests.
  There is no registry push and no CI build pipeline.
- **`models.yaml` is gitignored** and seeded from the committed `models.example.yaml`
  template by `make init` (mirrors how `settings.example.yaml` seeds
  `settings.local.yaml`). Edit `models.yaml` locally; never commit it.
- **LiteLLM master-key auth is ON** by default; every API call requires
  `LITELLM_MASTER_KEY`.

## Conventions for contributions

- Keep `models:` the last top-level key in `models.yaml` (the add-model
  appender relies on it).
- Add a test under `tools/tests/` for generator changes (pytest) or
  `scripts/tests/` for shell changes; `make test` runs both and must stay green.
  `make lint` shellchecks the scripts (advisory).
- More GB10 operational notes are in the README's "GB10 gotchas" section.
