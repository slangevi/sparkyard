# sparkyard

An SSOT-driven, multi-engine LLM stack for the **NVIDIA DGX Spark** (Grace-Blackwell
GB10, 128 GB unified memory). One `models.yaml` plus a machine-local
`settings.local.yaml` generate every live config; you operate the stack with
the `sparkyard` CLI, and talk to it over an OpenAI- and Anthropic-compatible
gateway (or a browser UI).

## Why

Running several LLMs on one box usually means duplicated settings, secrets, and
model lists smeared across an `.env`, a compose file, two `config.yaml`s, and a
pile of shell scripts. sparkyard has a **single source of truth**: edit
`models.yaml` + `settings.local.yaml`, run `sparkyard render`, and the llama-swap,
LiteLLM, and compose `.env` files are regenerated. Secrets live in exactly one
gitignored `secrets.env`. Adding a model is one command.

## Prerequisites

- An NVIDIA DGX Spark (GB10 / SM121, CUDA 13.1) with Docker + the NVIDIA
  Container Toolkit (the `nvidia` runtime).
- Python 3 (for the generator). Install the `sparkyard` command with `uv tool
  install ./tools` (or `pipx install ./tools`); `make venv` instead creates a
  project-local virtualenv at `tools/.venv` with the same console script.
- The custom stack images, built on the box with `sparkyard build` (llama-cpp +
  llama-swap); Ollama and LiteLLM pull pinned upstream images automatically.
- **For vLLM models only:** the `vllm-node` image, built from source for SM121.
  Run `sparkyard vllm-node` — it clones the external build repo and builds the image
  (a heavy one-time build, ~30 min; pin tracked in `settings.local.yaml` /
  [`vllm/VLLM_NODE_PROVENANCE.md`](vllm/VLLM_NODE_PROVENANCE.md)). llama.cpp
  (GGUF) models don't need it — they run on the `sparkyard build` images, so they're
  the lighter first-run path.

## Quickstart

```bash
git clone https://github.com/slangevi/sparkyard.git && cd sparkyard

uv tool install ./tools        # global `sparkyard` (or: pipx install ./tools)
sparkyard init                 # seed settings.local.yaml + models.yaml + secrets.env
# edit settings.local.yaml (paths) + models.yaml + secrets.env (HF_TOKEN)

sparkyard render               # generate .env + llama-swap/LiteLLM configs
sparkyard download             # fetch weights for models.yaml entries with hf_repo
sparkyard build                # build the local llama-cpp + llama-swap images
sparkyard vllm-node            # only if you run vLLM models (~30 min)
sparkyard start                # docker compose up -d
```

Every step is a `sparkyard` subcommand — `make` is optional (see below). `sparkyard stop` tears the stack down; `sparkyard update --check` previews component updates.

A model only serves once its weights are on disk: `sparkyard download` fetches weights
for both vLLM and GGUF entries (any entry with `hf_repo`). vLLM models also need
the `vllm-node` image (see Prerequisites); the llama.cpp GGUF example is the
lighter first run. `sparkyard doctor` reports which models' weights are present.

The gateway requires auth — call it with your `LITELLM_MASTER_KEY` (from
`secrets.env`) as the API key, or open the browser UI at `http://localhost:3000`
(first signup becomes admin):

```bash
curl http://localhost:14000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "hello"}]}'
```

## Commands

`sparkyard <cmd>` is the primary interface — install it with `uv tool install
./tools` (or `pipx install ./tools`), or use `tools/.venv/bin/sparkyard` after
`make venv`. It autodiscovers the repo root via the committed
`models.example.yaml` marker, so it runs from anywhere in the checkout (pass
`--models`/`--settings` to override).

| Command | What it does |
|---|---|
| `sparkyard init` | First-run onboarding (idempotent): seed `settings.local.yaml`, `models.yaml`, and `secrets.env`. |
| `sparkyard secrets` | Scaffold `secrets.env` and generate the random secrets (also run by `init`). |
| `sparkyard validate` | Structurally validate `models.yaml` + settings (fail-closed). |
| `sparkyard render` | Regenerate the live configs from the SSOT. |
| `sparkyard build` | Build the local llama-cpp + llama-swap images. |
| `sparkyard vllm-node [--variant mxfp4] [--print]` | Clone + build the vLLM serving image(s) for SM121 (base + tf5 by default). |
| `sparkyard start` / `sparkyard stop` | Start (`docker compose up -d`) / stop (`docker compose down`) the stack. |
| `sparkyard doctor` | Advisory on-disk report: which models' weights are present. |
| `sparkyard add-model <org/model> [--download] [--gguf-file <pat>]` | Introspect a HF repo (vLLM or GGUF), append an entry to `models.yaml`, render, optionally download. |
| `sparkyard download [--model <name>]` | Fetch HF weights for `models.yaml` entries that carry `hf_repo`. |
| `sparkyard update [COMPONENT]... [--check] [--notes]` | Check for + apply upstream component updates (`--check` = dry-run; `--notes` summarizes release/commit notes — llama-swap, litellm, open-webui, vllm-node — via your local gateway, each with an Apply/Review-first recommendation). Name one or more components (`ollama litellm litellm-db open-webui llama-swap llama-cpp vllm-node`) to scope the run; no names = all. |
| `sparkyard bench [--mode speed]` | Benchmark each served model — quality (tool-eval-bench) or speed (llama-benchy). |

**`make` aliases:** every command above also works as `make <cmd>` (e.g. `make
render`, `make start`) for muscle memory. `make venv` (bootstrap the project venv
+ install the console script), `make test`, and `make lint` are make-only
contributor tooling.

## Architecture

```
client → LiteLLM (:14000 → 4000) → llama-swap (127.0.0.1:28080 → 8080) → on-demand model container
                                                                 ├ vLLM      (safetensors, via launch.py)
                                                                 ├ llama.cpp (GGUF)
                                                                 └ Ollama
browser → Open WebUI (:3000) → LiteLLM
```

- **LiteLLM** is the single OpenAI/Anthropic-compatible gateway; it routes by
  model name to llama-swap over the Docker network.
- **llama-swap** spawns and evicts model containers on demand (it holds the
  Docker socket); each model idles out after its `ttl`. vLLM models are launched
  by `llama-swap/scripts/launch.py`, which sizes `--gpu-memory-utilization`
  adaptively from `/proc/meminfo`.
- The **SSOT generator** lives in `tools/sparkyard/`. `models.yaml` (gitignored;
  seeded from committed `models.example.yaml` by `sparkyard init`) +
  `settings.local.yaml` (gitignored; seeded from committed `settings.example.yaml`)
  are the inputs; `sparkyard render` produces the live `llama-swap/config.yaml`,
  `LiteLLM/config.yaml`, and `.env` (all gitignored generated artifacts).
  `docker-compose.yml` is tracked and parameterized by the generated `.env`.

## Ports

| Service | Host → container | Notes |
|---|---|---|
| LiteLLM | 14000 → 4000 | OpenAI/Anthropic gateway; authenticated (master key) |
| Open WebUI | 3000 → 8080 | browser chat; own login |
| llama-swap | 127.0.0.1:28080 → 8080 | model orchestrator; loopback only |
| Ollama | 127.0.0.1:11434 | loopback only |
| llama.cpp (persistent) | 127.0.0.1:19000 | always-on GGUF server; loopback only |
| LiteLLM Postgres | — (internal network) | no host port |

## Security

`docker compose up` exposes only two ports off-box, both authenticated:

| Port | Service | Auth |
|------|---------|------|
| 14000 | LiteLLM gateway (OpenAI/Anthropic API) | `LITELLM_MASTER_KEY` (sk-* on every call) |
| 3000  | Open WebUI | own login (first signup = admin) |

The inference engines and database are **not** published to the network:

- llama-swap (28080), Ollama (11434), llama.cpp (19000) bind `127.0.0.1` only.
- Postgres is reachable only on the internal compose network (no host port).

To reach the gateway/UI from another machine, point the client at the DGX's IP
on 14000/3000. To deliberately expose an internal engine (e.g. llama-swap to a
trusted host), change its bind from `127.0.0.1:` back to a host/`0.0.0.0` port —
and put authentication in front of it first.

Secrets live in one gitignored `secrets.env`; `sparkyard secrets` projects
least-privilege subsets so the database and gateway containers never receive
`HF_TOKEN`. Never paste the output of `docker compose config` into a shared
channel — it expands secrets. Rotate any token that may have leaked.

## Adding a model

```bash
sparkyard add-model Qwen/Qwen3-8B --download
sparkyard render && sparkyard start
```

The wizard introspects the HF repo, proposes a conservative vLLM entry (you
confirm), appends it to `models.yaml`, records its `hf_repo`, and — with
`--download` — fetches the weights.

`models.yaml` is gitignored and seeded from the committed `models.example.yaml`
template by `sparkyard init`.

### GGUF models

`sparkyard add-model <org/model-GGUF>` detects a GGUF repo and lets you pick a
quant:

- `sparkyard add-model bartowski/Qwen2.5-3B-Instruct-GGUF --gguf-file Q4_K_M`
  selects by substring; omit `--gguf-file` to get an interactive menu.
- The wizard emits a `llamacpp` entry (GB10 flags `--no-mmap` + unified memory,
  `--jinja`) and infers `ctx_size` from the repo's `config.json` when present,
  else defaults to 8192 with a warning to adjust.
- `--gguf-file Q4_K_M --download` (or a later `sparkyard download`) fetches
  the chosen quant — including all shards of a multi-part quant.

`sparkyard download` now fetches GGUF entries (any entry with `hf_repo`), not just
vLLM ones. Assumes repos keep `.gguf` files at the repo root (the common layout);
quants nested in subdirectories need manual placement.

Note: `sparkyard add-model` places GGUF weights under `{llm_root}/gguf/<org>/<model>/`; the shipped `models.example.yaml` entry instead reuses the shared `{llm_root}/ollama/` tree, so hand-added and auto-added GGUFs may live in different directories.

## GB10 / unified-memory gotchas

The DGX Spark's memory is **unified** (shared between CPU and GPU), which makes a
few things non-obvious:

- **~126.5 GB of used RAM crashes the box.** `launch.py` reserves headroom
  against a `SYSTEM_RAM_CEILING_GIB` (default 117.81 GiB) so it never plans past
  that, even when `/proc/meminfo` reports more free.
- **Don't hand-edit the generated `llama-swap/config.yaml`** — it's regenerated
  from `models.yaml` by `sparkyard render`. (The generator emits each launcher
  invocation as one folded `cmd:` line and is test-guarded against a YAML
  blank-line pitfall that would otherwise split it.)
- **llama.cpp needs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` and `--no-mmap`** (mmap is
  unsafe on unified memory).
- **llama-cpp is pinned to a recorded commit ref** (in `llama-cpp/llama-cpp.Dockerfile`)
  rather than floating to HEAD; run `sparkyard update llama-cpp` to check for new commits
  and rebuild at the latest `ggml-org/llama.cpp@master`.
- **Large models (120B+) need a generous `ready_timeout`** (1800s+); a 90 GB weight
  load far exceeds the default.

## License & credits

MIT — see [`LICENSE`](LICENSE). sparkyard is an independent reimplementation
inspired by prior DGX Spark work; see [`CREDITS.md`](CREDITS.md) for inspiration,
the tools it wraps, and the upstream projects it builds on.
