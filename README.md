# sparkyard

An SSOT-driven, multi-engine LLM stack for the **NVIDIA DGX Spark** (Grace-Blackwell
GB10, 128 GB unified memory). One `models.yaml` plus a machine-local
`settings.local.yaml` generate every live config; you operate the stack with
`make`, and talk to it over an OpenAI- and Anthropic-compatible gateway (or a
browser UI).

## Why

Running several LLMs on one box usually means duplicated settings, secrets, and
model lists smeared across an `.env`, a compose file, two `config.yaml`s, and a
pile of shell scripts. sparkyard has a **single source of truth**: edit
`models.yaml` + `settings.local.yaml`, run `make render`, and the llama-swap,
LiteLLM, and compose `.env` files are regenerated. Secrets live in exactly one
gitignored `secrets.env`. Adding a model is one command.

## Prerequisites

- An NVIDIA DGX Spark (GB10 / SM121, CUDA 13.1) with Docker + the NVIDIA
  Container Toolkit (the `nvidia` runtime).
- Python 3 (for the generator; `make venv` creates a local virtualenv under `tools/.venv`).
- The custom stack images, built on the box with `make build` (llama-cpp +
  llama-swap); Ollama and LiteLLM pull pinned upstream images automatically.
- **For vLLM models only:** the `vllm-node` image, built from source for SM121.
  This is a heavy one-time build (~30 min) via an external clone — see
  [`vllm/VLLM_NODE_PROVENANCE.md`](vllm/VLLM_NODE_PROVENANCE.md). llama.cpp (GGUF)
  models don't need it — they run on the `make build` images, so they're the
  lighter first-run path.

## Quickstart

```bash
git clone https://github.com/slangevi/sparkyard.git
cd sparkyard

make init                 # copies settings.example.yaml -> settings.local.yaml,
                          # models.example.yaml -> models.yaml, scaffolds secrets.env
# edit settings.local.yaml (llm_root + repo paths)
# edit models.yaml         (keep/trim the example models, or add your own)
# edit secrets.env         (HF_TOKEN only for gated/private models — get one at
#                           huggingface.co/settings/tokens; the rest auto-generate)

make render               # generate .env + llama-swap/config.yaml + LiteLLM/config.yaml
make download             # fetch weights for models.yaml entries that carry hf_repo
make build                # build the local llama-cpp + llama-swap images
docker compose up -d      # start the stack
```

A model only serves once its weights are on disk: `make download` fetches the
vLLM example weights; GGUF weights are fetched by hand (see [Adding a
model](#adding-a-model)). vLLM models also need the `vllm-node` image (see
Prerequisites); the llama.cpp GGUF example is the lighter first run. `make
doctor` reports which models' weights are present.

The gateway requires auth — call it with your `LITELLM_MASTER_KEY` (from
`secrets.env`) as the API key, or open the browser UI at `http://localhost:3000`
(first signup becomes admin):

```bash
curl http://localhost:14000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "hello"}]}'
```

## Make commands

| Command | What it does |
|---|---|
| `make init` | First-run onboarding (idempotent): seed `settings.local.yaml`, `models.yaml`, and `secrets.env`. |
| `make venv` | Create `tools/.venv` and install the generator's deps (auto-run by the targets below). |
| `make secrets` | Scaffold `secrets.env` and auto-generate the random secrets. |
| `make validate` | Structurally validate `models.yaml` + settings (fail-closed). |
| `make render` | Regenerate the live configs from the SSOT. |
| `make build` | Build the local llama-cpp + llama-swap images. |
| `make doctor` | Advisory on-disk report: which models' weights are present. |
| `make add-model HF_REPO=<org/model> [ADDARGS=--download]` | Introspect a HF repo, append a vLLM entry to `models.yaml`, render, optionally download. |
| `make download [MODEL=<name>]` | Fetch HF weights for `models.yaml` entries that carry `hf_repo`. |
| `make bench [MODE=speed]` | Benchmark each served model — quality (tool-eval-bench) or speed (llama-benchy). |
| `make test` | Run the test suite (generator pytest + shell tests). |
| `make lint` | Shellcheck the shell scripts (advisory if shellcheck isn't installed). |

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
  seeded from committed `models.example.yaml` by `make init`) +
  `settings.local.yaml` (gitignored; seeded from committed `settings.example.yaml`)
  are the inputs; `make render` produces the live `llama-swap/config.yaml`,
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

Secrets live in one gitignored `secrets.env`; `make secrets` projects
least-privilege subsets so the database and gateway containers never receive
`HF_TOKEN`. Never paste the output of `docker compose config` into a shared
channel — it expands secrets. Rotate any token that may have leaked.

## Adding a model

```bash
make add-model HF_REPO=Qwen/Qwen3-8B ADDARGS=--download
make render && docker compose up -d llama-swap litellm
```

The wizard introspects the HF repo, proposes a conservative vLLM entry (you
confirm), appends it to `models.yaml`, records its `hf_repo`, and — with
`--download` — fetches the weights. GGUF models are added to `models.yaml` by hand.

`models.yaml` is gitignored and seeded from the committed `models.example.yaml`
template by `make init`.

## GB10 / unified-memory gotchas

The DGX Spark's memory is **unified** (shared between CPU and GPU), which makes a
few things non-obvious:

- **~126.5 GB of used RAM crashes the box.** `launch.py` reserves headroom
  against a `SYSTEM_RAM_CEILING_GIB` (default 117.81 GiB) so it never plans past
  that, even when `/proc/meminfo` reports more free.
- **Don't hand-edit the generated `llama-swap/config.yaml`** — it's regenerated
  from `models.yaml` by `make render`. (The generator emits each launcher
  invocation as one folded `cmd:` line and is test-guarded against a YAML
  blank-line pitfall that would otherwise split it.)
- **llama.cpp needs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` and `--no-mmap`** (mmap is
  unsafe on unified memory).
- **Large models (120B+) need a generous `ready_timeout`** (1800s+); a 90 GB weight
  load far exceeds the default.

## License & credits

MIT — see [`LICENSE`](LICENSE). sparkyard is an independent reimplementation
inspired by prior DGX Spark work; see [`CREDITS.md`](CREDITS.md) for inspiration,
the tools it wraps, and the upstream projects it builds on.
