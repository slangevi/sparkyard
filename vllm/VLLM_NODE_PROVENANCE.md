# vllm-node image provenance

The ephemeral vLLM images that serve every model in this stack are built from
the external clone `vllm/build/spark-vllm-docker` (a clone of
`eugr/spark-vllm-docker`). That directory is gitignored, so this file pins the
exact upstream refs the in-use images were built from, making the build
reproducible from this repo.

## Images
- `vllm-node:latest` — standard SM121 (GB10) vLLM build; serves most models.
- ~~`vllm-node-tf5:latest`~~ — **retired 2026-08-24.** It existed to add transformers
  v5 on top of the base build. Upstream made v5 the default and reduced `--tf5` to a
  tag alias that "no longer alter[s] dependency resolution", so it produced a
  byte-equivalent duplicate of base at the cost of a full Rust toolchain compile.
  Verified identical before removal: same vLLM, torch 2.13.0, transformers 5.15.1,
  flashinfer 0.6.18. Point any `vllm-node-tf5` entry at `vllm-node:latest`.
- `vllm-node-mxfp4:latest` — CUTLASS MXFP4 variant (GPT-OSS-120B); built on demand,
  tracks its own ref inside the clone (see note below).

## Pinned refs

### `vllm-node:latest` — built 2026-08-23 (`--use-wheels`)

| Component  | Git commit  | Built artifact |
|------------|-------------|----------------|
| vLLM       | `040700aaa` | `vllm-0.26.1rc1.dev1105+g040700aaa.d20260822-cp312-cp312-linux_aarch64.whl` |
| FlashInfer | (wheel)     | `flashinfer_cubin-0.6.18`, `flashinfer_jit_cache-0.6.18-cp39-abi3-manylinux_2_28_aarch64`, `flashinfer_python-0.6.18` |
| PyTorch    | —           | `2.13.0+cu130` |
| Transformers | —         | `5.15.1` |

Built from published wheels rather than from source, so the vLLM ref above is the
commit the wheel was produced from, not one this repo compiled. A source build at
that ref uses the same upstream source but is not bit-identical.

## Reproduce

The one-command path (clones the repo, checks out the pinned ref, builds):

From prebuilt wheels (minutes, and the wheel set is already dependency-resolved —
this is how `vllm-node:latest` above was produced):

```bash
sparkyard vllm-node --variant base --use-wheels
```

From source at the pinned ref:

```bash
make vllm-node                 # vllm-node:latest (settings pin)
make vllm-node VARIANT=mxfp4   # vllm-node-mxfp4:latest (tracks its own ref)
make vllm-node VLLMARGS="--print"   # dry-run the plan
```

The pin lives in `settings.local.yaml` (`vllm.vllm_ref`, default `7852e50e4`);
this file mirrors it. Under the hood `make vllm-node` runs, inside the clone:

```bash
./build-and-copy.sh --vllm-ref 7852e50e4          # -> vllm-node:latest
./build-and-copy.sh --exp-mxfp4                    # -> vllm-node-mxfp4:latest (GPT-OSS-120B)
```

`build-and-copy.sh` rejects `--exp-mxfp4` combined with `--vllm-ref`, so the
mxfp4 image tracks its own pinned ref inside the clone rather than this one.

## Source of truth

The clone records each build's refs in `wheels/.vllm-commit` and
`wheels/.flashinfer-commit`; this file mirrors them. Re-record here after any
intentional `vllm-node` rebuild.
