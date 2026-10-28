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

### `vllm-node:latest` — (built 2026-10-26) from source

| Component  | Git commit  | Built artifact |
|------------|-------------|----------------|
| vLLM       | `5df63a720` | `0.32.1.dev0+g5df63a720.d20261026` |
| FlashInfer | `946200de1` | v0.7.0.post1 tag, `flashinfer-python 0.7.0.post1` (built with cubins) |
| PyTorch    | —           | `2.13.0+cu130` |
| Transformers | —         | `5.19.0` |

The vLLM commit is the v0.32.0 tag. The package reports `0.32.1.dev0` because
the build clone has no tags, so setuptools-scm cannot see `v0.32.0`.

Built from source with
`./build-and-copy.sh -j 8 --vllm-ref 5df63a720 --flashinfer-ref v0.7.0.post1`
(spark-vllm-docker `0bdc9c94`). `-j 8` keeps peak RAM bounded; the default 16
jobs exhausted memory on the FlashInfer stage. Total build time about 1 h 51 min.
`sparkyard vllm-node` passes the same `--vllm-ref` and `--flashinfer-ref` from
settings, so `make vllm-node` reproduces this image. It does not pass `-j`; on
this box, run the command above inside the clone to keep the job count at 8.

**Known issue: Gemma-4-26B-A4B-NVFP4 is pinned to `vllm-node:rollback-040700aaa`.**
On this image, vLLM's startup memory profiling of Gemma's vision encoder
(eager HF SDPA) pushes host memory far past the 0.28 gmem plan. Available RAM
fell from about 91 GiB to 9–15 GiB, and an unguarded load OOM-hung the box on
2026-10-26. The old image runs the same step with the same settings (16
max-size dummy images, 32768-token budget) and bottoms out at 53 GiB available,
so 0.32 uses about 40 GiB more there. The image count is not the cause, and
setting `video: 0` or `--max-num-batched-tokens 8192` did not help. No upstream
fix exists as of 2026-10-27. The pin is in the gitignored `models.yaml` (Gemma's
`image:`); retest on the next vllm-node update and drop the pin if it passes.
All the other on-disk vLLM models pass a smoke load on this image, including a
double load of Qwen3-Coder-Next.

Previous image (vLLM `040700aaa`, FlashInfer 0.6.18, `--use-wheels`, 2026-08-23)
is kept as `vllm-node:rollback-040700aaa` on both nodes.

## Reproduce

The one-command path (clones the repo, checks out the pinned ref, builds):

From prebuilt wheels (minutes, and the wheel set is already dependency-resolved;
the wheel set picks its own vLLM build, so it will not match the pinned ref):

```bash
sparkyard vllm-node --variant base --use-wheels
```

From source at the pinned ref:

```bash
make vllm-node                 # vllm-node:latest (settings pin)
make vllm-node VARIANT=mxfp4   # vllm-node-mxfp4:latest (tracks its own ref)
make vllm-node VLLMARGS="--print"   # dry-run the plan
```

The pins live in `settings.local.yaml`: `vllm.vllm_ref` (default `5df63a720`)
and `vllm.flashinfer_ref` (`v0.7.0.post1` unless overridden; source builds
only). This file mirrors them. Under the hood `make vllm-node` runs, inside the
clone:

```bash
./build-and-copy.sh --vllm-ref 5df63a720 --flashinfer-ref v0.7.0.post1   # -> vllm-node:latest
./build-and-copy.sh --exp-mxfp4                    # -> vllm-node-mxfp4:latest (GPT-OSS-120B)
```

`build-and-copy.sh` rejects `--exp-mxfp4` combined with `--vllm-ref`, so the
mxfp4 image tracks its own pinned ref inside the clone rather than this one.

## Source of truth

The clone records each build's refs in `wheels/.vllm-commit` and
`wheels/.flashinfer-commit`; this file mirrors them. Re-record here after any
intentional `vllm-node` rebuild.
