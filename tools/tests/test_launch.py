import importlib.util
import os
import textwrap

import pytest

# Import the standalone launcher by path (it is intentionally NOT a package module).
_LAUNCH = os.path.join(os.path.dirname(__file__), "..", "..",
                       "llama-swap", "scripts", "launch.py")
_spec = importlib.util.spec_from_file_location("launch", _LAUNCH)
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def _meminfo(tmp_path, total_kb, avail_kb, free_kb):
    p = tmp_path / "meminfo"
    p.write_text(f"MemTotal:       {total_kb} kB\n"
                 f"MemFree:        {free_kb} kB\n"
                 f"MemAvailable:   {avail_kb} kB\n")
    return str(p)


def _model_dir(tmp_path, cfg, weight_bytes):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(__import__("json").dumps(cfg))
    (d / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return str(d)


# --- g2: 2-decimal round matching awk printf "%.2f" ---
def test_g2_rounds_to_two_decimals():
    assert launch.g2(0.717384) == 0.72
    assert launch.g2(1.0) == 1.0
    assert launch.g2(13.0) == 13.0


# --- weights ---
def test_weights_gib_sums_safetensors(tmp_path):
    d = _model_dir(tmp_path, {"num_hidden_layers": 1, "num_attention_heads": 1,
                              "num_key_value_heads": 1, "hidden_size": 1, "head_dim": 1},
                   8589934592)  # exactly 8 GiB
    assert launch.weights_gib(d) == 8.0


# --- config parse: text_config nesting wins, head_dim derived, MHA fallback ---
def test_parse_config_prefers_text_config(tmp_path):
    cfg = {"num_hidden_layers": 999,
           "text_config": {"num_hidden_layers": 32, "num_attention_heads": 32,
                           "num_key_value_heads": 8, "hidden_size": 4096}}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, n_heads, n_kv, head_dim, _ = launch.parse_config(d)
    assert (n_layers, n_heads, n_kv, head_dim) == (32, 32, 8, 128)  # head_dim = 4096/32


def test_parse_config_mha_fallback_when_no_kv(tmp_path):
    cfg = {"num_hidden_layers": 16, "num_attention_heads": 16, "head_dim": 64}
    d = _model_dir(tmp_path, cfg, 1)
    _, n_heads, n_kv, head_dim, _ = launch.parse_config(d)
    assert n_kv == 16 and head_dim == 64  # n_kv defaults to n_heads


def test_parse_config_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        launch.parse_config(str(tmp_path / "nope"))


# --- gmem decision: in-range "sized" ---
def test_compute_gmem_sized_in_range(tmp_path):
    # layers=32 kv=8 head_dim=128 ctx=4096 kv_batch=4 kvb=1 -> kv=1.00 GiB
    # weights=75 -> need = 75+1+4 = 80.00
    # MemTotal 124.00 GiB, MemAvail 100.00 GiB, ceiling 117.81, overhead 6.3, buffer 5
    #   headroom = 124-117.81 = 6.19 ; total = 117.81-6.3 = 111.51
    #   free = 100 - 6.19 - 6.3 = 87.51 ; u_cap = (87.51-5)/111.51 = 0.7399...
    #   need(80) <= free(87.51) -> u = 80/111.51 = 0.7174 -> in [0.55,0.92] -> 0.72
    gmem, mode, _ = launch.compute_gmem(
        cfg_vals=(32, 32, 8, 128),
        params=launch.Params(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1,
                             gmin=0.55, gmax=0.92, safety=4.0, cuda_overhead=6.3,
                             ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0,
                             weights=75.0),
        meminfo=(124 * 1048576, 100 * 1048576, 100 * 1048576))
    assert mode == "sized" and gmem == 0.72


# --- gmem decision: fallback (need > free) clamps to gmax-or-u_cap ---
def test_compute_gmem_fallback_clamped(tmp_path):
    # need huge (weights=200) > free; u=gmax=0.92 capped at u_cap(0.7399) -> 0.74
    gmem, mode, _ = launch.compute_gmem(
        cfg_vals=(32, 32, 8, 128),
        params=launch.Params(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1,
                             gmin=0.55, gmax=0.92, safety=4.0, cuda_overhead=6.3,
                             ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0,
                             weights=200.0),
        meminfo=(124 * 1048576, 100 * 1048576, 100 * 1048576))
    assert mode == "fallback" and gmem == 0.74


# --- gmem decision: ERR when free below floor ---
def test_compute_gmem_err_below_floor():
    # tiny MemAvailable -> u_cap < gmin -> ERR
    gmem, mode, _ = launch.compute_gmem(
        cfg_vals=(32, 32, 8, 128),
        params=launch.Params(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1,
                             gmin=0.55, gmax=0.92, safety=4.0, cuda_overhead=6.3,
                             ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0,
                             weights=8.0),
        meminfo=(124 * 1048576, 20 * 1048576, 20 * 1048576))
    assert mode == "ERR" and gmem is None


# --- argv: basic, no PRE_LAUNCH ---
def test_build_argv_basic(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    env = launch.LaunchEnv(model_path="/models/vllm/X", model_host_path="/models/vllm/X",
                           container_name="vllm-x-9000", image="vllm-node:latest",
                           port="9000", host="0.0.0.0",
                           max_model_len=131072, max_num_seqs=10,
                           extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="vllm serve")
    argv = launch.build_argv("0.80", env, ["--kv-cache-dtype", "fp8"])
    assert argv[:6] == ["docker", "run", "--rm", "--name", "vllm-x-9000", "--runtime"]
    assert "-v" in argv and "/data/LLMs:/models" in argv
    # vllm args, in order, after the image
    i = argv.index("vllm-node:latest")
    assert argv[i + 1:] == ["vllm", "serve", "/models/vllm/X",
                            "--host", "0.0.0.0", "--port", "9000",
                            "--gpu-memory-utilization", "0.80",
                            "--max-model-len", "131072", "--max-num-seqs", "10",
                            "--kv-cache-dtype", "fp8"]


# --- argv: vLLM's cache dir persists across --rm launches ---
def _cache_env():
    return launch.LaunchEnv(model_path="/m", model_host_path="/m", container_name="c",
                            image="img", port="1", host="h", max_model_len=1, max_num_seqs=1,
                            extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="vllm serve")


def test_build_argv_mounts_vllm_cache_volume_by_default(monkeypatch):
    # torch.compile artifacts and FlashInfer's autotune results live under
    # /root/.cache/vllm; without a mount every cold load redoes ~7 min of work.
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    monkeypatch.delenv("VLLM_CACHE_VOLUME", raising=False)
    argv = launch.build_argv("0.50", _cache_env(), [])
    i = argv.index("sparkyard-vllm-cache:/root/.cache/vllm")
    assert argv[i - 1] == "-v"
    assert i < argv.index("img")


def test_build_argv_vllm_cache_volume_override(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    monkeypatch.setenv("VLLM_CACHE_VOLUME", "other-cache")
    argv = launch.build_argv("0.50", _cache_env(), [])
    assert "other-cache:/root/.cache/vllm" in argv


def test_build_argv_vllm_cache_volume_empty_disables(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    monkeypatch.setenv("VLLM_CACHE_VOLUME", "")
    argv = launch.build_argv("0.50", _cache_env(), [])
    assert not any(a.endswith(":/root/.cache/vllm") for a in argv)


# --- argv: empty VLLM_SERVE_PREFIX drops the prefix ---
def test_build_argv_empty_prefix(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    env = launch.LaunchEnv(model_path="/m", model_host_path="/m", container_name="c",
                           image="img", port="1", host="h", max_model_len=1, max_num_seqs=1,
                           extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="")
    argv = launch.build_argv("0.50", env, [])
    i = argv.index("img")
    assert argv[i + 1] == "/m"  # no "vllm serve" before the model path


# --- argv: PRE_LAUNCH_CMD wraps in bash -c with quoted exec ---
def test_build_argv_pre_launch(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    env = launch.LaunchEnv(model_path="/m", model_host_path="/m", container_name="c",
                           image="img", port="1", host="h", max_model_len=1, max_num_seqs=1,
                           extra_docker_args="", pre_launch_cmd="patch.sh", vllm_serve_prefix="vllm serve")
    argv = launch.build_argv("0.50", env, [])
    assert "--entrypoint" in argv and "/bin/bash" in argv
    assert argv[-2] == "-c"
    assert argv[-1].startswith("patch.sh && exec vllm serve /m")


def test_env_num_invalid_exits(monkeypatch):
    monkeypatch.setenv("GMEM_MIN", "notanumber")
    with pytest.raises(SystemExit):
        launch._env_num("GMEM_MIN", 0.55, float)


def test_compute_gmem_total_nonpositive_is_err():
    # ceiling == cuda_overhead -> total = 0 -> clean ERR (no ZeroDivisionError)
    gmem, mode, _ = launch.compute_gmem(
        cfg_vals=(32, 32, 8, 128),
        params=launch.Params(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1,
                             gmin=0.55, gmax=0.92, safety=4.0, cuda_overhead=6.3,
                             ceiling=6.3, kv_batch_realistic=4, free_buffer=5.0,
                             weights=8.0),
        meminfo=(124 * 1048576, 100 * 1048576, 100 * 1048576))
    assert mode == "ERR" and gmem is None


def test_parse_config_accepts_float_head_dim(tmp_path):
    cfg = {"num_hidden_layers": 16, "num_attention_heads": 16,
           "num_key_value_heads": 4, "hidden_size": 1024, "head_dim": 64.0}
    d = _model_dir(tmp_path, cfg, 1)
    _, _, n_kv, head_dim, _ = launch.parse_config(d)
    assert n_kv == 4 and head_dim == 64


def test_build_argv_requires_llm_root(monkeypatch):
    monkeypatch.delenv("LLM_ROOT_PATH", raising=False)
    env = launch.LaunchEnv(model_path="/m", model_host_path="/m", container_name="c",
                           image="img", port="1", host="h", max_model_len=1, max_num_seqs=1,
                           extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="vllm serve")
    with pytest.raises(SystemExit):
        launch.build_argv("0.50", env, [])


def test_read_meminfo_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        launch.read_meminfo(str(tmp_path / "nope"))


# --- hybrid attention: only full-attention layers hold a KV cache -------------
# Qwen3.5/3.6/3.8 (`qwen3_5`) interleave Gated DeltaNet ("linear_attention")
# with a full-attention layer every `full_attention_interval`. Sizing KV off
# num_hidden_layers overestimates by that interval (4x on Qwen3.8-27B) and
# drives gpu_memory_utilization high enough to OOM a 128 GB unified-memory box.

def test_parse_config_counts_only_full_attention_layers_from_layer_types(tmp_path):
    layer_types = ["linear_attention", "linear_attention",
                   "linear_attention", "full_attention"] * 16   # 64 layers, 16 full
    cfg = {"text_config": {"num_hidden_layers": 64, "num_attention_heads": 24,
                           "num_key_value_heads": 4, "head_dim": 256,
                           "layer_types": layer_types}}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (64, 16)


def test_parse_config_counts_only_full_attention_layers_from_interval(tmp_path):
    # No layer_types list; the interval alone must be honoured.
    cfg = {"text_config": {"num_hidden_layers": 64, "num_attention_heads": 24,
                           "num_key_value_heads": 4, "head_dim": 256,
                           "full_attention_interval": 4}}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (64, 16)


def test_parse_config_dense_model_counts_every_layer_as_attention(tmp_path):
    cfg = {"num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "hidden_size": 4096}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (32, 32)


def test_compute_gmem_sizes_kv_from_attention_layers_only():
    # Qwen3.8-27B shape: 64 layers / 16 full-attention, 4 kv heads, head_dim 256,
    # ctx 131072, kv_batch 4, fp8 KV. All 64 layers => 64.00 GiB; 16 => 16.00 GiB.
    params = launch.Params(max_model_len=131072, max_num_seqs=4, kv_dtype_bytes=1,
                           gmin=0.45, gmax=0.70, safety=4.0, cuda_overhead=6.3,
                           ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0,
                           weights=21.81)
    meminfo = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    _, _, hybrid = launch.compute_gmem((64, 24, 4, 256, 16), params, meminfo)
    assert hybrid["kv"] == 16.00
    assert hybrid["attn_layers"] == 16


def test_compute_gmem_accepts_legacy_four_tuple_as_all_attention():
    params = launch.Params(max_model_len=131072, max_num_seqs=4, kv_dtype_bytes=1,
                           gmin=0.45, gmax=0.70, safety=4.0, cuda_overhead=6.3,
                           ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0,
                           weights=21.81)
    meminfo = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    _, _, d = launch.compute_gmem((64, 24, 4, 256), params, meminfo)
    assert d["kv"] == 64.00


# --- hybrid attention, Nemotron-H flavour ------------------------------------
# nemotron_h has no layer_types/full_attention_interval; it encodes the stack as
# hybrid_override_pattern where '*' = attention, 'M' = Mamba, '-' = MLP.

def test_parse_config_counts_attention_layers_from_hybrid_override_pattern(tmp_path):
    cfg = {"num_hidden_layers": 42, "num_attention_heads": 40,
           "num_key_value_heads": 8, "head_dim": 128,
           "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (42, 4)


def test_parse_config_pattern_without_attention_marker_falls_back_to_all_layers(tmp_path):
    # A pattern we cannot interpret must never yield 0 attention layers, which
    # would size the KV cache to nothing. Fall back to the conservative count.
    cfg = {"num_hidden_layers": 12, "num_attention_heads": 8,
           "num_key_value_heads": 8, "head_dim": 64,
           "hybrid_override_pattern": "MMMMMMMMMMMM"}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (12, 12)


# nemotron_h / nemotron_h_puzzle also ship the stack as `layers_block_type`, a
# list of 'mamba' | 'moe' | 'attention'. Puzzle configs (NAS-derived) drop
# `num_hidden_layers` entirely and the block list is the only layer count there is.

def test_parse_config_counts_attention_layers_from_layers_block_type(tmp_path):
    blocks = ["mamba", "moe"] * 23 + ["attention", "moe"] * 3
    cfg = {"num_hidden_layers": 52, "num_attention_heads": 32,
           "num_key_value_heads": 2, "head_dim": 128,
           "layers_block_type": blocks}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (52, 3)


def test_parse_config_derives_layer_count_from_layers_block_type_when_absent(tmp_path):
    # nemotron_h_puzzle: no num_hidden_layers; 88 blocks, 8 of them attention.
    blocks = (["mamba", "moe"] * 40) + (["attention"] * 8)
    cfg = {"num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
           "layers_block_type": blocks}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (88, 8)


def test_parse_config_block_list_without_attention_falls_back_to_all_layers(tmp_path):
    # Never size the KV cache to zero on an uninterpretable block list.
    cfg = {"num_hidden_layers": 8, "num_attention_heads": 8,
           "num_key_value_heads": 8, "head_dim": 64,
           "layers_block_type": ["mamba"] * 8}
    d = _model_dir(tmp_path, cfg, 1)
    n_layers, _, _, _, n_attn = launch.parse_config(d)
    assert (n_layers, n_attn) == (8, 8)


# --- reclaim: restart named containers when memory is short, then recompute ---
_P = dict(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1, gmin=0.55, gmax=0.92,
          safety=4.0, cuda_overhead=6.3, ceiling=117.81, kv_batch_realistic=4,
          free_buffer=5.0, weights=8.0)
_SHORT = (124 * 1048576, 20 * 1048576, 20 * 1048576)
_ROOMY = (124 * 1048576, 100 * 1048576, 100 * 1048576)


def test_parse_reclaim_splits_on_spaces_and_commas():
    assert launch.parse_reclaim("comfyui other-app,third") == ["comfyui", "other-app", "third"]
    assert launch.parse_reclaim("") == []
    assert launch.parse_reclaim(None) == []


def test_parse_reclaim_refuses_non_container_names():
    for bad in ("x;id", "-rm", "a/b", "$(id)"):
        with pytest.raises(SystemExit):
            launch.parse_reclaim(bad)


class _Docker:
    """Fake `docker` runner: records calls, reports which containers run."""
    def __init__(self, running):
        self.running, self.calls = set(running), []

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            return "true" if argv[-1] in self.running else "false"
        return ""


def test_reclaim_restarts_only_running_listed_containers_then_recomputes():
    docker = _Docker(running=["comfyui"])
    reads = iter([_SHORT, _ROOMY])
    gmem, mode, _ = launch.reclaim_and_recompute(
        ["comfyui", "stopped-app"], (32, 32, 8, 128), launch.Params(**_P),
        read=lambda: next(reads), run=docker, sleep=lambda s: None, wait_s=10)
    assert mode != "ERR" and gmem is not None
    restarts = [c for c in docker.calls if c[1] == "restart"]
    assert restarts == [["docker", "restart", "comfyui"]]


def test_reclaim_polls_until_memory_frees():
    reads = iter([_SHORT, _SHORT, _SHORT, _ROOMY])
    slept = []
    gmem, mode, _ = launch.reclaim_and_recompute(
        ["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
        read=lambda: next(reads), run=_Docker(["comfyui"]), sleep=slept.append, wait_s=10)
    assert mode != "ERR" and len(slept) == 3


def test_reclaim_gives_up_after_wait_and_reports_err():
    gmem, mode, _ = launch.reclaim_and_recompute(
        ["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
        read=lambda: _SHORT, run=_Docker(["comfyui"]), sleep=lambda s: None, wait_s=3)
    assert mode == "ERR" and gmem is None


def test_reclaim_with_nothing_running_does_not_wait():
    slept = []
    _, mode, _ = launch.reclaim_and_recompute(
        ["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
        read=lambda: _SHORT, run=_Docker([]), sleep=slept.append, wait_s=30)
    assert mode == "ERR" and slept == []


def test_main_does_not_reclaim_when_memory_suffices(tmp_path, monkeypatch):
    mdir = _model_dir(tmp_path, {"num_hidden_layers": 32, "num_attention_heads": 32,
                                 "num_key_value_heads": 8, "head_dim": 128}, 1024)
    for k, v in dict(MODEL_PATH=mdir, MODEL_HOST_PATH=mdir, CONTAINER_NAME="c", IMAGE="i",
                     PORT="1", HOST="h", LLM_ROOT_PATH="/m", RECLAIM_CONTAINERS="comfyui",
                     SPARKYARD_MEMINFO_PATH=_meminfo(tmp_path, 124 * 1048576,
                                                     100 * 1048576, 100 * 1048576)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(launch, "reclaim_and_recompute",
                        lambda *a, **k: pytest.fail("reclaimed with enough memory"))
    launch.main(["--print"])


# --- reclaim margin: vLLM's real footprint exceeds gmem*total by ~10 GiB ---
def test_short_when_err():
    assert launch.short_of_memory(None, "ERR", {"free": 40.0, "total": 111.0}, margin=10)


def test_not_short_when_free_covers_share_plus_margin():
    assert not launch.short_of_memory(0.30, "sized", {"free": 50.0, "total": 111.0}, margin=10)  # 33.3+10 <= 50


def test_short_when_free_misses_share_plus_margin():
    assert launch.short_of_memory(0.34, "sized", {"free": 44.4, "total": 111.5}, margin=10)  # 37.9+10 > 44.4


def test_margin_zero_keeps_the_old_err_only_rule():
    assert not launch.short_of_memory(0.34, "sized", {"free": 44.4, "total": 111.5}, margin=0)


# --- busy guard: never restart a co-tenant that is working ---
def test_parse_cpu_percent():
    assert launch.parse_cpu("100.67%") == 100.67
    assert launch.parse_cpu("0.12%") == 0.12
    assert launch.parse_cpu("") is None and launch.parse_cpu("garbage") is None


class _BusyDocker(_Docker):
    """Running containers report CPU from a per-name script of readings."""
    def __init__(self, running, cpu):
        super().__init__(running)
        self.cpu = {k: list(v) for k, v in cpu.items()}

    def __call__(self, argv):
        if argv[:2] == ["docker", "stats"]:
            self.calls.append(argv)
            seq = self.cpu.get(argv[-1], ["0.1%"])
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return super().__call__(argv)


def _restarts(docker):
    return [c[2] for c in docker.calls if c[:2] == ["docker", "restart"]]


def test_idle_container_is_restarted_at_once():
    docker = _BusyDocker(["comfyui"], {"comfyui": ["0.1%"]})
    slept = []
    launch.reclaim_and_recompute(["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
                                 read=lambda: _ROOMY, run=docker, sleep=slept.append,
                                 wait_s=10, busy_cpu=20, busy_wait_s=300)
    assert _restarts(docker) == ["comfyui"] and slept == []


def test_busy_container_is_waited_for_then_restarted():
    docker = _BusyDocker(["comfyui"], {"comfyui": ["100.7%", "100.9%", "0.2%"]})
    slept = []
    launch.reclaim_and_recompute(["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
                                 read=lambda: _ROOMY, run=docker, sleep=slept.append,
                                 wait_s=10, busy_cpu=20, busy_wait_s=300)
    assert _restarts(docker) == ["comfyui"] and sum(slept) == 10   # two 5 s polls


def test_container_busy_for_the_whole_wait_is_left_alone():
    docker = _BusyDocker(["comfyui"], {"comfyui": ["100.7%"]})
    gmem, mode, _ = launch.reclaim_and_recompute(
        ["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
        read=lambda: _SHORT, run=docker, sleep=lambda s: None,
        wait_s=10, busy_cpu=20, busy_wait_s=30)
    assert _restarts(docker) == [] and mode == "ERR"


def test_unreadable_cpu_counts_as_idle():
    docker = _BusyDocker(["comfyui"], {"comfyui": [""]})
    launch.reclaim_and_recompute(["comfyui"], (32, 32, 8, 128), launch.Params(**_P),
                                 read=lambda: _ROOMY, run=docker, sleep=lambda s: None,
                                 wait_s=10, busy_cpu=20, busy_wait_s=300)
    assert _restarts(docker) == ["comfyui"]


def test_main_reclaims_on_margin_shortfall_even_without_err(tmp_path, monkeypatch):
    mdir = _model_dir(tmp_path, {"num_hidden_layers": 32, "num_attention_heads": 32,
                                 "num_key_value_heads": 8, "head_dim": 128}, 1024)
    # 58 GiB available -> free 45.51 after headroom/overhead, so u_cap 0.36 passes
    # the 0.34 floor (no ERR); but gmem 0.34 of 111.51 = 37.9, +10 margin = 47.9 > 45.51.
    for k, v in dict(MODEL_PATH=mdir, MODEL_HOST_PATH=mdir, CONTAINER_NAME="c", IMAGE="i",
                     PORT="1", HOST="h", LLM_ROOT_PATH="/m", RECLAIM_CONTAINERS="comfyui",
                     GMEM_MIN="0.34", GMEM_MAX="0.34", SAFETY_GIB="1",
                     SPARKYARD_MEMINFO_PATH=_meminfo(tmp_path, 124 * 1048576,
                                                     58 * 1048576, 58 * 1048576)).items():
        monkeypatch.setenv(k, v)
    called = []
    real = launch.compute_gmem
    monkeypatch.setattr(launch, "reclaim_and_recompute",
                        lambda names, cfg, params, **kw: (called.append(names) or real(cfg, params, launch.read_meminfo())))
    launch.main(["--print"])
    assert called == [["comfyui"]]


# --- two-node: meminfo parsing, tp-aware sizing, ssh/nccl/cluster argv ---
import shlex  # noqa: E402


def test_parse_meminfo_matches_read_meminfo(tmp_path):
    path = _meminfo(tmp_path, 124 * 1048576, 100 * 1048576, 90 * 1048576)
    text = open(path).read()
    assert launch.parse_meminfo(text) == launch.read_meminfo(path) == (
        124 * 1048576, 100 * 1048576, 90 * 1048576)


def _tp_params(tp=None):
    kw = dict(max_model_len=4096, max_num_seqs=10, kv_dtype_bytes=1,
              gmin=0.55, gmax=0.92, safety=4.0, cuda_overhead=6.3,
              ceiling=117.81, kv_batch_realistic=4, free_buffer=5.0, weights=75.0)
    if tp is not None:
        kw["tp"] = tp
    return launch.Params(**kw)


def test_compute_gmem_tp2_halves_weights_and_kv():
    mi = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    cfg = (32, 32, 8, 128)
    _, _, d1 = launch.compute_gmem(cfg, _tp_params(), mi)
    _, _, d1b = launch.compute_gmem(cfg, _tp_params(1), mi)
    assert d1 == d1b and d1["tp"] == 1
    _, _, d2 = launch.compute_gmem(cfg, _tp_params(2), mi)
    raw_kv = 2 * 32 * 8 * 128 * 4096 * 4 * 1 / launch.GIB
    assert d2["tp"] == 2
    assert d2["weights"] == launch.g2(d1["weights"] / 2)
    assert d2["kv"] == launch.g2(raw_kv / 2)
    assert d2["need"] == launch.g2(d2["weights"] + d2["kv"] + 4.0)
    assert d2["need"] < d1["need"]


def _cluster(**kw):
    base = dict(worker_host="spark2", worker_user="scott", worker_ip="10.0.0.2",
                head_ip="10.0.0.1", master_port="29500", eth_if="enp1s0f0np0",
                ib_hca="rocep1s0f0,roceP2p1s0f0", ssh_dir="/app/cluster")
    base.update(kw)
    return launch.ClusterEnv(**base)


def test_ssh_argv_shape():
    c = _cluster()
    remote = ["docker", "ps", "--filter", "name=a b"]
    argv = launch.ssh_argv(c, remote)
    assert argv == ["ssh", "-i", "/app/cluster/id_ed25519",
                    "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    "-o", "UserKnownHostsFile=/app/cluster/known_hosts",
                    "-o", "StrictHostKeyChecking=yes",
                    "-l", "scott", "spark2", "--", shlex.join(remote)]
    assert argv[-1] == "docker ps --filter 'name=a b'"


def test_nccl_env_order():
    assert launch.nccl_env(_cluster(), "10.0.0.2") == [
        "-e", "VLLM_HOST_IP=10.0.0.2",
        "-e", "NCCL_SOCKET_IFNAME=enp1s0f0np0",
        "-e", "GLOO_SOCKET_IFNAME=enp1s0f0np0",
        "-e", "UCX_NET_DEVICES=enp1s0f0np0",
        "-e", "TP_SOCKET_IFNAME=enp1s0f0np0",
        "-e", "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0",
        "-e", "NCCL_IB_DISABLE=0",
        "-e", "NCCL_IGNORE_CPU_AFFINITY=1"]


def test_cluster_env_from_environ(monkeypatch):
    vals = dict(WORKER_SSH_HOST="spark2", WORKER_SSH_USER="scott", WORKER_CX7_IP="10.0.0.2",
                HEAD_CX7_IP="10.0.0.1", MASTER_PORT="29500", ETH_IF="e0", IB_HCA="r0,r1")
    for k, v in vals.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CLUSTER_SSH_DIR", raising=False)
    c = launch.cluster_env_from_environ()
    assert (c.worker_host, c.worker_user, c.worker_ip, c.head_ip, c.master_port,
            c.eth_if, c.ib_hca, c.ssh_dir) == (
        "spark2", "scott", "10.0.0.2", "10.0.0.1", "29500", "e0", "r0,r1", "/app/cluster")
    monkeypatch.setenv("CLUSTER_SSH_DIR", "/x")
    assert launch.cluster_env_from_environ().ssh_dir == "/x"
    monkeypatch.delenv("IB_HCA")
    with pytest.raises(SystemExit):
        launch.cluster_env_from_environ()


def _cl_env(**kw):
    base = dict(model_path="/models/vllm/X", model_host_path="/models/vllm/X",
                container_name="vllm-x-9000", image="vllm-node:latest",
                port="9000", host="0.0.0.0", max_model_len=131072, max_num_seqs=10,
                extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="vllm serve")
    base.update(kw)
    return launch.LaunchEnv(**base)


def _sub(argv, seq):
    i = argv.index(seq[0])
    assert argv[i:i + len(seq)] == seq, (seq, argv)
    return i


def test_build_argv_cluster_rank0(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    argv = launch.build_argv("0.80", _cl_env(), ["--kv-cache-dtype", "fp8"],
                             cluster=_cluster(), rank=0, api_host="172.17.0.1")
    assert argv[:6] == ["docker", "run", "--rm", "--name", "vllm-x-9000", "--runtime"]
    assert "-d" not in argv and "--headless" not in argv
    assert "container:llama-swap" not in argv
    _sub(argv, ["--network", "host"])
    _sub(argv, ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1"])
    assert "--ipc=host" in argv and "--privileged" not in argv
    assert "VLLM_HOST_IP=10.0.0.1" in argv
    assert "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0" in argv
    assert argv.index("VLLM_HOST_IP=10.0.0.1") < argv.index("NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0")
    _sub(argv, ["--host", "172.17.0.1", "--port", "9000"])
    i = _sub(argv, ["--max-num-seqs", "10"])
    assert argv[i + 2:] == ["--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "0",
                            "--master-addr", "10.0.0.1", "--master-port", "29500",
                            "--kv-cache-dtype", "fp8"]


def test_build_argv_cluster_rank1(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    argv = launch.build_argv("0.80", _cl_env(), ["--kv-cache-dtype", "fp8"],
                             cluster=_cluster(), rank=1, api_host="172.17.0.1")
    assert argv[:7] == ["docker", "run", "-d", "--rm", "--name", "vllm-x-9000-w", "--runtime"]
    assert "container:llama-swap" not in argv
    _sub(argv, ["--network", "host"])
    _sub(argv, ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1"])
    assert "VLLM_HOST_IP=10.0.0.2" in argv and "VLLM_HOST_IP=10.0.0.1" not in argv
    assert "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0" in argv
    i = _sub(argv, ["--max-num-seqs", "10"])
    assert argv[i + 2:] == ["--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "1",
                            "--master-addr", "10.0.0.1", "--master-port", "29500",
                            "--headless", "--kv-cache-dtype", "fp8"]


def test_build_argv_cluster_pre_launch_wraps_cluster_args(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    argv = launch.build_argv("0.80", _cl_env(pre_launch_cmd="true"), [],
                             cluster=_cluster(), rank=1, api_host="172.17.0.1")
    script = argv[-1]
    assert script.startswith("true && exec vllm serve /models/vllm/X")
    assert "--node-rank 1 --master-addr 10.0.0.1 --master-port 29500 --headless" in script


def test_build_argv_single_node_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    env = _cl_env()
    a = launch.build_argv("0.80", env, ["--kv-cache-dtype", "fp8"])
    assert a == launch.build_argv("0.80", env, ["--kv-cache-dtype", "fp8"], cluster=None)
    assert "container:llama-swap" in a
    assert "--headless" not in a and "--nnodes" not in a


def test_build_argv_cluster_head_requires_api_host(monkeypatch, capsys):
    monkeypatch.setenv("LLM_ROOT_PATH", "/data/LLMs")
    with pytest.raises(SystemExit):
        launch.build_argv("0.80", _cl_env(), [], cluster=_cluster(), rank=0, api_host=None)
    assert "0.0.0.0" not in capsys.readouterr().out


# --- two-node orchestration: preflight, sizing, worker lifecycle, --stop ---
import signal as _signal  # noqa: E402
import types  # noqa: E402

_MEMINFO_ROOMY = ("MemTotal:       130023424 kB\nMemFree:        104857600 kB\n"
                  "MemAvailable:   104857600 kB\n")          # 124 / 100 / 100 GiB
_MEMINFO_FULL = ("MemTotal:       130023424 kB\nMemFree:        10485760 kB\n"
                 "MemAvailable:   10485760 kB\n")            # 124 / 10 / 10 GiB


class _Run:
    """Records argv lists; returns the first scripted (rc, out, err) whose key is
    a substring of the space-joined argv, else (0, "", "")."""
    def __init__(self, script=()):
        self.script = list(script)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for key, res in self.script:
            if key in joined:
                return res
        return (0, "", "")

    def joined(self):
        return [" ".join(c) for c in self.calls]


def _cl(**kw):
    base = dict(worker_host="spark-2", worker_user="acme", worker_ip="10.0.0.2",
                head_ip="10.0.0.1", master_port="29500", eth_if="enp1s0f0np0",
                ib_hca="rocep1s0f0", ssh_dir="/app/cluster")
    base.update(kw)
    return launch.ClusterEnv(**base)


def _big_env(**kw):
    base = dict(model_path="/models/vllm/Org/Big", model_host_path="/mnt/elsewhere/Org/Big",
                container_name="big-5800", image="vllm-node:latest", port="5800",
                host="0.0.0.0", max_model_len=4096, max_num_seqs=10,
                extra_docker_args="", pre_launch_cmd="", vllm_serve_prefix="vllm serve")
    base.update(kw)
    return launch.LaunchEnv(**base)


def test_preflight_ssh_failure_message(capsys):
    run = _Run([("-- true", (255, "", "ssh: connect to host spark-2: timed out"))])
    with pytest.raises(SystemExit) as e:
        launch.cluster_preflight(_cl(worker_host="10.0.0.2"), _big_env(), "/data/LLMs", run=run)
    assert e.value.code != 0
    err = capsys.readouterr().err
    assert "cannot ssh to acme@10.0.0.2" in err and "sparkyard cluster check" in err
    assert len(run.calls) == 1


def test_preflight_image_mismatch(capsys):
    run = _Run([("-- true", (0, "", "")),
                ("spark-2 -- docker image inspect", (0, "sha256:bbb\n", "")),
                ("docker image inspect", (0, "sha256:aaa\n", ""))])
    with pytest.raises(SystemExit):
        launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run)
    err = capsys.readouterr().err
    assert "sparkyard cluster sync" in err and "vllm-node:latest" in err
    assert not any("test -f" in j for j in run.joined())


def _layer_run(head_layers, worker_layers):
    def run(argv):
        remote = argv[0] == "ssh"
        if "{{.Id}}" in " ".join(argv):
            return (0, "sha256:606f1613\n" if remote else "sha256:ddca1497\n", "")
        if "docker image inspect" in " ".join(argv):
            return (0, (worker_layers if remote else head_layers) + "\n", "")
        return (0, "", "")
    return run


def test_preflight_same_layers_different_ids_passes():
    layers = '["sha256:l1","sha256:l2"]'
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=_layer_run(layers, layers))


def test_preflight_different_layers_fails(capsys):
    with pytest.raises(SystemExit):
        launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs",
                                 run=_layer_run('["sha256:l1"]', '["sha256:l2"]'))
    assert "sparkyard cluster sync" in capsys.readouterr().err


def test_preflight_inspects_layers_not_id():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run)
    insp = [c for c in run.calls if "docker image inspect" in " ".join(c)]
    assert insp and all("{{json .RootFS.Layers}}" in " ".join(c) for c in insp)


def test_preflight_missing_weights_uses_model_path(capsys):
    run = _Run([("test -f", (1, "", "")), ("docker image inspect", (0, "sha256:aaa\n", ""))])
    with pytest.raises(SystemExit):
        launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run, model="big")
    tests = [c for c in run.calls if "test -f" in c[-1]]
    assert tests and tests[0][-1] == "test -f /data/LLMs/vllm/Org/Big/config.json"
    assert tests[0][:2] == ["ssh", "-i"]
    err = capsys.readouterr().err
    assert "sparkyard cluster sync big" in err and "spark-2" in err


def test_preflight_removes_stale_worker():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run)
    last = run.calls[-1]
    assert last[0] == "ssh" and last[-1] == "docker rm -f big-5800-w"
    # local + remote image inspect both ran
    inspects = [c for c in run.calls if "docker image inspect" in " ".join(c)]
    assert len(inspects) == 2 and inspects[0][0] == "docker" and inspects[1][0] == "ssh"


def test_preflight_print_mode_skips_stale_removal():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run, remove_stale=False)
    assert not any("docker rm" in j for j in run.joined())


def test_read_worker_meminfo_parses():
    run = _Run([("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    assert launch.read_worker_meminfo(_cl(), run=run) == (130023424, 104857600, 104857600)
    assert run.calls[0][0] == "ssh" and run.calls[0][-1] == "cat /proc/meminfo"


def test_read_worker_meminfo_timeout_names_worker(capsys):
    run = _Run([("cat /proc/meminfo", (255, "", "Connection timed out during banner exchange"))])
    with pytest.raises(SystemExit):
        launch.read_worker_meminfo(_cl(), run=run)
    err = capsys.readouterr().err
    assert "spark-2" in err or "10.0.0.2" in err
    assert "acme@spark-2" in err


_CFG = (32, 32, 8, 128)


def test_size_cluster_takes_minimum():
    # weights 75 GiB, tp default 1 -> size_cluster forces tp=2: need 37.5+0.5+4 = 42.
    # head free 87.51 -> 42/111.51 = 0.38; worker (55 GiB avail) free 42.51 ->
    # u_cap (42.51-5)/111.51 = 0.336 caps it at 0.34.
    p = _tp_params(); p.gmin = 0.2
    p2 = _tp_params(2); p2.gmin = 0.2
    head = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    worker = (124 * 1048576, 55 * 1048576, 55 * 1048576)
    gmem, mode, dh, dw = launch.size_cluster(_CFG, p, head, worker)
    gh, _, _ = launch.compute_gmem(_CFG, p2, head)
    gw, _, _ = launch.compute_gmem(_CFG, p2, worker)
    assert (gh, gw) == (0.38, 0.34)
    assert dh["gmem"] == gh and dw["gmem"] == gw
    assert dh["tp"] == 2 and dw["tp"] == 2
    assert gmem == gw == min(gh, gw)
    assert gh > gw
    assert mode in ("sized", "fallback")
    assert p.tp == 1  # caller's params untouched


def test_size_cluster_err_if_either_err():
    p = _tp_params()
    ok = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    full = (124 * 1048576, 10 * 1048576, 10 * 1048576)
    assert launch.size_cluster(_CFG, p, ok, full)[:2] == (None, "ERR")
    assert launch.size_cluster(_CFG, p, full, ok)[:2] == (None, "ERR")


def test_size_cluster_fallback_if_either_fallback():
    p = _tp_params()
    p.weights = 200.0  # need 100+ per node > free -> fallback on both
    ok = (124 * 1048576, 100 * 1048576, 100 * 1048576)
    assert launch.size_cluster(_CFG, p, ok, ok)[1] == "fallback"


def _cluster_main_env(tmp_path, monkeypatch, **extra):
    mdir = _model_dir(tmp_path, {"num_hidden_layers": 32, "num_attention_heads": 32,
                                 "num_key_value_heads": 8, "head_dim": 128}, 1024)
    vals = dict(MODEL_PATH="/models/vllm/Org/Big", MODEL_HOST_PATH=mdir,
                CONTAINER_NAME="big-5800", IMAGE="vllm-node:latest", PORT="5800",
                HOST="0.0.0.0", LLM_ROOT_PATH="/data/LLMs", MAX_MODEL_LEN="4096",
                NODES="2", WORKER_SSH_HOST="spark-2", WORKER_SSH_USER="acme",
                WORKER_CX7_IP="10.0.0.2", HEAD_CX7_IP="10.0.0.1", MASTER_PORT="29500",
                ETH_IF="enp1s0f0np0", IB_HCA="rocep1s0f0", CLUSTER_API_HOST="172.17.0.1",
                SPARKYARD_MEMINFO_PATH=_meminfo(tmp_path, 124 * 1048576,
                                                100 * 1048576, 100 * 1048576))
    vals.update(extra)
    for k, v in vals.items():
        monkeypatch.setenv(k, v)
    for k in ("GMEM_OVERRIDE", "RECLAIM_CONTAINERS", "SPARKYARD_LAUNCH_PRINT", "EXTRA_PATHS"):
        if k not in extra:
            monkeypatch.delenv(k, raising=False)


def test_main_worker_short_names_worker(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch)
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_FULL, ""))])
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch, "run_cluster",
                        lambda *a, **k: pytest.fail("started a container"))
    with pytest.raises(SystemExit) as e:
        launch.main(["--served-model-name", "big"])
    assert e.value.code != 0
    assert "worker spark-2 short of memory" in capsys.readouterr().err
    assert not any("docker run" in j for j in run.joined())


def test_main_print_cluster(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch)
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    launch.main(["--print", "--served-model-name", "big"])
    cap = capsys.readouterr()
    out = cap.out
    assert "gpu_memory_utilization=" in out and "--- worker ---" in out
    head, worker = out.split("--- worker ---")
    assert "--headless" in worker and "--headless" not in head
    assert "172.17.0.1" in head and "0.0.0.0" not in head
    assert "big-5800-w" in worker
    # print mode: no side effects on the worker
    assert not any("docker rm" in j or "docker run" in j for j in run.joined())
    assert any("cat /proc/meminfo" in j for j in run.joined())
    err = cap.err
    assert "[auto-gmem] head: " in err and "[auto-gmem] worker: " in err


def test_main_cluster_resolves_host_docker_internal(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch)
    monkeypatch.delenv("CLUSTER_API_HOST")
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)

    def boom(name):
        assert name == "host.docker.internal"
        raise launch.socket.gaierror("nope")
    monkeypatch.setattr(launch.socket, "gethostbyname", boom)
    with pytest.raises(SystemExit):
        launch.main(["--print"])
    assert "cannot resolve host.docker.internal" in capsys.readouterr().err
    monkeypatch.setattr(launch.socket, "gethostbyname", lambda n: "192.168.65.254")
    launch.main(["--print"])
    assert "192.168.65.254" in capsys.readouterr().out.split("--- worker ---")[0]


def test_main_cluster_runs_worker_over_ssh_then_head(tmp_path, monkeypatch):
    _cluster_main_env(tmp_path, monkeypatch)
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    got = {}

    def fake_run_cluster(head, worker, stop, **kw):
        got.update(head=head, worker=worker, stop=stop)
        return 3
    monkeypatch.setattr(launch, "run_cluster", fake_run_cluster)
    with pytest.raises(SystemExit) as e:
        launch.main(["--served-model-name", "big"])
    assert e.value.code == 3
    assert got["head"][:2] == ["docker", "run"] and "--network" in got["head"]
    assert got["worker"][0] == "ssh" and got["worker"][-1].startswith("docker run -d")
    assert "--headless" in got["worker"][-1]
    assert got["stop"][0] == "ssh" and got["stop"][-1] == "docker stop big-5800-w"
    # preflight removed the stale worker before launch
    assert any(c[-1] == "docker rm -f big-5800-w" for c in run.calls)


class _Proc:
    def __init__(self, rc, on_wait=None):
        self.rc = rc
        self.returncode = None
        self.signals = []
        self.on_wait = on_wait

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self):
        if self.on_wait:
            self.on_wait()
        self.returncode = self.rc
        return self.rc


def _fake_signal():
    handlers = {}
    mod = types.SimpleNamespace(SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT,
                                signal=lambda s, h: handlers.__setitem__(s, h))
    return mod, handlers


def test_run_cluster_worker_fail_never_starts_head(capsys):
    run = _Run([("docker run -d", (125, "", "Unable to find image 'vllm-node:latest' locally"))])
    started = []
    sig, _ = _fake_signal()
    with pytest.raises(SystemExit) as e:
        launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                           ["ssh", "w", "--", "docker stop x-w"], run=run,
                           popen=lambda a: started.append(a), signal_mod=sig)
    assert e.value.code != 0
    assert started == []
    assert "Unable to find image 'vllm-node:latest' locally" in capsys.readouterr().err


def test_run_cluster_stops_worker_on_head_exit():
    run = _Run()
    proc = _Proc(1)
    stop = ["ssh", "w", "--", "docker stop x-w"]
    sig, _ = _fake_signal()
    rc = launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                            stop, run=run, popen=lambda a: proc, signal_mod=sig)
    assert rc == 1
    assert run.calls[-1] == stop


def test_run_cluster_forwards_sigterm_and_cleans_up():
    run = _Run()
    sig, handlers = _fake_signal()
    proc = _Proc(-_signal.SIGTERM,
                 on_wait=lambda: handlers[_signal.SIGTERM](_signal.SIGTERM, None))
    stop = ["ssh", "w", "--", "docker stop x-w"]
    rc = launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                            stop, run=run, popen=lambda a: proc, signal_mod=sig)
    assert _signal.SIGINT in handlers and _signal.SIGTERM in handlers
    assert proc.signals == [_signal.SIGTERM]
    assert run.calls[-1] == stop
    assert rc == 128 + _signal.SIGTERM


def test_run_cluster_signal_before_head_start_skips_head():
    sig, handlers = _fake_signal()
    started = []

    class _R(_Run):
        def __call__(self, argv):
            if "docker run -d" in " ".join(argv):
                handlers[_signal.SIGTERM](_signal.SIGTERM, None)
            return super().__call__(argv)
    run = _R()
    stop = ["ssh", "w", "--", "docker stop x-w"]
    rc = launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                            stop, run=run, popen=lambda a: started.append(a), signal_mod=sig)
    assert started == [] and run.calls[-1] == stop and rc == 128 + _signal.SIGTERM


def test_stop_cluster_tolerates_missing():
    run = _Run([("docker stop", (1, "", "No such container"))])
    assert launch.stop_cluster(_cl(), "big-5800", run=run) == 0
    assert len(run.calls) == 2
    assert run.calls[0][0] == "ssh" and run.calls[0][-1] == "docker stop big-5800-w"
    assert run.calls[1] == ["docker", "stop", "big-5800"]


def test_stop_cluster_tolerates_runner_errors():
    def run(argv):
        raise FileNotFoundError("ssh")
    assert launch.stop_cluster(_cl(), "big-5800", run=run) == 0


def test_main_stop_needs_only_container_and_cluster_env(monkeypatch):
    for k in ("MODEL_PATH", "MODEL_HOST_PATH", "IMAGE", "PORT", "HOST", "LLM_ROOT_PATH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in dict(CONTAINER_NAME="big-5800", NODES="2", WORKER_SSH_HOST="spark-2",
                     WORKER_SSH_USER="acme", WORKER_CX7_IP="10.0.0.2", HEAD_CX7_IP="10.0.0.1",
                     MASTER_PORT="29500", ETH_IF="e0", IB_HCA="r0").items():
        monkeypatch.setenv(k, v)
    run = _Run([("docker stop", (1, "", "No such container"))])
    monkeypatch.setattr(launch, "_run", run)
    assert launch.main(["--stop"]) == 0
    assert [c[-1] for c in run.calls] == ["docker stop big-5800-w", "big-5800"]


def test_main_worker_short_does_not_reclaim_on_head(tmp_path, monkeypatch):
    _cluster_main_env(tmp_path, monkeypatch, RECLAIM_CONTAINERS="comfyui")
    # head short too (70 GiB avail -> ERR), so reclaim would trigger if reached
    sub = tmp_path / "h"; sub.mkdir()
    monkeypatch.setenv("SPARKYARD_MEMINFO_PATH",
                       _meminfo(sub, 124 * 1048576, 70 * 1048576, 70 * 1048576))
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_FULL, ""))])
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch, "reclaim_and_recompute",
                        lambda *a, **k: pytest.fail("reclaimed on head for a short worker"))
    with pytest.raises(SystemExit):
        launch.main(["--print"])


# --- final-review fixes: abort before start, stale head, worker-fail cleanup,
# --- docker permission hint, --stop ${PID}, worker log streaming ---

def test_preflight_removes_stale_local_head_before_worker():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run)
    assert ["docker", "rm", "-f", "big-5800"] in run.calls
    assert run.calls.index(["docker", "rm", "-f", "big-5800"]) < len(run.calls) - 1


def test_preflight_print_mode_keeps_local_head():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run, remove_stale=False)
    assert ["docker", "rm", "-f", "big-5800"] not in run.calls


def test_preflight_worker_docker_permission_denied_hint(capsys):
    denied = ("permission denied while trying to connect to the Docker daemon socket at "
              "unix:///var/run/docker.sock")
    run = _Run([("spark-2 -- docker image inspect", (1, "", denied)),
                ("docker image inspect", (0, "sha256:aaa\n", ""))])
    with pytest.raises(SystemExit) as e:
        launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run)
    assert e.value.code != 0
    err = capsys.readouterr().err
    assert "sudo usermod -aG docker acme" in err and "spark-2" in err
    assert "cluster sync" not in err


def test_run_cluster_worker_fail_stops_worker_best_effort(capsys):
    run = _Run([("docker run -d", (255, "", "Connection reset by peer"))])
    stop = ["ssh", "w", "--", "docker stop x-w"]
    sig, _ = _fake_signal()
    with pytest.raises(SystemExit) as e:
        launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                           stop, run=run, popen=lambda a: pytest.fail("head started"),
                           signal_mod=sig)
    assert e.value.code != 0
    assert run.calls[-1] == stop
    assert "Connection reset by peer" in capsys.readouterr().err


def test_run_cluster_worker_fail_stop_error_still_fatal(capsys):
    def run(argv):
        if "docker stop" in " ".join(argv):
            raise FileNotFoundError("ssh")
        return (1, "", "boom")
    sig, _ = _fake_signal()
    with pytest.raises(SystemExit) as e:
        launch.run_cluster(["h"], ["ssh", "--", "docker run -d x"], ["ssh", "--", "docker stop x-w"],
                           run=run, popen=lambda a: None, signal_mod=sig)
    assert e.value.code != 0 and "boom" in capsys.readouterr().err


def test_install_abort_sets_flag_and_forwards():
    sig, handlers = _fake_signal()
    state = launch.install_abort(sig)
    assert set(handlers) == {_signal.SIGTERM, _signal.SIGINT}
    handlers[_signal.SIGTERM](_signal.SIGTERM, None)
    assert state["sig"] == _signal.SIGTERM
    proc = _Proc(0)
    state["proc"] = proc
    handlers[_signal.SIGINT](_signal.SIGINT, None)
    assert proc.signals == [_signal.SIGINT]


def test_run_cluster_aborted_before_worker_starts_nothing(capsys):
    sig, handlers = _fake_signal()
    state = launch.install_abort(sig)
    handlers[_signal.SIGTERM](_signal.SIGTERM, None)   # arrived during preflight
    run = _Run()
    rc = launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                            ["ssh", "w", "--", "docker stop x-w"], run=run,
                            popen=lambda a: pytest.fail("head started"), signal_mod=sig,
                            state=state)
    assert rc == 128 + _signal.SIGTERM
    assert run.calls == []
    assert "aborted" in capsys.readouterr().err


def test_run_cluster_with_state_does_not_reinstall_handlers():
    sig, handlers = _fake_signal()
    state = launch.install_abort(sig)
    first = dict(handlers)
    proc = _Proc(0)
    launch.run_cluster(["h"], ["ssh", "--", "docker run -d x"], ["ssh", "--", "docker stop x-w"],
                       run=_Run(), popen=lambda a: proc, signal_mod=sig, state=state)
    assert handlers == first and state["proc"] is proc


def test_main_cluster_signal_during_preflight_aborts_before_any_start(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch)
    sig, handlers = _fake_signal()
    monkeypatch.setattr(launch, "signal", types.SimpleNamespace(
        SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT, getsignal=lambda s: None,
        signal=lambda s, h: handlers.__setitem__(s, h)))

    class _R(_Run):
        def __call__(self, argv):
            if " ".join(argv).endswith("cat /proc/meminfo"):
                # llama-swap's cmdStop SIGTERMs the launcher mid-sizing
                handlers[_signal.SIGTERM](_signal.SIGTERM, None)
            return super().__call__(argv)
    run = _R([("docker image inspect", (0, "sha256:aaa\n", "")),
              ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("popen"))
    with pytest.raises(SystemExit) as e:
        launch.main(["--served-model-name", "big"])
    assert e.value.code == 128 + _signal.SIGTERM
    assert not any("docker run" in j for j in run.joined())


def test_main_cluster_restores_signal_handlers(tmp_path, monkeypatch):
    _cluster_main_env(tmp_path, monkeypatch)
    installed = []
    monkeypatch.setattr(launch, "signal", types.SimpleNamespace(
        SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT, getsignal=lambda s: f"prev{s}",
        signal=lambda s, h: installed.append((s, h))))
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch, "run_cluster", lambda *a, **k: 0)
    with pytest.raises(SystemExit):
        launch.main(["--served-model-name", "big"])
    assert installed[-2:] == [(_signal.SIGTERM, f"prev{_signal.SIGTERM}"),
                              (_signal.SIGINT, f"prev{_signal.SIGINT}")]


def test_main_print_cluster_installs_no_handlers(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "signal", types.SimpleNamespace(
        SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT, getsignal=lambda s: None,
        signal=lambda s, h: pytest.fail("handler installed in --print")))
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    launch.main(["--print"])


def test_stop_cluster_sigterms_launcher_pid_first():
    order = []
    run = _Run()
    orig = run.__call__

    def rec(argv):
        order.append(("run", argv[-1]))
        return orig(argv)
    assert launch.stop_cluster(_cl(), "big-5800", run=rec, pid="4242",
                               kill=lambda p, s: order.append(("kill", p, s))) == 0
    assert order[0] == ("kill", 4242, _signal.SIGTERM)
    assert [o[1] for o in order[1:]] == ["docker stop big-5800-w", "big-5800"]


@pytest.mark.parametrize("pid", ["", "abc", "0", "1", "-5", "${PID}"])
def test_stop_cluster_ignores_bad_pid(pid):
    killed = []
    assert launch.stop_cluster(_cl(), "big-5800", run=_Run(), pid=pid,
                               kill=lambda p, s: killed.append(p)) == 0
    assert killed == []


def test_stop_cluster_tolerates_dead_pid():
    def kill(p, s):
        raise ProcessLookupError(p)
    run = _Run()
    assert launch.stop_cluster(_cl(), "big-5800", run=run, pid="4242", kill=kill) == 0
    assert len(run.calls) == 2


def test_main_stop_passes_pid(monkeypatch):
    for k, v in dict(CONTAINER_NAME="big-5800", NODES="2", WORKER_SSH_HOST="spark-2",
                     WORKER_SSH_USER="acme", WORKER_CX7_IP="10.0.0.2", HEAD_CX7_IP="10.0.0.1",
                     MASTER_PORT="29500", ETH_IF="e0", IB_HCA="r0").items():
        monkeypatch.setenv(k, v)
    killed = []
    monkeypatch.setattr(launch.os, "kill", lambda p, s: killed.append((p, s)))
    monkeypatch.setattr(launch, "_run", _Run())
    assert launch.main(["--stop", "4242"]) == 0
    assert killed == [(4242, _signal.SIGTERM)]


class _LogProc:
    def __init__(self, lines):
        import io
        self.stdout = io.StringIO("".join(lines))
        self.terminated = False
        self.argv = None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_run_cluster_streams_worker_logs_after_worker_and_stops_them(capsys):
    events = []
    logp = _LogProc(["loading weights\n", "ready"])

    def log_popen(argv, **kw):
        events.append("logs")
        logp.argv = argv
        return logp

    class _R(_Run):
        def __call__(self, argv):
            j = " ".join(argv)
            events.append("worker" if "docker run -d" in j else "stop" if "docker stop" in j else j)
            return super().__call__(argv)

    def popen(a):
        events.append("head")
        return _Proc(0)
    sig, _ = _fake_signal()
    logs = ["ssh", "w", "--", "docker logs -f x-w"]
    rc = launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                            ["ssh", "w", "--", "docker stop x-w"], run=_R(), popen=popen,
                            signal_mod=sig, logs_argv=logs, log_popen=log_popen)
    assert rc == 0
    assert events == ["worker", "logs", "head", "stop"]
    assert logp.argv == logs and logp.terminated
    launch._join_log_threads(timeout=2)
    err = capsys.readouterr().err
    assert "[worker] loading weights\n" in err and "[worker] ready\n" in err


def test_run_cluster_log_stream_failure_is_harmless():
    def log_popen(argv, **kw):
        raise OSError("no ssh")
    sig, _ = _fake_signal()
    rc = launch.run_cluster(["h"], ["ssh", "--", "docker run -d x"], ["ssh", "--", "docker stop x-w"],
                            run=_Run(), popen=lambda a: _Proc(7), signal_mod=sig,
                            logs_argv=["ssh", "--", "docker logs -f x-w"], log_popen=log_popen)
    assert rc == 7


def test_run_cluster_no_log_stream_when_worker_fails():
    started = []
    sig, _ = _fake_signal()
    with pytest.raises(SystemExit):
        launch.run_cluster(["h"], ["ssh", "--", "docker run -d x"], ["ssh", "--", "docker stop x-w"],
                           run=_Run([("docker run -d", (125, "", "x"))]), popen=lambda a: None,
                           signal_mod=sig, logs_argv=["ssh"],
                           log_popen=lambda *a, **k: started.append(a))
    assert started == []


def test_main_cluster_passes_logs_argv(tmp_path, monkeypatch):
    _cluster_main_env(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "signal", types.SimpleNamespace(
        SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT, getsignal=lambda s: None,
        signal=lambda s, h: None))
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    got = {}
    monkeypatch.setattr(launch, "run_cluster", lambda *a, **k: got.update(k) or 0)
    with pytest.raises(SystemExit):
        launch.main(["--served-model-name", "big"])
    assert got["logs_argv"][0] == "ssh" and got["logs_argv"][-1] == "docker logs -f big-5800-w"
    assert got["state"]["sig"] is None


# --- extra paths (draft models) in cluster preflight ---

def test_parse_extra_paths():
    assert launch.parse_extra_paths(None) == []
    assert launch.parse_extra_paths("") == []
    assert launch.parse_extra_paths("vllm/Org/Big-DSpark,vllm/x/f.json") == [
        "vllm/Org/Big-DSpark", "vllm/x/f.json"]


@pytest.mark.parametrize("bad", ["../etc", "/etc/passwd", "a/../b", "a/..", "a b", "a;b",
                                 "a/$(x)", "-rf", "a//b"])
def test_parse_extra_paths_rejects_unsafe(bad, capsys):
    with pytest.raises(SystemExit) as e:
        launch.parse_extra_paths(bad)
    assert e.value.code != 0
    assert "EXTRA_PATHS" in capsys.readouterr().err


def test_preflight_tests_extra_paths_on_worker():
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", ""))])
    launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs/", run=run, model="big",
                             extra_paths=["vllm/Org/Big-DSpark", "vllm/x/f.json"])
    j = run.joined()
    cfg = next(i for i, c in enumerate(run.calls) if c[-1].startswith("test -f"))
    tests = [c for c in run.calls if c[-1].startswith("test -e")]
    assert [c[-1] for c in tests] == ["test -e /data/LLMs/vllm/Org/Big-DSpark",
                                      "test -e /data/LLMs/vllm/x/f.json"]
    assert all(c[:2] == ["ssh", "-i"] for c in tests)
    assert run.calls.index(tests[0]) > cfg
    assert j[-1].endswith("docker rm -f big-5800-w")


def test_preflight_missing_extra_path_fatal(capsys):
    run = _Run([("test -e /data/LLMs/vllm/Org/Big-DSpark", (1, "", "")),
                ("docker image inspect", (0, "sha256:aaa\n", ""))])
    with pytest.raises(SystemExit) as e:
        launch.cluster_preflight(_cl(), _big_env(), "/data/LLMs", run=run, model="big",
                                 extra_paths=["vllm/Org/Big-DSpark"])
    assert e.value.code != 0
    err = capsys.readouterr().err
    assert ("[cluster] FATAL: worker spark-2 is missing /data/LLMs/vllm/Org/Big-DSpark "
            "(referenced by this model's flags) — run sparkyard cluster sync big") in err
    assert not any("docker rm" in j for j in run.joined())


def test_main_cluster_preflights_extra_paths_from_env(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch, EXTRA_PATHS="vllm/Org/Big-DSpark")
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    launch.main(["--print", "--served-model-name", "big"])
    assert any(c[-1] == "test -e /data/LLMs/vllm/Org/Big-DSpark" for c in run.calls)


def test_main_cluster_rejects_bad_extra_paths(tmp_path, monkeypatch, capsys):
    _cluster_main_env(tmp_path, monkeypatch, EXTRA_PATHS="../../etc")
    run = _Run()
    monkeypatch.setattr(launch, "_run", run)
    with pytest.raises(SystemExit):
        launch.main(["--print", "--served-model-name", "big"])
    assert run.calls == []


# --- worker death while the head runs ---

import subprocess as _subprocess  # noqa: E402

_POLL = ["ssh", "w", "--", "docker container inspect -f '{{.State.Running}}' x-w"]
_STOP = ["ssh", "w", "--", "docker stop x-w"]


class _PollProc:
    """Head proc whose wait(timeout) times out until it is signalled or has
    been waited on `exit_after` times."""
    def __init__(self, rc=0, exit_after=None):
        self.rc, self.exit_after = rc, exit_after
        self.returncode = None
        self.signals, self.waits = [], []

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.signals:
            self.returncode = -self.signals[0]
            return self.returncode
        if self.exit_after is not None and len(self.waits) >= self.exit_after:
            self.returncode = self.rc
            return self.rc
        if timeout is None:
            raise AssertionError("blocking wait on a head that never exits")
        raise _subprocess.TimeoutExpired("head", timeout)


class _SeqRun(_Run):
    """Like _Run, but the worker poll returns `polls` in order (then alive)."""
    def __init__(self, polls):
        super().__init__()
        self.polls = list(polls)

    def __call__(self, argv):
        if "container inspect -f" in " ".join(argv):
            self.calls.append(list(argv))
            return self.polls.pop(0) if self.polls else (0, "true\n", "")
        return super().__call__(argv)


def _cluster_with_poll(proc, run, sig=None):
    sig = sig or _fake_signal()[0]
    return launch.run_cluster(["docker", "run", "head"], ["ssh", "w", "--", "docker run -d x"],
                              _STOP, run=run, popen=lambda a: proc, signal_mod=sig,
                              poll_argv=_POLL, poll_interval=0.01, worker_name="x-w")


def test_run_cluster_worker_gone_stops_head(capsys):
    proc = _PollProc()
    run = _SeqRun([(0, "true\n", ""), (1, "", "Error: No such object: x-w\n")])
    rc = _cluster_with_poll(proc, run)
    assert rc != 0
    assert proc.signals == [_signal.SIGTERM]
    assert run.calls[-1] == _STOP
    assert "[cluster] worker x-w exited before/while serving — stopping head" in \
        capsys.readouterr().err
    assert all(t == 0.01 for t in proc.waits[:-1])


def test_run_cluster_worker_not_running_stops_head():
    proc = _PollProc()
    rc = _cluster_with_poll(proc, _SeqRun([(0, "false\n", "")]))
    assert rc != 0 and proc.signals == [_signal.SIGTERM]


def test_run_cluster_transient_ssh_failures_are_not_death(capsys):
    proc = _PollProc(rc=0, exit_after=8)
    polls = [(255, "", "timeout"), (255, "", "timeout"), (0, "true\n", ""),
             (255, "", "timeout"), (255, "", "timeout")]
    rc = _cluster_with_poll(proc, _SeqRun(polls))
    assert rc == 0 and proc.signals == []
    assert "stopping head" not in capsys.readouterr().err


def test_run_cluster_three_ssh_failures_in_a_row_is_death(capsys):
    proc = _PollProc()
    rc = _cluster_with_poll(proc, _SeqRun([(255, "", "t")] * 3))
    assert rc != 0 and proc.signals == [_signal.SIGTERM]
    assert "stopping head" in capsys.readouterr().err


def test_run_cluster_poller_survives_runner_oserror():
    proc = _PollProc(rc=0, exit_after=4)

    class _R(_SeqRun):
        def __call__(self, argv):
            if "container inspect -f" in " ".join(argv) and not self.calls:
                self.calls.append(list(argv))
                raise OSError("ssh vanished")
            return super().__call__(argv)
    assert _cluster_with_poll(proc, _R([])) == 0 and proc.signals == []


def test_run_cluster_head_exit_while_polling_returns_head_code():
    proc = _PollProc(rc=5, exit_after=3)
    run = _SeqRun([])
    assert _cluster_with_poll(proc, run) == 5
    assert run.calls[-1] == _STOP


def test_run_cluster_no_death_poll_after_signal(capsys):
    sig, handlers = _fake_signal()
    proc = _PollProc()
    calls = {"n": 0}

    class _R(_SeqRun):
        def __call__(self, argv):
            if "container inspect -f" in " ".join(argv):
                calls["n"] += 1
                # llama-swap stops the model: launcher gets SIGTERM (forwarded to
                # the head), cmdStop removes the worker.
                handlers[_signal.SIGTERM](_signal.SIGTERM, None)
                self.calls.append(list(argv))
                return (1, "", "Error: No such object: x-w")
            return super().__call__(argv)
    rc = _cluster_with_poll(proc, _R([]), sig=sig)
    assert rc == 128 + _signal.SIGTERM
    assert proc.signals == [_signal.SIGTERM]   # forwarded once, not re-sent
    assert "stopping head" not in capsys.readouterr().err


def test_main_cluster_passes_poll_argv(tmp_path, monkeypatch):
    _cluster_main_env(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "signal", types.SimpleNamespace(
        SIGTERM=_signal.SIGTERM, SIGINT=_signal.SIGINT, getsignal=lambda s: None,
        signal=lambda s, h: None))
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    got = {}
    monkeypatch.setattr(launch, "run_cluster", lambda *a, **k: got.update(k) or 0)
    with pytest.raises(SystemExit):
        launch.main(["--served-model-name", "big"])
    assert got["poll_argv"][0] == "ssh"
    assert got["poll_argv"][-1] == ("docker container inspect -f '{{.State.Running}}' "
                                    "big-5800-w")
    assert got["worker_name"] == "big-5800-w"
    assert got["poll_run"] is launch._run_poll


def test_run_poll_timeout_reads_as_ssh_failure(monkeypatch):
    def boom(*a, **k):
        assert k["timeout"] == 15
        raise _subprocess.TimeoutExpired("ssh", 15)
    monkeypatch.setattr(launch.subprocess, "run", boom)
    assert launch._run_poll(["ssh"])[0] == 255


@pytest.mark.parametrize("res,want", [
    ((0, "true\n", ""), "alive"),
    ((0, "Welcome to DGX OS\ntrue\n", ""), "alive"),   # shell noise before the answer
    ((0, "false\n", ""), "gone"),
    ((0, "motd\nfalse\n", ""), "gone"),
    ((0, "", ""), "unknown"),
    ((0, "garbage\n", ""), "unknown"),
    ((1, "", "Error: No such object: x-w"), "gone"),
    ((1, "", "Error response from daemon: No such container: x-w"), "gone"),
    ((1, "", "permission denied"), "unknown"),
    ((255, "", "Connection timed out"), "unknown"),
])
def test_worker_status_table(res, want):
    assert launch._worker_status(lambda a: res, ["ssh"]) == want


def test_run_cluster_garbage_poll_output_counts_as_miss_not_death(capsys):
    proc = _PollProc(rc=0, exit_after=4)
    rc = _cluster_with_poll(proc, _SeqRun([(0, "garbage\n", ""), (0, "", "")]))
    assert rc == 0 and proc.signals == []
    proc = _PollProc()
    rc = _cluster_with_poll(proc, _SeqRun([(0, "garbage\n", "")] * 3))
    assert rc != 0 and proc.signals == [_signal.SIGTERM]


# --- floor_warning: a single-node gmem.min floor over-reserving at tp=2 ---
def _floor_diag(tp=2, need=23.49, total=111.51):
    return {"tp": tp, "need": need, "total": total}


def test_floor_warning_when_floor_pins_tp2_far_above_need():
    w = launch.floor_warning(0.45, "sized", _floor_diag(), 0.45, 8.0)
    assert w.startswith("[auto-gmem] WARNING: gmem.min 0.45")
    assert "26.7GiB above the tp=2 estimate" in w  # 0.45 * 111.51 - 23.49


def test_floor_warning_quiet_when_estimate_decides():
    # need/total 0.30 > gmin 0.21: the estimate, not the floor, set gmem.
    assert launch.floor_warning(0.30, "sized", _floor_diag(need=33.45), 0.21, 8.0) is None


def test_floor_warning_quiet_under_threshold_single_node_or_fallback():
    assert launch.floor_warning(0.28, "sized", _floor_diag(need=27.0), 0.28, 8.0) is None
    assert launch.floor_warning(0.45, "sized", _floor_diag(tp=1), 0.45, 8.0) is None
    assert launch.floor_warning(0.45, "fallback", _floor_diag(), 0.45, 8.0) is None


@pytest.mark.parametrize("gmin, warns", [("0.45", True), ("0.05", False)])
def test_main_cluster_warns_when_floor_pins_gmem(tmp_path, monkeypatch, capsys, gmin, warns):
    # The fixture model needs ~4.5 GiB per node, so a 0.45 floor reserves ~45 more.
    _cluster_main_env(tmp_path, monkeypatch, GMEM_MIN=gmin)
    run = _Run([("docker image inspect", (0, "sha256:aaa\n", "")),
                ("cat /proc/meminfo", (0, _MEMINFO_ROOMY, ""))])
    monkeypatch.setattr(launch, "_run", run)
    launch.main(["--print"])
    assert ("[auto-gmem] WARNING: gmem.min" in capsys.readouterr().err) is warns
