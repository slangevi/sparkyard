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
