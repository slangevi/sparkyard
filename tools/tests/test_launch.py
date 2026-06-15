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
    n_layers, n_heads, n_kv, head_dim = launch.parse_config(d)
    assert (n_layers, n_heads, n_kv, head_dim) == (32, 32, 8, 128)  # head_dim = 4096/32


def test_parse_config_mha_fallback_when_no_kv(tmp_path):
    cfg = {"num_hidden_layers": 16, "num_attention_heads": 16, "head_dim": 64}
    d = _model_dir(tmp_path, cfg, 1)
    _, n_heads, n_kv, head_dim = launch.parse_config(d)
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
    _, _, n_kv, head_dim = launch.parse_config(d)
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
