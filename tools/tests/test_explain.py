import types
from sparkyard import explain


def _cfg(cmd):
    return {"models": {"M": {"cmd": cmd, "cmdStop": "docker stop x"}}}


CMD = ("env MODEL_PATH=/models/vllm/o/M CONTAINER_NAME=vllm-m-${PORT}\n"
       "IMAGE=vllm-node:latest PORT=${PORT} HOST=${host}\n"
       "MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 KV_DTYPE_BYTES=1\n"
       "GMEM_MIN=0.35 GMEM_MAX=0.45 SAFETY_GIB=4\n"
       "python3 /app/scripts/launch.py --quantization compressed-tensors\n")


def test_builds_a_print_invocation_for_the_named_model():
    argv = explain.print_argv(_cfg(CMD), "M", port="5999")
    joined = " ".join(argv)
    assert "--print" in joined, "must dry-run, never launch"
    assert "launch.py" in joined
    assert "${PORT}" not in joined and "${host}" not in joined, "placeholders unresolved"
    assert "5999" in joined


def test_unknown_model_is_fail_closed():
    try:
        explain.print_argv(_cfg(CMD), "nope")
    except explain.ExplainError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("unknown model must raise")


def test_llamacpp_models_have_no_launcher_to_dry_run():
    cfg = {"models": {"G": {"cmd": "docker run --rm --name llamacpp-g ... -m /models/x.gguf"}}}
    try:
        explain.print_argv(cfg, "G")
    except explain.ExplainError as e:
        assert "launch" in str(e).lower() or "vllm" in str(e).lower()
    else:
        raise AssertionError("a non-launcher model must say so, not emit junk")


def test_summarise_extracts_the_sizing_lines():
    out = explain.summarise(
        "[auto-gmem] cfg: layers=64 attn_layers=16 kv_heads=4\n"
        "[auto-gmem] weights=21.81GiB kv=16.00GiB safety=4GiB → need=41.81GiB\n"
        "gpu_memory_utilization=0.37\n"
        "docker\nrun\n--rm\n")
    assert "attn_layers=16" in out and "need=41.81GiB" in out
    assert "0.37" in out
