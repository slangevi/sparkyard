"""Normalize raw model dicts into fully-resolved Model objects.

A Model merges per-engine `defaults` with per-model overrides and derives the
fields the templates consume (container path, host path, served names).
"""
import re
from dataclasses import dataclass

# A container-side `/models/<rel>` reference: not inside another path (e.g. not
# the tail of /data/models/x), up to a quote, backslash (escaped JSON),
# whitespace, comma or `}`. A preceding `--chat-template` (spaced or `=`) is
# captured so it can be skipped.
_MODELS_REF = re.compile(r"(--chat-template[=\s]+)?(?<![\w./-])/models/([^'\"\\\s,}]+)")
# In extra_docker_args, `/models/...` right after `:` is a bind-mount target
# (`-v host:/models/x:ro`), not a path under llm_root: skipped.
_MODELS_REF_DOCKER = re.compile(r"(--chat-template[=\s]+)?(?<![\w./:-])/models/([^'\"\\\s,}]+)")


@dataclass
class Model:
    name: str
    engine: str
    raw: dict
    defaults: dict

    def _get(self, key, fallback=None):
        if key in self.raw:
            return self.raw[key]
        if key in self.defaults:
            return self.defaults[key]
        return fallback

    # --- common ---
    @property
    def tier(self):
        return self.raw.get("tier")

    @property
    def ttl(self):
        return self._get("ttl", 3600)

    @property
    def ready_timeout(self):
        return self._get("ready_timeout", 600)

    @property
    def check_endpoint(self):
        return self._get("check_endpoint", "/health")

    @property
    def image(self):
        return self._get("image")

    @property
    def container(self):
        return self.raw["container"]

    @property
    def served_names(self):
        return [self.name] + list(self.raw.get("aliases", []))

    @property
    def litellm(self):
        params = dict(self.raw.get("litellm", {}))
        # drop_params strips reasoning_effort for an openai/ deployment of a model
        # LiteLLM does not know, so Claude Code's effortLevel never reached the
        # chat template. vLLM ignores the kwarg for templates that don't read it.
        # An explicit allowed_openai_params in models.yaml wins.
        if (self.engine == "vllm" and params.get("mode") != "embedding"
                and params.get("supports_reasoning") is not False):
            params.setdefault("allowed_openai_params", ["reasoning_effort"])
        return params

    @property
    def hf_repo(self):
        return self.raw.get("hf_repo")

    # --- vllm ---
    @property
    def model_path(self):
        return "/models/" + self.raw["path"]

    @property
    def model_host_path(self):
        if "host_path" in self.raw:
            return "/models/" + self.raw["host_path"]
        return self.model_path

    @property
    def max_model_len(self):
        return self.raw["max_model_len"]

    @property
    def max_num_seqs(self):
        return self.raw["max_num_seqs"]

    @property
    def kv_dtype_bytes(self):
        return self.raw["kv_dtype_bytes"]

    @property
    def gmem_min(self):
        return self.raw.get("gmem", {}).get("min", self.defaults.get("gmem_min"))

    @property
    def gmem_max(self):
        return self.raw.get("gmem", {}).get("max", self.defaults.get("gmem_max"))

    @property
    def gmem_override(self):
        return self.raw.get("gmem", {}).get("override")

    @property
    def nodes(self):
        return self.raw.get("nodes", 1)

    @property
    def safety_gib(self):
        return self._get("safety_gib", 6)

    @property
    def extra_docker_args(self):
        return self.raw.get("extra_docker_args")

    @property
    def pre_launch_cmd(self):
        return self.raw.get("pre_launch_cmd")

    @property
    def reclaim_containers(self):
        return self.raw.get("reclaim_containers") or []

    @property
    def chat_template(self):
        return self.raw.get("chat_template")

    @property
    def chat_template_path(self):
        ct = self.chat_template
        if ct is None:
            return None
        return ct if ct.startswith("/") else "/models/" + ct

    @property
    def vllm_flags(self):
        return list(self.raw.get("vllm_flags", []))

    @property
    def extra_paths(self):
        """Paths under llm_root (relative, no `/models/` prefix) that this model
        references besides its own `path` — e.g. a speculative-decoding draft
        model in `--speculative-config '{"model":"/models/..."}'`. Collected from
        every `/models/<rel>` in vllm_flags, extra_docker_args and pre_launch_cmd,
        deduped in first-seen order, trailing `/` stripped. A bind-mount target
        in extra_docker_args (`-v host:/models/x`) is not one.

        Excluded: the model's own `path` (and anything under it, which syncing
        the model already covers), and chat templates (the `chat_template` field
        or a `--chat-template` flag value) — only the API server on the head
        reads those, so a nodes: 2 worker never needs them."""
        own = str(self.raw.get("path", "")).strip("/")
        ct = self.chat_template
        skip = {ct.strip("/")} if isinstance(ct, str) and not ct.startswith("/") else set()
        if isinstance(ct, str) and ct.startswith("/models/"):
            skip.add(ct[len("/models/"):].strip("/"))
        sources = [(f, _MODELS_REF) for f in self.vllm_flags if isinstance(f, str)]
        for fld, rx in (("extra_docker_args", _MODELS_REF_DOCKER),
                        ("pre_launch_cmd", _MODELS_REF)):
            v = self.raw.get(fld)
            if isinstance(v, str):
                sources.append((v, rx))
        out = []
        for text, rx in sources:
            for m in rx.finditer(text):
                rel = m.group(2).rstrip("/")
                if m.group(1) or not rel or rel in skip:
                    continue
                if own and (rel == own or rel.startswith(own + "/")):
                    continue
                if rel not in out:
                    out.append(rel)
        return out

    # --- llamacpp ---
    @property
    def gguf(self):
        return "/models/" + self.raw["gguf"]

    @property
    def mount(self):
        return self.raw["mount"]

    @property
    def ctx_size(self):
        return self.raw["ctx_size"]

    @property
    def n_gpu_layers(self):
        return self._get("n_gpu_layers", 99)

    @property
    def parallel(self):
        return self._get("parallel", 1)

    @property
    def unified_memory(self):
        return self.raw.get("unified_memory", True)

    @property
    def no_mmap(self):
        """Emit `--load-mode none` (mmap is unsafe on GB10 unified memory).
        Named for the deprecated `--no-mmap` spelling it replaced."""
        return self.raw.get("no_mmap", True)

    @property
    def llamacpp_flags(self):
        return list(self.raw.get("llamacpp_flags", []))


def load_models(raw_config):
    """Build Model objects from a parsed models.yaml dict."""
    all_defaults = raw_config.get("defaults", {})
    models = []
    for entry in raw_config.get("models", []):
        engine = entry["engine"]
        models.append(Model(
            name=entry["name"],
            engine=engine,
            raw=entry,
            defaults=all_defaults.get(engine, {}),
        ))
    return models
