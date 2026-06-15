"""Normalize raw model dicts into fully-resolved Model objects.

A Model merges per-engine `defaults` with per-model overrides and derives the
fields the templates consume (container path, host path, served names).
"""
from dataclasses import dataclass


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
        return self.raw.get("litellm", {})

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
    def safety_gib(self):
        return self._get("safety_gib", 6)

    @property
    def extra_docker_args(self):
        return self.raw.get("extra_docker_args")

    @property
    def pre_launch_cmd(self):
        return self.raw.get("pre_launch_cmd")

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
