"""Machine settings loaded from settings.local.yaml."""
from dataclasses import dataclass
from typing import Optional
import yaml

from .placeholders import resolve

# vLLM build defaults — used when settings.local.yaml has no `vllm:` block (or a
# partial one). The ref mirrors vllm/VLLM_NODE_PROVENANCE.md; bump both together.
DEFAULT_VLLM_UPSTREAM = "https://github.com/eugr/spark-vllm-docker"
DEFAULT_VLLM_CLONE_PATH = "{repo_path}/vllm/build/spark-vllm-docker"
DEFAULT_VLLM_REF = "7852e50e4"


@dataclass
class VllmBuild:
    upstream: str
    clone_path: str   # placeholders already resolved (absolute)
    vllm_ref: str


@dataclass
class Settings:
    llm_root: str
    repo_path: str
    home: str = ""              # optional; used by {home} in user mounts
    vllm: Optional[VllmBuild] = None   # set in load(); None only if constructed directly

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        s = cls(
            llm_root=data["llm_root"],
            repo_path=data["repo_path"],
            home=data.get("home", ""),
        )
        v = data.get("vllm") or {}
        s.vllm = VllmBuild(
            upstream=v.get("upstream", DEFAULT_VLLM_UPSTREAM),
            clone_path=resolve(v.get("clone_path", DEFAULT_VLLM_CLONE_PATH),
                               s.placeholder_map()),
            vllm_ref=v.get("vllm_ref", DEFAULT_VLLM_REF),
        )
        return s

    def placeholder_map(self):
        return {
            "llm_root": self.llm_root,
            "repo_path": self.repo_path,
            "home": self.home,
        }
