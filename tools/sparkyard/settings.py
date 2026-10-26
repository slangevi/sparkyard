"""Machine settings loaded from settings.local.yaml."""
import os
from dataclasses import dataclass, field
from typing import Optional
import yaml

from .placeholders import resolve

# vLLM build defaults — used when settings.local.yaml has no `vllm:` block (or a
# partial one). The ref mirrors vllm/VLLM_NODE_PROVENANCE.md; bump both together.
DEFAULT_VLLM_UPSTREAM = "https://github.com/eugr/spark-vllm-docker"
DEFAULT_VLLM_CLONE_PATH = "{repo_path}/vllm/build/spark-vllm-docker"
DEFAULT_VLLM_REF = "040700aaa"


@dataclass
class VllmBuild:
    upstream: str
    clone_path: str   # placeholders already resolved (absolute)
    vllm_ref: str


@dataclass
class ClusterWorker:
    name: str
    ssh_host: str
    ssh_user: str
    cx7_ip: str


@dataclass
class Cluster:
    head_cx7_ip: str
    eth_if: str
    ib_hca: str
    workers: list = field(default_factory=list)   # list[ClusterWorker]
    master_port: int = 29501

    @property
    def worker(self) -> ClusterWorker:
        return self.workers[0]


@dataclass
class Settings:
    llm_root: str
    repo_path: str
    home: str = ""              # optional; used by {home} in user mounts
    vllm: Optional[VllmBuild] = None   # set in load(); None only if constructed directly
    cluster: Optional[Cluster] = None  # set in load() when a `cluster:` block exists

    @property
    def cluster_dir(self) -> str:
        return os.path.join(self.repo_path, "llama-swap", "cluster")

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
        c = data.get("cluster")
        if c is not None:
            s.cluster = Cluster(
                head_cx7_ip=c["head_cx7_ip"],
                eth_if=c["eth_if"],
                ib_hca=c["ib_hca"],
                workers=[ClusterWorker(name=w["name"], ssh_host=w["ssh_host"],
                                       ssh_user=w["ssh_user"], cx7_ip=w["cx7_ip"])
                         for w in c["workers"]],
                master_port=c.get("master_port", 29501),
            )
        return s

    def placeholder_map(self):
        return {
            "llm_root": self.llm_root,
            "repo_path": self.repo_path,
            "home": self.home,
        }
