"""Machine settings loaded from settings.local.yaml."""
from dataclasses import dataclass
import yaml


@dataclass
class Settings:
    llm_root: str
    repo_path: str
    home: str = ""              # optional; used by {home} in user mounts

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            llm_root=data["llm_root"],
            repo_path=data["repo_path"],
            home=data.get("home", ""),
        )

    def placeholder_map(self):
        return {
            "llm_root": self.llm_root,
            "repo_path": self.repo_path,
            "home": self.home,
        }
