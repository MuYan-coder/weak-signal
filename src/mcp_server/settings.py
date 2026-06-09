from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class WeakSignalMCPSettings:
    project_root: Path
    data_dir: Path | None = None
    result_dir: Path | None = None
    memory_dir: Path | None = None
    config_domain_pack_dir: Path | None = None
    tool_prefix: str = "weak_signal_"
    resource_scheme: str = "weak-signal"
    max_sync_sample_size: int = 100
    max_job_sample_size: int = 1000
    max_resource_chars: int = 50000
    enable_db_backend: bool = False
    enable_llm: bool = True
    default_domain_pack: str = "neutral"
    job_backend: Literal["in_process", "external"] = "in_process"

    def __post_init__(self):
        # Data classes with frozen=True need special handling for setting attributes in post_init
        # We bypass frozen check by using object.__setattr__
        if self.data_dir is None:
            object.__setattr__(self, 'data_dir', self.project_root / "data")
        if self.result_dir is None:
            object.__setattr__(self, 'result_dir', self.project_root / "result")
        if self.memory_dir is None:
            object.__setattr__(self, 'memory_dir', self.project_root / "memory")
        if self.config_domain_pack_dir is None:
            object.__setattr__(self, 'config_domain_pack_dir', self.project_root / "src" / "config" / "domain_packs")

    @classmethod
    def from_env(cls) -> "WeakSignalMCPSettings":
        import os
        # Attempt to get project root from env, or default to current working directory
        project_root = Path(os.getenv("WEAK_SIGNAL_PROJECT_ROOT", Path.cwd())).resolve()
        return cls(project_root=project_root)
