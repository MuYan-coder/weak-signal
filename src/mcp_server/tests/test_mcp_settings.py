from pathlib import Path
from src.mcp_server.settings import WeakSignalMCPSettings

def test_settings_default_directory_resolution():
    root = Path("/tmp/my_project")
    settings = WeakSignalMCPSettings(project_root=root)
    
    assert settings.data_dir == root / "data"
    assert settings.result_dir == root / "result"
    assert settings.memory_dir == root / "memory"
    assert settings.config_domain_pack_dir == root / "src" / "config" / "domain_packs"
    assert settings.tool_prefix == "weak_signal_"
    assert settings.max_sync_sample_size == 100
