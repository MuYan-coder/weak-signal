import pytest
from pathlib import Path
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import list_data_sources

import tempfile
import shutil

@pytest.fixture
def settings():
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    (data_dir / "专利测试数据.xlsx").touch()
    (data_dir / "无关文件.txt").touch() # should be ignored
    
    nested_dir = data_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "test_report.csv").touch()
    
    yield WeakSignalMCPSettings(project_root=tmp_path, data_dir=data_dir)
    
    shutil.rmtree(tmp_path, ignore_errors=True)

def test_list_data_sources_flat(settings):
    result = list_data_sources(settings, include_nested=False)
    assert result["data_dir"] == str(settings.data_dir).replace("\\", "/")
    
    files = result["files"]
    assert len(files) == 1
    assert files[0]["name"] == "专利测试数据.xlsx"
    assert files[0]["source_type_hint"] == "patent"

def test_list_data_sources_nested(settings):
    result = list_data_sources(settings, include_nested=True)
    files = result["files"]
    assert len(files) == 2
    names = [f["name"] for f in files]
    assert "专利测试数据.xlsx" in names
    assert "test_report.csv" in names
    
    for f in files:
        if f["name"] == "test_report.csv":
            assert f["source_type_hint"] == "report"

def test_list_data_sources_empty_dir():
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    empty_data_dir = tmp_path / "empty_data"
    settings = WeakSignalMCPSettings(project_root=tmp_path, data_dir=empty_data_dir)
    
    # Not created yet
    result = list_data_sources(settings)
    assert result["files"] == []
    
    # Created but empty
    empty_data_dir.mkdir()
    result = list_data_sources(settings)
    assert result["files"] == []
    
    shutil.rmtree(tmp_path, ignore_errors=True)
