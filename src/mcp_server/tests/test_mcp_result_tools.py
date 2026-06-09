import pytest
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path

from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import read_result_resource_tool, list_runs_tool

@pytest.fixture
def settings():
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    
    # Create a mock run directory
    run_dir = result_dir / "20260608_120000"
    run_dir.mkdir()
    
    # Create some mock resources
    (run_dir / "report.txt").write_text("Mock Report Content", encoding="utf-8")
    
    signals_data = [{"signal_id": "WS001", "signal_type": "weak_signal"}]
    with open(run_dir / "signals.json", "w", encoding="utf-8") as f:
        json.dump(signals_data, f)
        
    events_df = pd.DataFrame({"id": ["event_1"], "title": ["Event 1"]})
    events_df.to_csv(run_dir / "events.csv", index=False)
    
    yield WeakSignalMCPSettings(
        project_root=tmp_path,
        result_dir=result_dir
    )
    
    shutil.rmtree(tmp_path, ignore_errors=True)

def test_list_runs_tool(settings):
    result = list_runs_tool(limit=10, settings=settings)
    assert "error" not in result
    runs = result["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "20260608_120000"
    assert "report" in runs[0]["resources"]

def test_read_result_resource_tool_txt(settings):
    uri = "weak-signal://results/20260608_120000/report"
    result = read_result_resource_tool(uri, settings)
    assert "error" not in result
    assert result["resource_type"] == "report"
    assert result["data"] == "Mock Report Content"

def test_read_result_resource_tool_json(settings):
    uri = "weak-signal://results/20260608_120000/signals"
    result = read_result_resource_tool(uri, settings)
    assert "error" not in result
    assert result["resource_type"] == "signals"
    assert len(result["data"]) == 1
    assert result["data"][0]["signal_id"] == "WS001"

def test_read_result_resource_tool_csv_fallback(settings):
    uri = "weak-signal://results/20260608_120000/events"
    result = read_result_resource_tool(uri, settings)
    assert "error" not in result
    assert result["resource_type"] == "events"
    assert len(result["data"]) == 1
    assert result["data"][0]["id"] == "event_1"

def test_read_result_resource_tool_invalid_uri(settings):
    uri = "weak-signal://results/invalid_uri"
    result = read_result_resource_tool(uri, settings)
    assert "error" in result
    assert "Malformed resource URI" in result["error"]
