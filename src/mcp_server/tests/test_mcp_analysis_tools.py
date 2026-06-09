import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import run_analysis_tool, start_analysis_job_tool

@pytest.fixture
def settings():
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create a dummy data file
    dummy_data = data_dir / "test_data.xlsx"
    pd.DataFrame({"test": [1, 2]}).to_excel(dummy_data, index=False)
    
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    
    yield WeakSignalMCPSettings(
        project_root=tmp_path,
        data_dir=data_dir,
        result_dir=result_dir
    )
    
    shutil.rmtree(tmp_path, ignore_errors=True)

@patch("src.core.pipeline.AnalysisPipeline.run_full_pipeline")
def test_run_analysis_tool_mock(mock_run, settings):
    # Setup mock result
    mock_run.return_value = {
        "result_dir": settings.result_dir / "20260608_120000",
        "domain_context": MagicMock(domain_pack_id="neutral", domain_pack_hash="hash_123"),
        "events_df": pd.DataFrame({"id": [1, 2]}),
        "candidate_forms_df": pd.DataFrame({"id": [1]}),
        "signals_df": pd.DataFrame({"signal_type": ["weak_signal", "hotspot"]}),
        "report": {"preview": "This is a preview"}
    }
    
    result = run_analysis_tool(
        sample_size=10,
        data_path="data/test_data.xlsx",
        settings=settings
    )
    
    assert "error" not in result
    assert result["domain_pack_id"] == "neutral"
    assert result["event_count"] == 2
    assert result["candidate_count"] == 1
    assert result["signal_count"] == 2
    assert result["weak_signal_count"] == 1
    assert "resources" in result

def test_run_analysis_tool_limit(settings):
    result = run_analysis_tool(
        sample_size=200,
        settings=settings
    )
    
    assert "error" in result
    assert "exceeds synchronous limit" in result["error"]

def test_start_analysis_job_tool(settings):
    result = start_analysis_job_tool(sample_size=200, settings=settings)
    assert "error" not in result
    assert result["job_id"] == "job_12345"
    assert result["status"] == "queued"
