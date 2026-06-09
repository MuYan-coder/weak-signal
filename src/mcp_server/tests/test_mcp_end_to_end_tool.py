import pytest
from unittest.mock import patch
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import run_end_to_end_analysis_tool
from pathlib import Path

@pytest.fixture
def mock_settings(tmp_path):
    return WeakSignalMCPSettings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        result_dir=tmp_path / "result",
        memory_dir=tmp_path / "memory"
    )

@patch("src.mcp_server.service.prepare_domain_pack_tool")
@patch("src.mcp_server.service.get_recommended_source_counts_tool")
@patch("src.mcp_server.service.run_analysis_tool")
def test_run_end_to_end_analysis_tool_success(
    mock_run_analysis,
    mock_get_counts,
    mock_prepare,
    mock_settings
):
    # Setup mocks
    mock_prepare.return_value = {
        "status": "ready",
        "domain_pack_id": "test_pack_123",
        "domain_pack_hash": "hash123",
    }
    
    mock_get_counts.return_value = {
        "total_count": 100,
        "final_counts": {"paper": 20, "patent": 20}
    }
    
    mock_run_analysis.return_value = {
        "result_dir": "result/20260101_120000",
        "event_count": 50,
        "candidate_count": 25,
        "signal_count": 10,
        "weak_signal_count": 2,
        "report_preview": "This is a mock report preview.",
        "resources": {
            "report.txt": "weak-signal://results/20260101_120000/report"
        }
    }

    # Execute
    result = run_end_to_end_analysis_tool(
        field_name="固态电池",
        keywords=["固态", "电解质"],
        sample_size=40,
        settings=mock_settings
    )

    # Verify
    assert "error" not in result
    assert result["status"] == "success"
    assert result["domain_pack_id"] == "test_pack_123"
    assert result["summary"]["weak_signal_count"] == 2
    assert "This is a mock report preview" in result["report_preview"]
    assert "report.txt" in result["resources"]

    # Verify mock calls
    mock_prepare.assert_called_once()
    assert mock_prepare.call_args[1]["field_name"] == "固态电池"
    
    mock_get_counts.assert_called_once()
    assert mock_get_counts.call_args[1]["pack_ref"] == "test_pack_123"
    
    mock_run_analysis.assert_called_once()
    assert mock_run_analysis.call_args[1]["domain_pack_ref"] == "test_pack_123"
    assert mock_run_analysis.call_args[1]["sample_size"] == 40
    assert mock_run_analysis.call_args[1]["source_config"] == {"sources": ["paper", "patent"], "counts": {"paper": 20, "patent": 20}}

@patch("src.mcp_server.service.prepare_domain_pack_tool")
def test_run_end_to_end_analysis_tool_prepare_error(
    mock_prepare,
    mock_settings
):
    mock_prepare.return_value = {"error": "Mock prepare error"}
    
    result = run_end_to_end_analysis_tool(
        field_name="固态电池",
        keywords=["固态", "电解质"],
        settings=mock_settings
    )
    
    assert "error" in result
    assert "Mock prepare error" in result["error"]
