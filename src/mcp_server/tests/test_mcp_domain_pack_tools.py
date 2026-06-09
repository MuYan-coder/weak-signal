import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.service import (
    validate_domain_pack_tool,
    get_recommended_source_counts_tool,
    prepare_domain_pack_tool
)

@pytest.fixture
def settings():
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    
    yield WeakSignalMCPSettings(
        project_root=tmp_path,
        memory_dir=memory_dir,
        result_dir=result_dir
    )
    
    shutil.rmtree(tmp_path, ignore_errors=True)

def test_validate_domain_pack_neutral(settings):
    # neutral 是预置包，可以直接加载
    result = validate_domain_pack_tool("neutral", require_dry_run=False, settings=settings)
    assert "error" not in result
    assert result.get("domain_pack_id") == "neutral"
    assert result.get("is_valid") is False # Neutral pack 校验通常失败，因为它缺少必要的自定义字段

@patch("src.data_access.repository.DataRepository")
def test_get_recommended_source_counts_mock(mock_repo_cls, settings):
    # Mock database
    mock_repo = MagicMock()
    # 模拟 get_source_counts 返回数据
    mock_repo.get_source_counts.return_value = {"paper": 100, "patent": 50, "news": 0}
    mock_repo_cls.from_env.return_value = mock_repo
    
    result = get_recommended_source_counts_tool(
        pack_ref="neutral",
        total_sample_size=10,
        source_types=["paper", "patent", "news"],
        backend="mock",
        settings=settings
    )
    
    assert "error" not in result
    assert result["backend"] == "mock"
    assert "final_counts" in result
    # neutral 没有权重，应该能退化出平均或0的分配
    assert "paper" in result["final_counts"]

@patch("src.domain.domain_pack_workflow.DomainPackWorkflow")
def test_prepare_domain_pack_mock(mock_workflow_cls, settings):
    mock_workflow = MagicMock()
    mock_result = MagicMock()
    mock_result.status = "blocked"
    mock_result.manual_review_required = True
    mock_result.domain_pack.pack_id = "test_robot"
    mock_result.domain_pack.domain_pack_hash = "abcdefg"
    mock_result.domain_pack.domain_pack_version = "v1"
    
    # Validation report mock
    mock_val_report = MagicMock()
    mock_val_report.to_dict.return_value = {"is_valid": False, "errors": ["test"]}
    mock_result.validation_report = mock_val_report
    mock_result.review_hints = ["hint1"]
    
    mock_workflow.prepare_domain_pack.return_value = mock_result
    mock_workflow_cls.return_value = mock_workflow
    
    result = prepare_domain_pack_tool(
        field_id="test_robot",
        field_name="测试机器人",
        keywords=["robot"],
        settings=settings
    )
    
    assert "error" not in result
    assert result["domain_pack_id"] == "test_robot"
    assert result["status"] == "blocked"
    assert "validation" in result
    assert result["review_hints"] == ["hint1"]
