import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.serialization import (
    dataframe_summary,
    to_jsonable,
    result_resources,
    report_preview
)

@pytest.fixture
def settings():
    return WeakSignalMCPSettings(project_root=Path("/tmp/root"))

def test_dataframe_summary():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["a", "b", "c", "d", "e", "f"]
    })
    summary = dataframe_summary(df, preview_rows=2)
    assert summary["row_count"] == 6
    assert summary["columns"] == ["id", "name"]
    assert len(summary["preview"]) == 2
    assert summary["preview"][0]["id"] == 1

def test_to_jsonable_path(settings):
    p = Path("/tmp/root/data/file.txt")
    jsonable = to_jsonable(p, settings)
    assert jsonable == "data/file.txt"
    
def test_to_jsonable_numpy():
    data = {
        "int": np.int64(42),
        "float": np.float32(3.14),
        "bool": np.bool_(True),
        "nan": np.nan,
        "arr": np.array([1, 2, 3])
    }
    result = to_jsonable(data)
    assert isinstance(result["int"], int)
    assert result["int"] == 42
    assert isinstance(result["float"], float)
    assert isinstance(result["bool"], bool)
    assert result["bool"] is True
    assert result["nan"] is None
    assert result["arr"] == [1, 2, 3]

def test_to_jsonable_dataframe():
    df = pd.DataFrame({"A": [1, 2]})
    result = to_jsonable(df)
    assert "row_count" in result
    assert result["row_count"] == 2

def test_result_resources(settings):
    uris = result_resources("run123", settings)
    assert f"{settings.resource_scheme}://results/run123/report" in uris
    assert len(uris) == 5

def test_report_preview():
    short_text = "abc"
    assert report_preview(short_text, 10) == "abc"
    
    long_text = "a" * 100
    preview = report_preview(long_text, 10)
    assert preview.startswith("a" * 10)
    assert "Truncated" in preview
