import pytest
from pathlib import Path
from src.mcp_server.settings import WeakSignalMCPSettings
from src.mcp_server.path_guard import (
    assert_allowed_artifact,
    resolve_safe_path,
    resolve_result_dir
)

import tempfile
import shutil

@pytest.fixture
def test_settings():
    # 使用项目根目录下的临时文件夹来绕过 Windows 系统临时目录挂载异常问题
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=base_dir))
    
    # 创建模拟的目录结构
    data_dir = tmp_dir / "data"
    result_dir = tmp_dir / "result"
    memory_dir = tmp_dir / "memory"
    
    data_dir.mkdir()
    result_dir.mkdir()
    memory_dir.mkdir()
    
    # 模拟一个运行目录
    (result_dir / "20260608_120000").mkdir()
    (result_dir / "20260608_120000" / "report.txt").touch()
    
    # 模拟数据文件
    (data_dir / "专利测试数据.xlsx").touch()
    
    yield WeakSignalMCPSettings(
        project_root=tmp_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        memory_dir=memory_dir
    )
    
    # 清理
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_resolve_safe_path_allowed(test_settings):
    # 允许 data/专利测试数据.xlsx
    allowed_roots = [test_settings.data_dir, test_settings.result_dir]
    path = resolve_safe_path("data/专利测试数据.xlsx", test_settings, allowed_roots, must_exist=True)
    assert path.name == "专利测试数据.xlsx"

    # 允许 result/<run_id>/report.txt
    path2 = resolve_safe_path("result/20260608_120000/report.txt", test_settings, allowed_roots, must_exist=True)
    assert path2.name == "report.txt"

def test_resolve_safe_path_denied(test_settings):
    allowed_roots = [test_settings.data_dir]
    
    # 拒绝项目根目录的 .env (即使存在)
    env_file = test_settings.project_root / ".env"
    env_file.touch()
    with pytest.raises(PermissionError):
        resolve_safe_path(".env", test_settings, allowed_roots, must_exist=False)
        
    # 拒绝逃逸路径 ../README.md
    with pytest.raises(PermissionError):
        resolve_safe_path("../README.md", test_settings, allowed_roots, must_exist=False)
        
    # 拒绝项目外绝对路径
    with pytest.raises(PermissionError):
        resolve_safe_path("/etc/passwd", test_settings, allowed_roots, must_exist=False)
        # Windows 的绝对路径
        resolve_safe_path("C:/Windows/System32/cmd.exe", test_settings, allowed_roots, must_exist=False)

def test_resolve_safe_path_custom_settings():
    # 自定义配置，测试路径守卫使用 settings 中的目录
    base_dir = Path(__file__).parent.parent.parent.parent / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(dir=base_dir))
    
    custom_data_dir = tmp_path / "custom_data"
    custom_data_dir.mkdir()
    (custom_data_dir / "test.xlsx").touch()
    
    settings = WeakSignalMCPSettings(
        project_root=tmp_path,
        data_dir=custom_data_dir
    )
    
    # 测试自定义目录允许
    path = resolve_safe_path(custom_data_dir / "test.xlsx", settings, [settings.data_dir], must_exist=True)
    assert path.name == "test.xlsx"
    
    # 默认 data 目录现在不在允许列表
    default_data = tmp_path / "data"
    default_data.mkdir()
    (default_data / "test2.xlsx").touch()
    with pytest.raises(PermissionError):
        resolve_safe_path(default_data / "test2.xlsx", settings, [settings.data_dir], must_exist=True)
        
    shutil.rmtree(tmp_path, ignore_errors=True)

def test_resolve_result_dir(test_settings):
    # 测试普通 run_id
    path = resolve_result_dir("20260608_120000", test_settings)
    assert path == test_settings.result_dir / "20260608_120000"
    
    # 测试不存在的 run_id
    with pytest.raises(FileNotFoundError):
        resolve_result_dir("non_existent", test_settings)
        
    # 测试非法路径注入
    with pytest.raises(PermissionError):
        resolve_result_dir("../data", test_settings)

def test_assert_allowed_artifact():
    assert_allowed_artifact("report.txt")
    assert_allowed_artifact("events.json")
    assert_allowed_artifact("domain_pack.yaml")
    
    with pytest.raises(ValueError):
        assert_allowed_artifact("unauthorized.txt")
    
    with pytest.raises(ValueError):
        assert_allowed_artifact(".env")
