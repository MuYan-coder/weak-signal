import os
import sys
import queue
import threading
import traceback
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings('ignore')

import json
import altair as alt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="产业技术弱信号识别系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-active {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .stage-complete {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .stage-pending {
        background-color: #f5f5f5;
        border-left: 4px solid #9e9e9e;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        opacity: 0.6;
    }
    .log-container {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
    }
    .log-entry {
        padding: 5px 0;
        border-bottom: 1px solid #eee;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.pipeline import AnalysisPipeline
from src.core.agent import TechForesightAgent
from src.data_access.models import SourceQuery
from src.data_access.repository import DataRepository
from src.domain import (
    DomainContext,
    DomainPackGenerationRequest,
    DomainPackGenerator,
    build_domain_pack_review_state,
    is_current_generated_domain_pack,
    load_domain_pack,
    recommend_source_counts,
    validate_domain_pack,
)
from src.utils.config import Config
from src.utils.env_config import ensure_env_loaded
from src.utils.llm_client import get_provider_and_client
from src.validation.event_quality import merge_event_quality_into_raw_data
from src.validation.temporal_validator import merge_temporal_validation_into_candidates

ensure_env_loaded()
Config.ensure_dirs()

STAGE_NAMES = [
    "数据加载",
    "事件抽取与质量评分",
    "候选成形",
    "弱信号评分",
    "主题细化",
    "反向验证",
    "时间验证",
    "信号生成",
    "报告生成",
]

DB_SOURCE_TYPES = ["paper", "news", "policy", "report", "patent"]
DB_RAW_SOURCE_TYPES = ["literature", "consulting", "policy", "report", "patent"]
DB_SOURCE_LABELS = {
    "paper": "文献",
    "news": "资讯",
    "policy": "政策",
    "report": "研报",
    "patent": "专利",
}
DB_DEFAULT_SAMPLE_TOTAL = 240

DOMAIN_SEARCH_TEMPLATES = {
    "自定义领域": {
        "field_id": "",
        "field_name": "",
        "keywords": "",
        "synonyms": "",
        "exclude_terms": "招聘, 培训, 广告",
        "preset_pack_ref": "",
    },
    "人形机器人 preset": {
        "field_id": "humanoid_robot",
        "field_name": "人形机器人",
        "keywords": "人形机器人, 人型机器人, 仿人机器人, 双足机器人, 仿生机器人, 足式机器人, 通用机器人, 具身机器人, 服务机器人, humanoid robot, bipedal robot, anthropomorphic robot, android robot, legged robot, general-purpose robot, embodied robot",
        "synonyms": "双足行走, 步态规划, 步态生成, 运动控制, 动态平衡, 全身控制, 质心动力学, 零力矩点, ZMP, 落脚点规划, 接触规划, 多接触运动, 地形适应, 抗扰控制, humanoid robotics, bipedal locomotion, gait planning, gait generation, locomotion control, dynamic balance, whole-body control, centroidal dynamics, zero moment point, capture point, footstep planning, multi-contact locomotion, terrain adaptation, disturbance rejection, 全身运动规划, 逆运动学, 逆动力学, 力矩控制, 阻抗控制, 导纳控制, 接触力控制, 模型预测控制, 最优控制, 运动规划, 机器人操作, 灵巧操作, 灵巧手, 抓取规划, 双臂协作, 手眼协调, 移动操作, 全身操作, 接触丰富操作, 力控抓取",
        "exclude_terms": "招聘, 培训, 广告, 课程, 招生, 销售",
        "preset_pack_ref": "humanoid_robot",
    },
    "具身智能": {
        "field_id": "embodied_ai",
        "field_name": "具身智能",
        "keywords": "具身智能, embodied intelligence, VLA, 灵巧手, 双足机器人",
        "synonyms": "vision-language-action, 机器人基础模型, world model, 脑体协同",
        "exclude_terms": "招聘, 培训, 广告, 课程",
        "preset_pack_ref": "",
    },
    "世界模型": {
        "field_id": "world_model",
        "field_name": "世界模型",
        "keywords": "世界模型, world model, 物理世界仿真, 生成式视频",
        "synonyms": "physical simulation, video generation model",
        "exclude_terms": "招聘, 培训, 广告, 课程",
        "preset_pack_ref": "",
    },
    "太空制造": {
        "field_id": "space_manufacturing",
        "field_name": "太空制造",
        "keywords": "太空制造, 空间制造, 轨道制造, 零重力制造, 在轨服务, 空间工厂, 太空3D打印",
        "synonyms": "space manufacturing, in-orbit manufacturing, zero-gravity manufacturing, in-space manufacturing, orbital factory, space 3D printing, 空间装配, 空间机器人操作",
        "exclude_terms": "招聘, 培训, 广告, 科幻电影, 游戏",
        "preset_pack_ref": "",
    },
    "工业软件": {
        "field_id": "industrial_software",
        "field_name": "工业软件",
        "keywords": "工业软件, CAD, CAM, CAE, PLM, MES, ERP, 工业互联网, 数字孪生, 制造执行系统",
        "synonyms": "industrial software, computer-aided design, computer-aided engineering, product lifecycle management, manufacturing execution system, digital twin, 工业自动化软件, 研发设计类软件, 生产制造类软件, 运维服务类软件",
        "exclude_terms": "招聘, 培训, 广告, 销售, 代理",
        "preset_pack_ref": "",
    },
    "智能传感器": {
        "field_id": "smart_sensors",
        "field_name": "智能传感器",
        "keywords": "智能传感器, 柔性传感器, 激光雷达, 视觉传感器, 触觉传感器, 惯性传感器, 仿生传感器, MEMS",
        "synonyms": "smart sensor, flexible sensor, LiDAR, vision sensor, tactile sensor, inertial sensor, biomimetic sensor, micro-electromechanical systems, 边缘计算传感器, 智能感知, 多模态感知",
        "exclude_terms": "招聘, 培训, 广告, 手机评测, 相机评测",
        "preset_pack_ref": "",
    },
    "新型电子材料": {
        "field_id": "new_electronic_materials",
        "field_name": "新型电子材料",
        "keywords": "新型电子材料, 半导体材料, 宽禁带半导体, 碳化硅, 氮化镓, 二维材料, 钙钛矿, 拓扑绝缘体",
        "synonyms": "new electronic materials, semiconductor materials, wide bandgap semiconductor, SiC, GaN, 2D materials, perovskite, topological insulator, 量子材料, 柔性电子材料, 纳米电子材料",
        "exclude_terms": "招聘, 培训, 广告, 股票分析, 行情",
        "preset_pack_ref": "",
    },
}

def get_fragment_decorator(run_every: Optional[str] = None):
    """兼容不同 Streamlit 版本的 fragment 装饰器。"""
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        return fragment(run_every=run_every)

    experimental_fragment = getattr(st, "experimental_fragment", None)
    if experimental_fragment is not None:
        return experimental_fragment(run_every=run_every)

    def passthrough(func):
        return func

    return passthrough

def init_analysis_state():
    """初始化分析任务状态"""
    defaults = {
        "analysis_running": False,
        "analysis_stop_requested": False,
        "analysis_logs": [],
        "analysis_progress": 0,
        "analysis_stage_name": "",
        "analysis_stage_status": "pending",
        "analysis_results": None,
        "analysis_error": None,
        "analysis_queue": None,
        "analysis_thread": None,
        "analysis_cancel_event": None,
        "analysis_outcome": "idle",
        "analysis_full_refresh_requested": False,
        "analysis_run_description": "",
        "domain_pack_current": None,
        "domain_pack_validation_report": None,
        "domain_pack_analysis_confirmed": False,
        "domain_pack_confirmed_counts": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def append_ui_log(message, level="info"):
    """向页面状态追加日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.analysis_logs.append({
        "time": timestamp,
        "message": message,
        "level": level,
    })

def render_logs(log_container, logs):
    """渲染日志面板"""
    with log_container:
        log_html = ""
        for log in logs[-100:]:
            color = {
                "error": "#f44336",
                "success": "#4caf50",
                "warning": "#ff9800",
                "info": "#2196f3",
            }.get(log["level"], "#666")
            icon = {
                "error": "❌",
                "success": "✅",
                "warning": "⚠️",
                "info": "ℹ️",
            }.get(log["level"], "ℹ️")
            log_html += (
                f"<div class='log-entry'><span style='color: #666'>[{log['time']}]</span> "
                f"{icon} <span style='color: {color}'>{log['message']}</span></div>"
            )
        st.markdown(f"<div class='log-container'>{log_html}</div>", unsafe_allow_html=True)

def render_stage_status(status_container, stage_name, status, progress):
    """渲染阶段状态"""
    with status_container:
        stages_html = ""
        for i, name in enumerate(STAGE_NAMES):
            if progress >= len(STAGE_NAMES):
                css_class = "stage-complete"
                icon = "✅"
            elif i < progress:
                css_class = "stage-complete"
                icon = "✅"
            elif name == stage_name and status == "active":
                css_class = "stage-active"
                icon = "🔄"
            else:
                css_class = "stage-pending"
                icon = "⏳"
            stages_html += f"<div class='{css_class}'>{icon} {name}</div>"
        st.markdown(stages_html, unsafe_allow_html=True)

def cleanup_finished_analysis():
    """清理已完成任务的线程引用"""
    worker = st.session_state.analysis_thread
    if worker is not None and not worker.is_alive() and not st.session_state.analysis_running:
        st.session_state.analysis_thread = None
        st.session_state.analysis_queue = None
        st.session_state.analysis_cancel_event = None

def process_analysis_events():
    """处理后台分析线程发回的事件"""
    event_queue = st.session_state.analysis_queue
    if event_queue is None:
        cleanup_finished_analysis()
        return

    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break

        event_type = event.get("type")
        if event_type == "log":
            st.session_state.analysis_logs.append({
                "time": event["time"],
                "message": event["message"],
                "level": event["level"],
            })
        elif event_type == "stage":
            st.session_state.analysis_stage_name = event["stage_name"]
            st.session_state.analysis_stage_status = event["status"]
            st.session_state.analysis_progress = event["progress"]
        elif event_type == "result":
            st.session_state.analysis_results = event["results"]
            st.session_state.analysis_running = False
            st.session_state.analysis_stop_requested = False
            st.session_state.analysis_error = None
            st.session_state.analysis_outcome = "completed"
            st.session_state.analysis_stage_name = "完成"
            st.session_state.analysis_stage_status = "complete"
            st.session_state.analysis_progress = len(STAGE_NAMES)
            st.session_state.analysis_full_refresh_requested = True
        elif event_type == "cancelled":
            st.session_state.analysis_running = False
            st.session_state.analysis_stop_requested = False
            st.session_state.analysis_outcome = "cancelled"
            st.session_state.analysis_results = None
            st.session_state.analysis_full_refresh_requested = True
        elif event_type == "error":
            st.session_state.analysis_running = False
            st.session_state.analysis_stop_requested = False
            st.session_state.analysis_outcome = "failed"
            st.session_state.analysis_error = event["message"]
            st.session_state.analysis_full_refresh_requested = True

    worker = st.session_state.analysis_thread
    if worker is not None and not worker.is_alive() and st.session_state.analysis_running:
        st.session_state.analysis_running = False
        if st.session_state.analysis_outcome == "idle":
            st.session_state.analysis_outcome = "failed"
            append_ui_log("分析线程已结束，但没有返回结果。", "warning")
            st.session_state.analysis_full_refresh_requested = True

    cleanup_finished_analysis()

def start_analysis_task(
    source_counts,
    use_cache,
    use_llm_strategy,
    data_path=None,
    sample_size=None,
    resume_mode=None,
    resume_path=None,
    source_config=None,
):
    """启动后台分析任务"""
    st.session_state.analysis_logs = []
    st.session_state.analysis_progress = 0
    st.session_state.analysis_stage_name = ""
    st.session_state.analysis_stage_status = "pending"
    st.session_state.analysis_results = None
    st.session_state.analysis_error = None
    st.session_state.analysis_running = True
    st.session_state.analysis_stop_requested = False
    st.session_state.analysis_outcome = "running"
    st.session_state.analysis_full_refresh_requested = False
    if resume_mode == "events":
        st.session_state.analysis_run_description = f"从已抽取事件继续分析：{resume_path}"
    elif resume_mode == "report":
        st.session_state.analysis_run_description = f"从历史结果重新生成报告：{resume_path}"
    elif source_config and source_config.get("backend") == "db":
        query = source_config["query"]
        q_name = getattr(query, "tech_field_name", None)
        if not q_name and isinstance(query, dict):
            q_name = query.get("tech_field_name")
        q_name = q_name or "db"
        st.session_state.analysis_run_description = f"数据库检索流程：{q_name}"
    elif data_path:
        st.session_state.analysis_run_description = f"完整流程：{data_path}"
    else:
        total_samples = int(sample_size or sum((source_counts or {}).values()))
        st.session_state.analysis_run_description = f"完整流程：{total_samples} 条数据"

    event_queue = queue.Queue()
    cancel_event = threading.Event()
    worker = threading.Thread(
        target=run_analysis,
        args=(source_counts, use_cache, use_llm_strategy, event_queue, cancel_event, data_path, sample_size, resume_mode, resume_path, source_config),
        daemon=True,
        name="analysis-worker",
    )

    st.session_state.analysis_queue = event_queue
    st.session_state.analysis_cancel_event = cancel_event
    st.session_state.analysis_thread = worker
    worker.start()

def request_stop_analysis():
    """请求停止当前分析任务"""
    if not st.session_state.analysis_running or st.session_state.analysis_stop_requested:
        return

    st.session_state.analysis_stop_requested = True
    st.session_state.analysis_outcome = "stopping"
    cancel_event = st.session_state.analysis_cancel_event
    if cancel_event is not None:
        cancel_event.set()
    append_ui_log("已发送停止请求，将在当前阶段完成后中止分析。", "warning")

def render_analysis_runtime_panel():
    """渲染分析运行状态面板。"""
    process_analysis_events()

    if st.session_state.analysis_run_description:
        st.caption(st.session_state.analysis_run_description)

    action_col, hint_col = st.columns([1, 3])
    with action_col:
        stop_button = st.button(
            "⏹ 停止分析",
            use_container_width=True,
            disabled=not st.session_state.analysis_running,
            key="analysis_stop_button",
        )
    with hint_col:
        if st.session_state.analysis_running:
            st.caption("状态区正在局部刷新，页面主体不会整页闪烁。")
        else:
            st.caption("分析未运行时，状态区不会自动刷新。")

    if stop_button:
        request_stop_analysis()

    progress_value = min(st.session_state.analysis_progress / len(STAGE_NAMES), 1.0)
    st.progress(progress_value)
    render_stage_status(st.container(), st.session_state.analysis_stage_name, st.session_state.analysis_stage_status, st.session_state.analysis_progress)
    render_logs(st.container(), st.session_state.analysis_logs)

    if st.session_state.analysis_stop_requested:
        st.warning("已发送停止请求，系统会在当前阶段结束后尽快停止。")
    elif st.session_state.analysis_running:
        st.info("分析正在后台执行，状态区会自动刷新。")
    elif st.session_state.analysis_outcome == "cancelled":
        st.warning("当前分析已停止。")
    elif st.session_state.analysis_outcome == "failed" and st.session_state.analysis_error:
        st.error(f"分析失败：{st.session_state.analysis_error}")

    if st.session_state.analysis_full_refresh_requested:
        st.session_state.analysis_full_refresh_requested = False
        st.rerun()

def render_live_analysis_panel():
    """渲染带自动刷新的运行态面板。"""
    analysis_panel = get_fragment_decorator(
        run_every="1s" if st.session_state.analysis_running else None
    )(render_analysis_runtime_panel)
    analysis_panel()

def get_available_data_sources():
    """获取可用的数据源文件"""
    data_files = list(Config.DATA_DIR.glob("*.csv")) + \
                 list(Config.DATA_DIR.glob("*.xlsx")) + \
                 list(Config.DATA_DIR.glob("*.json"))

    sources = {
        '专利': {'files': [], 'count': 0},
        '文献': {'files': [], 'count': 0},
        '研报': {'files': [], 'count': 0},
        '资讯': {'files': [], 'count': 0}
    }

    for f in data_files:
        fname = f.name.lower()
        try:
            if f.suffix == '.csv':
                df = pd.read_csv(f)
            elif f.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(f)
            else:
                df = pd.read_json(f)
            count = len(df)
        except:
            count = 0

        if '专利' in fname or 'patent' in fname:
            sources['专利']['files'].append(f.name)
            sources['专利']['count'] += count
        elif '文献' in fname or 'literature' in fname or 'paper' in fname:
            sources['文献']['files'].append(f.name)
            sources['文献']['count'] += count
        elif '研报' in fname or 'report' in fname:
            sources['研报']['files'].append(f.name)
            sources['研报']['count'] += count
        elif '资讯' in fname or 'news' in fname:
            sources['资讯']['files'].append(f.name)
            sources['资讯']['count'] += count

    return sources

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🔧 分析流程")

        stages = [
            ("📁 数据加载", "从多种数据源加载技术文本"),
            ("📝 事件抽取与质量评分", "抽取结构化事件并同步评估证据质量"),
            ("🔧 候选成形", "生成候选技术对象"),
            ("📊 弱信号评分", "计算多维评分指标"),
            ("🎯 主题细化", "LLM小主题判断与收口"),
            ("✅ 反向验证", "验证候选有效性"),
            ("⏱ 时间验证", "验证候选证据的跨时间增长与延续性"),
            ("🚀 信号生成", "生成弱信号列表"),
            ("📋 报告生成", "LLM生成分析报告")
        ]

        for name, desc in stages:
            with st.expander(name):
                st.markdown(f"*{desc}*")

        st.markdown("---")

        st.markdown("## 🤖 LLM状态")
        provider, client = get_provider_and_client()
        if client:
            st.success(f"✅ {provider} API 已连接")

            extraction_model = os.getenv("EXTRACTION_MODEL", "未配置")
            report_model = os.getenv("REPORT_MODEL", "未配置")
            agent_model = os.getenv("AGENT_MODEL", "未配置")

            st.markdown(f"**事件抽取**: `{extraction_model}`")
            st.markdown(f"**报告生成**: `{report_model}`")
            st.markdown(f"**智能体**: `{agent_model}`")
        else:
            st.error("❌ LLM未连接")

        st.markdown("---")

        st.markdown("## 📚 历史结果")
        result_dirs = sorted(Config.RESULT_DIR.iterdir(), key=lambda x: x.name, reverse=True)[:5]
        for rd in result_dirs:
            if rd.is_dir():
                file_count = len(list(rd.glob("*")))
                st.markdown(f"📁 `{rd.name}` ({file_count}文件)")

def render_header():
    """渲染页面头部"""
    st.markdown('<div class="main-header">🔍 产业技术弱信号识别系统</div>', unsafe_allow_html=True)
    ##st.markdown('<div class="sub-header">基于大模型驱动的产业技术弱信号识别系统</div>', unsafe_allow_html=True)

def render_data_source_selection(sources):
    """渲染数据源选择"""
    st.markdown("## 📁 数据源配置")

    cols = st.columns(4)
    source_counts = {}

    for i, (source_name, info) in enumerate(sources.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px; padding: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 10px;">
                <div style="font-size: 2rem; font-weight: bold;">{info['count']}</div>
                <div style="font-size: 0.9rem;">{source_name}数据</div>
            </div>
            """, unsafe_allow_html=True)

            count = st.number_input(
                f"采样数量",
                min_value=0,
                max_value=info['count'],
                value=min(25, info['count']),
                step=5,
                key=f"count_{source_name}"
            )
            source_counts[source_name] = count

    total = sum(source_counts.values())
    st.info(f"📊 总计选择 **{total}** 条数据")

    return source_counts, total

def split_csv_terms(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]

def available_domain_pack_options():
    options = []
    seen_hashes = set()
    pack_dirs = [
        Path(__file__).parent / "src" / "config" / "domain_packs",
        Config.MEMORY_DIR / "domain_packs",
    ]
    for pack_dir in pack_dirs:
        if not pack_dir.exists():
            continue
        for path in sorted(pack_dir.glob("*")):
            if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            try:
                pack = load_domain_pack(path)
            except Exception:
                continue
            if pack.pack_id == "neutral" or pack.domain_pack_hash in seen_hashes:
                continue
            if not is_current_generated_domain_pack(pack):
                continue
            seen_hashes.add(pack.domain_pack_hash)
            options.append({
                "label": f"{pack.pack_name} · {pack.pack_id} · {pack.domain_pack_version} · {pack.domain_pack_hash}",
                "ref": str(path),
                "pack": pack,
            })
    return options

def set_current_domain_pack(pack):
    st.session_state.domain_pack_current = pack
    st.session_state.domain_pack_validation_report = validate_domain_pack(
        pack,
        require_dry_run=False,
    )
    st.session_state.domain_pack_analysis_confirmed = False
    st.session_state.domain_pack_confirmed_counts = {}

def render_domain_pack_validation(report):
    if report is None:
        return
    if report.is_valid:
        st.success(f"Domain Pack 基础校验通过：{report.gate_status}")
    else:
        st.error(f"Domain Pack 基础校验未通过：{report.gate_status}")
    if report.errors:
        st.markdown("**错误**")
        for item in report.errors:
            st.error(item)
    if report.warnings:
        st.markdown("**提醒**")
        for item in report.warnings:
            st.warning(item)
    if report.checks:
        st.dataframe(
            pd.DataFrame(
                [{"检查项": key, "状态": value} for key, value in report.checks.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

def render_domain_pack_review_panel(available_counts):
    pack = st.session_state.get("domain_pack_current")
    report = st.session_state.get("domain_pack_validation_report")
    if pack is None:
        st.info("请先生成 Domain Pack，或复用已有 Domain Pack。")
        return {}, False

    st.markdown("#### 当前领域运行包")
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Pack ID", pack.pack_id)
    with meta_cols[1]:
        st.metric("版本", pack.domain_pack_version)
    with meta_cols[2]:
        st.metric("Hash", pack.domain_pack_hash)
    with meta_cols[3]:
        st.metric("模式", pack.source.get("mode", ""))

    with st.expander("运行包核心规则预览", expanded=False):
        strategy = pack.search_strategy if isinstance(pack.search_strategy, dict) else {}
        formation = pack.candidate_formation if isinstance(pack.candidate_formation, dict) else {}
        preview_rows = [
            {"项目": "核心关键词", "内容": ", ".join(strategy.get("core_keywords", [])[:12])},
            {"项目": "排除词", "内容": ", ".join(strategy.get("exclude_terms", [])[:12])},
            {"项目": "候选对象类型", "内容": ", ".join(formation.get("technical_object_types", [])[:12])},
            {"项目": "泛化壳词", "内容": ", ".join(formation.get("shell_terms", [])[:12])},
        ]
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    render_domain_pack_validation(report)

    st.markdown("#### 数据源数量推荐")
    recommendation = recommend_source_counts(
        pack,
        available_counts=available_counts,
        total_sample_size=DB_DEFAULT_SAMPLE_TOTAL,
        source_types=DB_SOURCE_TYPES,
    )
    explicit_counts = {}
    count_cols = st.columns(len(DB_SOURCE_TYPES))
    rows_by_source = recommendation.rows_by_source
    for index, source_type in enumerate(DB_SOURCE_TYPES):
        row = rows_by_source.get(source_type)
        default_count = row.recommended_count if row else 0
        with count_cols[index]:
            explicit_counts[source_type] = int(st.number_input(
                f"{DB_SOURCE_LABELS[source_type]}",
                min_value=0,
                max_value=1000,
                value=int(default_count),
                step=5,
                key=f"db_count_{source_type}_{pack.domain_pack_hash}",
            ))

    final_recommendation = recommend_source_counts(
        pack,
        available_counts=available_counts,
        total_sample_size=DB_DEFAULT_SAMPLE_TOTAL,
        explicit_counts=explicit_counts,
        source_types=DB_SOURCE_TYPES,
    )
    review_state = build_domain_pack_review_state(
        pack,
        available_counts=available_counts,
        total_sample_size=DB_DEFAULT_SAMPLE_TOTAL,
        explicit_counts=explicit_counts,
        source_types=DB_SOURCE_TYPES,
        confirmed=bool(st.session_state.get("domain_pack_analysis_confirmed", False)),
    )
    final_recommendation = review_state.recommendation
    st.session_state.domain_pack_confirmed_counts = review_state.final_counts
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "来源": DB_SOURCE_LABELS.get(row.source_type, row.source_type),
                    "权重": row.weight,
                    "可用": row.available_count,
                    "推荐": row.recommended_count,
                    "最终": row.final_count,
                    "原因": row.limit_reason,
                }
                for row in final_recommendation.rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    confirmed = st.checkbox(
        "使用当前运行包开始分析",
        key="domain_pack_analysis_confirmed",
        disabled=not review_state.validation_report.is_valid,
    )
    return review_state.final_counts, bool(confirmed and review_state.validation_report.is_valid and review_state.total_count > 0)

@st.dialog("预览 Domain Pack 内容", width="large")
def show_domain_pack_preview(pack_dict):
    st.json(pack_dict)

def render_domain_pack_controls(field_id, field_name, keywords, synonyms, exclude_terms):
    st.markdown("#### 领域运行包")
    preset_ref = DOMAIN_SEARCH_TEMPLATES.get(
        st.session_state.get("selected_domain_template", "自定义领域"),
        {},
    ).get("preset_pack_ref", "")

    action_cols = st.columns([1, 1, 1])
    with action_cols[0]:
        refresh_cache = st.checkbox("重新生成运行包", value=False)
        if st.button("生成 Domain Pack", use_container_width=True):
            if not field_id or not field_name or not keywords:
                st.error("请先填写技术领域 ID、名称和至少一个检索词。")
            else:
                status_placeholder = st.empty()
                status_placeholder.info("⏳ 准备生成 Domain Pack...")
                try:
                    request = DomainPackGenerationRequest(
                        field_id=field_id,
                        field_name=field_name,
                        keywords=keywords,
                        synonyms=synonyms,
                        exclude_terms=exclude_terms,
                        source_types=DB_SOURCE_TYPES,
                    )

                    def update_progress(stage_name, stage_idx):
                        stage_labels = {
                            "domain_modeling": "阶段 1/4: 领域建模 (Domain Modeling)...",
                            "candidate_formation": "阶段 2/4: 候选成形范式 (Candidate Formation)...",
                            "weak_signal_rules": "阶段 3/4: 弱信号识别规则 (Weak Signal Rules)...",
                            "integration": "阶段 4/4: 整合与校验 (Integration & Check)...",
                        }
                        label = stage_labels.get(stage_name, f"阶段 {stage_idx}/4")
                        status_placeholder.info(f"⏳ {label}")

                    pack = DomainPackGenerator(memory_dir=Config.MEMORY_DIR / "domain_packs").generate(
                        request,
                        refresh_cache=refresh_cache,
                        progress_fn=update_progress,
                    )
                    set_current_domain_pack(pack)
                    status_placeholder.empty()
                    st.success("Domain Pack 已生成并完成基础校验。")
                except Exception as exc:
                    status_placeholder.empty()
                    st.error(f"Domain Pack 生成失败: {exc}")

    with action_cols[1]:
        if preset_ref:
            if st.button("使用 preset Domain Pack", use_container_width=True):
                try:
                    set_current_domain_pack(load_domain_pack(preset_ref))
                    st.success("已载入 preset Domain Pack。")
                except Exception as exc:
                    st.error(f"载入 preset Domain Pack 失败: {exc}")
        else:
            st.caption("当前模板仅填充表单，需要生成或复用运行包。")

    with action_cols[2]:
        options = available_domain_pack_options()
        if options:
            selected_label = st.selectbox(
                "复用已有 Domain Pack",
                options=[item["label"] for item in options],
            )
            col_load, col_preview, col_del = st.columns(3)
            with col_load:
                if st.button("载入运行包", use_container_width=True):
                    selected = next(item for item in options if item["label"] == selected_label)
                    set_current_domain_pack(load_domain_pack(selected["ref"]))
                    st.success("已载入已有 Domain Pack。")
            with col_preview:
                if st.button("预览运行包", use_container_width=True):
                    selected = next(item for item in options if item["label"] == selected_label)
                    show_domain_pack_preview(selected["pack"].to_dict())
            with col_del:
                if st.button("删除运行包", use_container_width=True):
                    selected = next(item for item in options if item["label"] == selected_label)
                    try:
                        os.remove(selected["ref"])
                        st.success("运行包已删除！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
        else:
            st.caption("暂无可复用运行包。")


def run_analysis(
    source_counts,
    use_cache,
    use_llm_strategy,
    event_queue,
    cancel_event,
    data_path=None,
    sample_size=None,
    resume_mode=None,
    resume_path=None,
    source_config=None,
):
    """运行分析流程"""

    def emit_event(event_type, **payload):
        event_queue.put({"type": event_type, **payload})

    def add_log(message, level="info"):
        emit_event(
            "log",
            time=datetime.now().strftime("%H:%M:%S"),
            message=message,
            level=level,
        )

    def update_stage(stage_name, status, progress):
        emit_event(
            "stage",
            stage_name=stage_name,
            status=status,
            progress=progress,
        )

    def check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("analysis_cancelled")

    if source_config and source_config.get("backend") == "db":
        query = source_config["query"]
        q_counts = getattr(query, "counts", None)
        if q_counts is None and isinstance(query, dict):
            q_counts = query.get("counts", {})
        q_counts = q_counts or {}
        total_samples = sum(q_counts.values())
    else:
        source_counts = source_counts or {}
        source_config = {
            'counts': {
                'patent': source_counts.get('专利', 0),
                'literature': source_counts.get('文献', 0),
                'report': source_counts.get('研报', 0),
                'news': source_counts.get('资讯', 0)
            }
        }
        total_samples = int(sample_size or sum(source_counts.values()))

    if total_samples == 0 and not data_path and not resume_mode:
        add_log("请至少选择一个数据源", "error")
        emit_event("error", message="未选择任何数据源")
        return None

    domain_context = None
    if isinstance(source_config, dict):
        domain_context = source_config.get("domain_context")

    if resume_mode == "events":
        add_log("开始从已抽取事件继续分析", "info")
    elif resume_mode == "report":
        add_log("开始从历史结果重新生成报告", "info")
    else:
        add_log(f"开始分析，共 {total_samples} 条数据", "info")

    pipeline = AnalysisPipeline(domain_context=domain_context)
    add_log(
        "Domain Pack: "
        f"{pipeline.domain_context.domain_pack_id} / "
        f"{pipeline.domain_context.domain_pack_version} / "
        f"{pipeline.domain_context.domain_pack_hash}",
        "info",
    )

    try:
        if resume_mode == "events":
            add_log(f"从已抽取事件继续分析: {resume_path}", "info")
            check_cancel()

            events_df = pipeline._read_dataframe_file(Path(resume_path))
            if events_df.empty:
                add_log("已抽取事件为空，无法继续分析", "error")
                emit_event("error", message="已抽取事件为空，无法继续分析")
                return None

            raw_data = pipeline._raw_data_from_events(events_df)
            source_result_dir = Path(resume_path).parent if resume_path else None
            pipeline._resolve_domain_context(
                source_result_dir=source_result_dir,
                events_df=events_df,
                raw_data=raw_data,
            )
            add_log(f"读取了 {len(events_df)} 个事件，已跳过数据加载和事件抽取", "success")

            update_stage("事件抽取与质量评分", "active", 1)
            add_log("正在回挂事件质量评分...", "info")
            check_cancel()
            events_df = pipeline._score_events_during_extraction(events_df, raw_data)
            event_quality_df = pipeline.latest_event_quality_df.copy()
            raw_data = merge_event_quality_into_raw_data(raw_data, event_quality_df)
            check_cancel()
            add_log(f"事件结构化与质量评分完成，共 {len(event_quality_df)} 条", "success")

            update_stage("候选成形", "active", 2)
            add_log("正在进行候选成形...", "info")
            check_cancel()
            candidate_forms_df = pipeline._form_candidates(events_df, raw_data)
            candidate_forms_df = pipeline._apply_candidate_event_quality(candidate_forms_df, event_quality_df)
            candidate_count_before_dedupe = len(candidate_forms_df)
            candidate_forms_df = pipeline._dedupe_candidate_flow(candidate_forms_df)
            check_cancel()
            add_log(f"生成了 {candidate_count_before_dedupe} 个候选对象，流转去重后 {len(candidate_forms_df)} 个", "success")

            update_stage("弱信号评分", "active", 3)
            add_log("正在进行弱信号评分...", "info")
            check_cancel()
            scored_df = pipeline._score_candidates(candidate_forms_df)
            check_cancel()
            add_log(f"评分完成，共 {len(scored_df)} 个候选", "success")

            update_stage("主题细化", "active", 4)
            add_log("正在进行主题细化...", "info")
            check_cancel()
            refined_df = pipeline._refine_topics(scored_df)
            check_cancel()
            add_log("主题细化完成", "success")

            update_stage("反向验证", "active", 5)
            add_log("正在进行反向验证...", "info")
            check_cancel()
            validated_df = pipeline._validate_reverse(refined_df, candidate_forms_df)
            check_cancel()
            add_log("反向验证完成", "success")

            update_stage("时间验证", "active", 6)
            add_log("正在进行时间验证...", "info")
            check_cancel()
            temporal_validation_df = pipeline._validate_temporal(validated_df, events_df, raw_data)
            validated_df = merge_temporal_validation_into_candidates(validated_df, temporal_validation_df)
            check_cancel()
            add_log(f"时间验证完成，共 {len(temporal_validation_df)} 条", "success")

            update_stage("信号生成", "active", 7)
            add_log("正在生成信号...", "info")
            check_cancel()
            signals_output = pipeline._generate_signals(validated_df, raw_data)
            check_cancel()

            if isinstance(signals_output, dict):
                signals_df = signals_output.get("candidates_df", pd.DataFrame())
                near_strong_df = signals_output.get("near_strong_candidates_df", pd.DataFrame())
            else:
                signals_df = signals_output
                near_strong_df = pd.DataFrame()
            add_log(f"生成了 {len(signals_df)} 个信号", "success")

            update_stage("报告生成", "active", 8)
            add_log("正在生成报告...", "info")
            check_cancel()
            report = pipeline._generate_report(signals_output)
            check_cancel()
            add_log("报告生成完成", "success")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = Config.RESULT_DIR / f"{timestamp}_from_events"
            result_dir.mkdir(parents=True, exist_ok=True)

            pipeline._save_results(result_dir, events_df, candidate_forms_df, scored_df, refined_df, validated_df, signals_output, report, raw_data=raw_data)
            signals_df = pipeline._assign_signal_ids(signals_df)

            results = {
                "result_dir": str(result_dir),
                "domain_context": pipeline.domain_context,
                "events_df": events_df,
                "event_quality_df": pipeline.latest_event_quality_df,
                "candidate_forms_df": candidate_forms_df,
                "scored_df": scored_df,
                "refined_df": refined_df,
                "reverse_validation_df": pipeline.latest_reverse_validation_df,
                "temporal_validation_df": pipeline.latest_temporal_validation_df,
                "validated_df": validated_df,
                "signals_df": signals_df,
                "signals_output": signals_output,
                "near_strong_df": near_strong_df,
                "family_metrics_df": pipeline.latest_family_metrics_df,
                "source_documents_df": pipeline.latest_source_documents_df,
                "signal_evidence_links_df": pipeline.latest_signal_evidence_links_df,
                "signal_reliability_df": pipeline.latest_signal_reliability_df,
                "weak_signals_comparison_df": pipeline.latest_weak_signals_comparison_df,
                "report": report,
                "raw_data": raw_data,
            }
            check_cancel()
            update_stage("完成", "complete", len(STAGE_NAMES))
            add_log(f"结果已保存到: {result_dir}", "success")
            emit_event("result", results=results)
            return results

        if resume_mode == "report":
            update_stage("报告生成", "active", 8)
            add_log(f"从历史结果重新生成报告: {resume_path}", "info")
            check_cancel()

            results = pipeline.regenerate_report_from_result(Path(resume_path))
            check_cancel()
            if not results:
                add_log("报告重新生成失败", "error")
                emit_event("error", message="报告重新生成失败")
                return None

            update_stage("完成", "complete", len(STAGE_NAMES))
            add_log(f"新报告已保存到: {results.get('result_dir')}", "success")
            emit_event("result", results=results)
            return results

        update_stage("数据加载", "active", 0)
        add_log("正在加载数据...", "info")
        check_cancel()

        raw_data = pipeline._load_data(Path(data_path) if data_path else None, total_samples, source_config)
        check_cancel()

        if raw_data.empty:
            add_log("数据加载失败", "error")
            emit_event("error", message="数据加载失败")
            return None

        add_log(f"加载了 {len(raw_data)} 条数据", "success")
        update_stage("事件抽取与质量评分", "active", 1)
        add_log("正在进行事件抽取与质量评分...", "info")
        check_cancel()

        cache_dir = Config.MEMORY_DIR / "cache"
        events_df = pipeline._extract_events(
            raw_data,
            use_cache,
            cache_dir,
            data_path=Path(data_path) if data_path else None,
            source_config=source_config,
        )
        check_cancel()

        if events_df is None or events_df.empty:
            add_log("事件抽取失败", "error")
            emit_event("error", message="事件抽取失败")
            return None

        event_quality_df = pipeline.latest_event_quality_df.copy()
        raw_data = merge_event_quality_into_raw_data(raw_data, event_quality_df)
        check_cancel()
        add_log(f"抽取了 {len(events_df)} 个事件，并完成 {len(event_quality_df)} 条事件质量评分", "success")

        update_stage("候选成形", "active", 2)
        add_log("正在进行候选成形...", "info")
        check_cancel()

        candidate_forms_df = pipeline._form_candidates(events_df, raw_data)
        candidate_forms_df = pipeline._apply_candidate_event_quality(candidate_forms_df, event_quality_df)
        candidate_count_before_dedupe = len(candidate_forms_df)
        candidate_forms_df = pipeline._dedupe_candidate_flow(candidate_forms_df)
        check_cancel()
        add_log(f"生成了 {candidate_count_before_dedupe} 个候选对象，流转去重后 {len(candidate_forms_df)} 个", "success")

        update_stage("弱信号评分", "active", 3)
        add_log("正在进行弱信号评分...", "info")
        check_cancel()

        scored_df = pipeline._score_candidates(candidate_forms_df)
        check_cancel()
        add_log(f"评分完成，共 {len(scored_df)} 个候选", "success")

        update_stage("主题细化", "active", 4)
        add_log("正在进行主题细化...", "info")
        check_cancel()

        refined_df = pipeline._refine_topics(scored_df)
        check_cancel()
        add_log("主题细化完成", "success")

        update_stage("反向验证", "active", 5)
        add_log("正在进行反向验证...", "info")
        check_cancel()

        validated_df = pipeline._validate_reverse(refined_df, candidate_forms_df)
        check_cancel()
        add_log("反向验证完成", "success")

        update_stage("时间验证", "active", 6)
        add_log("正在进行时间验证...", "info")
        check_cancel()
        temporal_validation_df = pipeline._validate_temporal(validated_df, events_df, raw_data)
        validated_df = merge_temporal_validation_into_candidates(validated_df, temporal_validation_df)
        check_cancel()
        add_log(f"时间验证完成，共 {len(temporal_validation_df)} 条", "success")

        update_stage("信号生成", "active", 7)
        add_log("正在生成信号...", "info")
        check_cancel()

        signals_output = pipeline._generate_signals(validated_df, raw_data)
        check_cancel()

        if isinstance(signals_output, dict):
            signals_df = signals_output.get('candidates_df', pd.DataFrame())
            near_strong_df = signals_output.get('near_strong_candidates_df', pd.DataFrame())
        else:
            signals_df = signals_output
            near_strong_df = pd.DataFrame()

        add_log(f"生成了 {len(signals_df)} 个信号", "success")

        update_stage("报告生成", "active", 8)
        add_log("正在生成报告...", "info")
        check_cancel()

        report = pipeline._generate_report(signals_output)
        check_cancel()
        add_log("报告生成完成", "success")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Config.RESULT_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        pipeline._save_results(result_dir, events_df, candidate_forms_df, scored_df, refined_df, validated_df, signals_output, report, raw_data=raw_data)
        signals_df = pipeline._assign_signal_ids(signals_df)
        add_log(f"结果已保存到: {result_dir}", "success")

        update_stage("完成", "complete", len(STAGE_NAMES))
        add_log("🎉 分析流程全部完成！", "success")

        results = {
            'domain_context': pipeline.domain_context,
            'events_df': events_df,
            'event_quality_df': pipeline.latest_event_quality_df,
            'candidate_forms_df': candidate_forms_df,
            'scored_df': scored_df,
            'refined_df': refined_df,
            'reverse_validation_df': pipeline.latest_reverse_validation_df,
            'temporal_validation_df': pipeline.latest_temporal_validation_df,
            'validated_df': validated_df,
            'signals_df': signals_df,
            'signals_output': signals_output,
            'near_strong_df': near_strong_df,
            'family_metrics_df': pipeline.latest_family_metrics_df,
            'source_documents_df': pipeline.latest_source_documents_df,
            'signal_evidence_links_df': pipeline.latest_signal_evidence_links_df,
            'signal_reliability_df': pipeline.latest_signal_reliability_df,
            'weak_signals_comparison_df': pipeline.latest_weak_signals_comparison_df,
            'report': report,
            'result_dir': str(result_dir),
            'raw_data': raw_data
        }
        emit_event("result", results=results)
        return results

    except InterruptedError:
        add_log("分析已停止。", "warning")
        emit_event("cancelled")
        return None

    except Exception as e:
        add_log(f"分析失败: {e}", "error")
        add_log(traceback.format_exc(), "error")
        emit_event("error", message=str(e))
        return None

def render_results(results):
    """渲染结果"""
    if not results:
        return

    def sort_by_available_scores(df, preferred_columns):
        available_columns = [column for column in preferred_columns if column in df.columns]
        if not available_columns:
            return df
        sortable_df = df.copy()
        for column in available_columns:
            sortable_df[column] = pd.to_numeric(sortable_df[column], errors='coerce')
        return sortable_df.sort_values(
            by=available_columns,
            ascending=[False] * len(available_columns),
            na_position='last'
        )

    def build_unique_candidate_view(df):
        if df is None or df.empty:
            return df
        dedupe_columns = ['candidate_cluster_id', 'display_candidate_name', 'tech_name']
        dedupe_column = next((column for column in dedupe_columns if column in df.columns), None)
        if dedupe_column is None:
            return df
        sorted_df = sort_by_available_scores(
            df,
            ['weak_signal_score', 'hotspot_score', 'source_count', 'total_mentions', 'cluster_evidence_count']
        )
        dedupe_values = sorted_df[dedupe_column].fillna("").astype(str).str.strip()
        non_empty_mask = ~dedupe_values.str.lower().isin({"", "nan", "none", "null", "nat"})
        deduped_non_empty = sorted_df[non_empty_mask].drop_duplicates(subset=[dedupe_column], keep='first')
        empty_key_rows = sorted_df[~non_empty_mask]
        if empty_key_rows.empty:
            return deduped_non_empty
        return pd.concat([deduped_non_empty, empty_key_rows], ignore_index=True)

    def build_unique_signal_view(df):
        if df is None or df.empty:
            return df
        sorted_df = sort_by_available_scores(
            df,
            ['weak_signal_score', 'hotspot_score', 'cluster_evidence_count', 'source_count', 'total_mentions']
        )
        visible_identity = [
            column
            for column in ['display_candidate_name', 'mechanism_core', 'signal_bucket']
            if column in sorted_df.columns
        ]
        if visible_identity:
            return sorted_df.drop_duplicates(subset=visible_identity, keep='first')
        return build_unique_candidate_view(sorted_df)

    def numeric_value(value, default=0.0):
        try:
            converted = pd.to_numeric(value, errors='coerce')
            if pd.isna(converted):
                return default
            return float(converted)
        except Exception:
            return default

    def get_display_score(row):
        weak_score = numeric_value(row.get('weak_signal_score'), default=None)
        if weak_score is not None and weak_score > 0:
            return weak_score
        hotspot_score = numeric_value(row.get('hotspot_score'), default=None)
        if hotspot_score is not None and hotspot_score > 0:
            return hotspot_score
        return numeric_value(row.get('score'), default=0.0)

    def safe_text(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none", "null", "nat"} else text

    def safe_preview(value, limit=100):
        text = safe_text(value)
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def safe_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def table_key_suffix(value):
        text = safe_text(value) or "active"
        return re.sub(r"[^0-9A-Za-z_\-]+", "_", text)[-48:] or "active"

    def render_paginated_dataframe(df, columns, key_prefix, page_size=20, empty_message="暂无可展示数据。"):
        if df is None or df.empty:
            st.info(empty_message)
            return pd.DataFrame()

        view_df = df[columns].copy() if columns else df.copy()
        total_rows = len(view_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        result_suffix = table_key_suffix(results.get('result_dir', 'active'))
        page_key = f"{key_prefix}_page_{result_suffix}"
        if page_key in st.session_state and int(st.session_state[page_key]) > total_pages:
            st.session_state[page_key] = total_pages

        page_cols = st.columns([1, 1, 2])
        with page_cols[0]:
            st.metric("记录数", total_rows)
        with page_cols[1]:
            st.metric("每页展示", page_size)
        with page_cols[2]:
            page_number = int(st.number_input(
                "页码",
                min_value=1,
                max_value=total_pages,
                value=min(int(st.session_state.get(page_key, 1)), total_pages),
                step=1,
                key=page_key,
            ))

        start = (page_number - 1) * page_size
        end = min(start + page_size, total_rows)
        st.caption(f"第 {page_number}/{total_pages} 页，展示第 {start + 1}-{end} 条")
        page_df = view_df.iloc[start:end].copy()
        st.dataframe(page_df, use_container_width=True)
        return page_df

    def render_horizontal_bar_chart(series, category_label="类别", value_label="数量", max_rows=30):
        if series is None:
            return
        if isinstance(series, pd.Series):
            chart_df = series.reset_index()
            chart_df.columns = [category_label, value_label]
        else:
            chart_df = pd.DataFrame(series)
            if chart_df.empty or len(chart_df.columns) < 2:
                return
            chart_df = chart_df.iloc[:, :2]
            chart_df.columns = [category_label, value_label]
        if chart_df.empty:
            return

        chart_df[category_label] = chart_df[category_label].astype(str)
        chart_df[value_label] = pd.to_numeric(chart_df[value_label], errors="coerce").fillna(0)
        chart_df = chart_df[chart_df[category_label].str.strip() != ""]
        if chart_df.empty:
            return
        if chart_df[category_label].duplicated().any():
            aggregate = "sum" if value_label in {"数量", "记录数", "命中记录"} else "max"
            chart_df = chart_df.groupby(category_label, as_index=False, sort=False)[value_label].agg(aggregate)

        chart_df = chart_df.sort_values(by=value_label, ascending=False, kind="stable").head(max_rows)
        category_order = chart_df[category_label].tolist()
        height = max(120, min(560, len(chart_df) * 34))
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y(
                    f"{category_label}:N",
                    sort=category_order,
                    axis=alt.Axis(title=None, labelAngle=0, labelLimit=260),
                ),
                x=alt.X(
                    f"{value_label}:Q",
                    axis=alt.Axis(title=None, tickMinStep=1),
                    scale=alt.Scale(domainMin=0, nice=True),
                ),
                tooltip=[
                    alt.Tooltip(f"{category_label}:N", title=category_label),
                    alt.Tooltip(f"{value_label}:Q", title=value_label),
                ],
            )
            .properties(height=height)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("## 📈 分析结果")
    domain_context = results.get('domain_context')
    if domain_context is not None:
        st.caption(
            "Domain Pack: "
            f"{domain_context.domain_pack_id} · "
            f"{domain_context.domain_pack_version} · "
            f"{domain_context.domain_pack_hash} · "
            f"{domain_context.runtime_mode}"
        )
    else:
        domain_meta = {}
        for frame_key in ["events_df", "signals_df", "candidate_forms_df"]:
            frame = results.get(frame_key)
            if isinstance(frame, pd.DataFrame) and not frame.empty and "domain_pack_hash" in frame.columns:
                row = frame.iloc[0]
                domain_meta = {
                    "domain_pack_id": row.get("domain_pack_id", ""),
                    "domain_pack_version": row.get("domain_pack_version", ""),
                    "domain_pack_hash": row.get("domain_pack_hash", ""),
                }
                break
        if domain_meta.get("domain_pack_hash"):
            st.caption(
                "Domain Pack: "
                f"{domain_meta.get('domain_pack_id') or 'unknown'} · "
                f"{domain_meta.get('domain_pack_version') or 'unknown'} · "
                f"{domain_meta.get('domain_pack_hash')}"
            )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📋 抽取事件", len(results['events_df']))
    with col2:
        st.metric("🔧 候选对象", len(build_unique_candidate_view(results['candidate_forms_df'])))
    with col3:
        st.metric("📊 评分候选", len(build_unique_candidate_view(results['scored_df'])))
    with col4:
        signals_df = results['signals_df']
        if 'signal_type' in signals_df.columns:
            weak_count = len(signals_df[signals_df['signal_type'] == 'weak_signal'])
        else:
            weak_count = len(signals_df)
        st.metric("🎯 弱信号", weak_count)
    with col5:
        temporal_validation_df = results.get('temporal_validation_df', pd.DataFrame())
        if isinstance(temporal_validation_df, pd.DataFrame) and 'monitoring_priority' in temporal_validation_df.columns:
            watch_count = int((temporal_validation_df['monitoring_priority'].astype(str) == 'high').sum())
        else:
            watch_count = 0
        st.metric("⏱ 高频观测", watch_count)
    with col6:
        st.metric("📝 报告", "已生成")

    tabs = st.tabs([
        "📊 概览",
        "📋 事件列表",
        "🧾 事件质量",
        "🔧 候选成形",
        "📊 弱信号评分",
        "🎯 弱信号生成",
        "⏱ 持续观测",
        "📝 分析报告",
    ])

    with tabs[0]:
        st.markdown("### 📊 数据源分布")
        raw_data = results['raw_data']
        if 'source_type' in raw_data.columns:
            source_dist = raw_data['source_type'].value_counts()
            render_horizontal_bar_chart(source_dist, "数据源", "数量")

        st.markdown("### 📈 信号类型分布")
        signals_df = results['signals_df']
        if 'signal_type' in signals_df.columns:
            signal_dist = signals_df['signal_type'].value_counts()
            render_horizontal_bar_chart(signal_dist, "信号类型", "数量")

        event_quality_df = results.get('event_quality_df', pd.DataFrame())
        if isinstance(event_quality_df, pd.DataFrame) and not event_quality_df.empty:
            st.markdown("### 证据质量概览")
            quality_cols = st.columns(3)
            with quality_cols[0]:
                quality_series = (
                    event_quality_df['event_quality_score']
                    if 'event_quality_score' in event_quality_df.columns
                    else pd.Series([0] * len(event_quality_df))
                )
                avg_quality = pd.to_numeric(quality_series, errors='coerce').fillna(0).mean()
                st.metric("事件平均质量", f"{avg_quality:.1f}")
            with quality_cols[1]:
                foresight_series = (
                    event_quality_df['foresight_relevance_score']
                    if 'foresight_relevance_score' in event_quality_df.columns
                    else pd.Series([0] * len(event_quality_df))
                )
                avg_foresight = pd.to_numeric(foresight_series, errors='coerce').fillna(0).mean()
                st.metric("平均预见相关度", f"{avg_foresight:.1f}")
            with quality_cols[2]:
                high_quality = int((pd.to_numeric(quality_series, errors='coerce').fillna(0) >= 7).sum())
                st.metric("高质量事件", high_quality)

        temporal_validation_df = results.get('temporal_validation_df', pd.DataFrame())
        has_temporal = isinstance(temporal_validation_df, pd.DataFrame) and not temporal_validation_df.empty
        if has_temporal:
            st.markdown("### 时间验证与持续观测")
            temporal_cols = st.columns(4)
            with temporal_cols[0]:
                if 'temporal_validation_passed' in temporal_validation_df.columns:
                    passed = temporal_validation_df['temporal_validation_passed'].astype(str).str.lower().isin(['true', '1', 'yes', '通过']).sum()
                else:
                    passed = 0
                st.metric("时间验证通过", int(passed))
            with temporal_cols[1]:
                if 'temporal_momentum_score' in temporal_validation_df.columns:
                    momentum = pd.to_numeric(temporal_validation_df['temporal_momentum_score'], errors='coerce').fillna(0).mean()
                else:
                    momentum = 0
                st.metric("平均动量分", f"{momentum:.1f}")
            with temporal_cols[2]:
                if 'monitoring_priority' in temporal_validation_df.columns:
                    high_watch = int((temporal_validation_df['monitoring_priority'].astype(str) == 'high').sum())
                else:
                    high_watch = 0
                st.metric("高频观测对象", high_watch)
            with temporal_cols[3]:
                if 'monitoring_priority' in temporal_validation_df.columns:
                    needs_dates = int((temporal_validation_df['monitoring_priority'].astype(str) == 'needs_dates').sum())
                else:
                    needs_dates = 0
                st.metric("待补日期", needs_dates)

        family_metrics_df = results.get('family_metrics_df', pd.DataFrame())
        if (
            isinstance(family_metrics_df, pd.DataFrame)
            and not family_metrics_df.empty
            and 'family_coverage' in family_metrics_df.columns
        ):
            covered_families = family_metrics_df[
                pd.to_numeric(family_metrics_df['family_coverage'], errors='coerce').fillna(0) > 0
            ].copy()
            if not covered_families.empty:
                st.markdown("### 对象族承接概览")

                priority_series = (
                    covered_families['priority'].astype(str).str.lower()
                    if 'priority' in covered_families.columns
                    else pd.Series(dtype='object')
                )
                cross_source_series = (
                    pd.to_numeric(covered_families['family_cross_source_count'], errors='coerce').fillna(0)
                    if 'family_cross_source_count' in covered_families.columns
                    else pd.Series(dtype='float64')
                )

                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.metric("覆盖对象族", int(len(covered_families)))
                with summary_cols[1]:
                    st.metric("高优先级覆盖", int((priority_series == 'high').sum()))
                with summary_cols[2]:
                    st.metric("跨源对象族", int((cross_source_series > 0).sum()))

                preview_cols = [
                    'canonical_term',
                    'priority',
                    'family_coverage',
                    'family_cross_source_count',
                    'family_semantic_consistency',
                ]
                preview_cols = [c for c in preview_cols if c in covered_families.columns]

                sort_cols = [c for c in ['family_cross_source_count', 'family_coverage'] if c in covered_families.columns]
                if sort_cols:
                    covered_families = covered_families.sort_values(
                        by=sort_cols,
                        ascending=[False] * len(sort_cols),
                        na_position='last',
                    )

                if preview_cols:
                    st.dataframe(covered_families[preview_cols].head(10), use_container_width=True)


    with tabs[1]:
        st.markdown("### 📋 抽取的事件要素")
        events_df = results['events_df']
        if not events_df.empty:
            display_cols = [
                'event_id',
                'title',
                'event_schema_version',
                'schema_migration_mode',
                'subject',
                'action',
                'technology',
                'technical_object',
                'mechanism',
                'task',
                'data_modality',
                'method',
                'evidence_span',
                'confidence',
                'source_extraction_mode',
                'scene',
                'time',
                'source_type',
            ]
            display_cols = [c for c in display_cols if c in events_df.columns]
            render_paginated_dataframe(
                events_df,
                display_cols,
                "events_table",
                page_size=20,
                empty_message="当前结果未包含抽取事件。",
            )

    with tabs[2]:
        st.markdown("### 🧾 事件质量评分")
        event_quality_df = results.get('event_quality_df', pd.DataFrame())
        if isinstance(event_quality_df, pd.DataFrame) and not event_quality_df.empty:
            quality_cols = [
                'event_id',
                'source_type',
                'title',
                'event_quality_score',
                'event_completeness_score',
                'tech_relevance_score',
                'traceability_score',
                'evidence_span_score',
                'confidence_score',
                'uncertainty_risk_score',
                'foresight_relevance_score',
                'source_reliability_score',
                'non_marketing_score',
                'event_quality_tier',
                'event_quality_reason',
            ]
            quality_cols = [c for c in quality_cols if c in event_quality_df.columns]
            render_paginated_dataframe(
                event_quality_df,
                quality_cols,
                "event_quality_table",
                page_size=20,
                empty_message="当前结果未包含事件质量评分。",
            )
            if 'event_quality_tier' in event_quality_df.columns:
                st.markdown("#### 质量层级分布")
                render_horizontal_bar_chart(event_quality_df['event_quality_tier'].value_counts(), "质量层级", "数量")
        else:
            st.info("当前结果未包含事件质量评分。")

    with tabs[3]:
        st.markdown("### 🔧 候选对象成形结果")
        candidates_df = results['candidate_forms_df']
        if not candidates_df.empty:
            unique_candidates_df = build_unique_candidate_view(candidates_df)
            unique_candidates_df = sort_by_available_scores(
                unique_candidates_df,
                ['cluster_evidence_count', 'source_count', 'total_mentions']
            )
            st.caption(
                f"当前展示 {len(unique_candidates_df)} 个唯一候选对象，原始候选成形记录共 {len(candidates_df)} 条。"
            )
            display_cols = [
                'display_candidate_name',
                'stable_object_label',
                'mechanism_core',
                'candidate_stage',
                'topic_granularity',
                'display_tier',
                'family_priority',
                'cluster_evidence_count',
                'source_count',
                'candidate_evidence_quality',
                'candidate_evidence_foresight_relevance',
                'high_foresight_evidence_count',
            ]
            display_cols = [c for c in display_cols if c in unique_candidates_df.columns]
            render_paginated_dataframe(
                unique_candidates_df,
                display_cols,
                "candidate_forms_table",
                page_size=20,
                empty_message="当前结果未包含候选对象。",
            )

            if 'topic_granularity' in candidates_df.columns:
                st.markdown("#### 主题粒度分布")
                granularity_dist = unique_candidates_df['topic_granularity'].value_counts()
                render_horizontal_bar_chart(granularity_dist, "主题粒度", "数量")

    with tabs[4]:
        st.markdown("### 📊 弱信号评分")
        scored_df = results['scored_df']
        if not scored_df.empty:
            unique_scored_df = build_unique_candidate_view(scored_df)
            score_cols = ['display_candidate_name', 'weak_signal_score', 'hotspot_score', 'source_count', 'total_mentions']
            score_cols = [c for c in score_cols if c in unique_scored_df.columns]

            sorted_df = sort_by_available_scores(
                unique_scored_df,
                ['weak_signal_score', 'hotspot_score', 'source_count', 'total_mentions']
            )
            render_paginated_dataframe(
                sorted_df,
                score_cols,
                "scored_candidates_table",
                page_size=20,
                empty_message="当前结果未包含弱信号评分候选。",
            )

            st.markdown("#### 弱信号评分分布")
            if 'weak_signal_score' in unique_scored_df.columns:
                score_chart_df = sorted_df.head(20).copy()
                labels = (
                    score_chart_df['display_candidate_name'].astype(str).tolist()
                    if 'display_candidate_name' in score_chart_df.columns
                    else [str(item) for item in score_chart_df.index.tolist()]
                )
                render_horizontal_bar_chart(
                    pd.DataFrame({
                        "候选对象": labels,
                        "弱信号分": pd.to_numeric(score_chart_df['weak_signal_score'], errors='coerce').fillna(0).tolist(),
                    }),
                    "候选对象",
                    "弱信号分",
                    max_rows=20,
                )

            reverse_validation_df = results.get('reverse_validation_df', pd.DataFrame())
            if isinstance(reverse_validation_df, pd.DataFrame) and not reverse_validation_df.empty:
                st.markdown("#### 反向验证摘要")
                rv_cols = [
                    'display_candidate_name',
                    'reverse_validation_status',
                    'source_semantic_consistency',
                    'object_semantic_alignment',
                    'release_alignment_risk',
                ]
                rv_cols = [c for c in rv_cols if c in reverse_validation_df.columns]
                if rv_cols:
                    render_paginated_dataframe(
                        reverse_validation_df,
                        rv_cols,
                        "reverse_validation_table",
                        page_size=20,
                        empty_message="当前结果未包含反向验证摘要。",
                    )

    with tabs[5]:
        st.markdown("### 🎯 弱信号生成")
        signals_df = results['signals_df']

        if 'signal_type' in signals_df.columns:
            weak_signals = signals_df[signals_df['signal_type'] == 'weak_signal']
        else:
            weak_signals = signals_df

        if not weak_signals.empty:
            def dataframe_records(value):
                if isinstance(value, pd.DataFrame):
                    return value.to_dict(orient='records') if not value.empty else []
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                return []

            def source_documents_from_raw_data(raw_data):
                if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
                    return []
                rows = []
                seen = set()
                for _, source_row in raw_data.reset_index(drop=True).iterrows():
                    source_id = safe_text(source_row.get('source_id')) or safe_text(source_row.get('id'))
                    title = safe_text(source_row.get('title')) or safe_text(source_row.get('name'))
                    text = safe_text(source_row.get('text')) or safe_text(source_row.get('content')) or safe_text(source_row.get('abstract'))
                    source_type = normalize_trace_source_type(source_row.get('source_type'))
                    if not source_id:
                        source_id = f"source_{len(rows) + 1}"
                    if source_id in seen:
                        continue
                    seen.add(source_id)
                    overview = safe_text(source_row.get('snippet')) or safe_text(source_row.get('summary')) or text or title
                    rows.append({
                        'source_id': source_id,
                        'source_type': source_type,
                        'title': title,
                        'authors': safe_text(source_row.get('authors')) or safe_text(source_row.get('author')) or safe_text(source_row.get('inventors')),
                        'org': safe_text(source_row.get('org')) or safe_text(source_row.get('organization')) or safe_text(source_row.get('source')),
                        'affiliations': safe_text(source_row.get('affiliations')) or safe_text(source_row.get('affiliation')) or safe_text(source_row.get('institution')),
                        'date': safe_text(source_row.get('date')) or safe_text(source_row.get('time')) or safe_text(source_row.get('year')),
                        'publish_time': safe_text(source_row.get('publish_time')) or safe_text(source_row.get('publish_date')) or safe_text(source_row.get('public_date')) or safe_text(source_row.get('date')),
                        'url': safe_text(source_row.get('url')) or safe_text(source_row.get('link')),
                        'keywords': safe_text(source_row.get('keywords')) or safe_text(source_row.get('tags')) or safe_text(source_row.get('entities')),
                        'venue': safe_text(source_row.get('venue')) or safe_text(source_row.get('journal')) or safe_text(source_row.get('conference')),
                        'source_name': safe_text(source_row.get('source_name')) or safe_text(source_row.get('source')) or safe_text(source_row.get('publisher')),
                        'abstract': safe_text(source_row.get('abstract')) or safe_text(source_row.get('summary')) or safe_text(source_row.get('snippet')),
                        'main_content': safe_text(source_row.get('main_content')) or safe_text(source_row.get('content')) or text,
                        'industry': safe_text(source_row.get('industry')) or safe_text(source_row.get('lz_industry')) or safe_text(source_row.get('stock_name')),
                        'document_code': safe_text(source_row.get('document_code')) or safe_text(source_row.get('report_code')) or safe_text(source_row.get('doi/arxiv_id')) or safe_text(source_row.get('doi')),
                        'classification': safe_text(source_row.get('classification')) or safe_text(source_row.get('ipc')) or safe_text(source_row.get('cpc')) or safe_text(source_row.get('type')),
                        'overview': safe_preview(overview, limit=420),
                        'full_text': text,
                        'text_sha256': safe_text(source_row.get('text_sha256')),
                    })
                return rows

            def normalize_trace_source_type(value):
                text = safe_text(value)
                mapping = {
                    '专利': 'patent',
                    '文献': 'paper',
                    '论文': 'paper',
                    '研报': 'report',
                    '资讯': 'news',
                    'patent': 'patent',
                    'paper': 'paper',
                    'literature': 'paper',
                    'report': 'report',
                    'news': 'news',
                }
                return mapping.get(text, mapping.get(text.lower(), text.lower() or 'unknown'))

            def title_key(source_type, title):
                normalized = re.sub(r"[\s_\-]+", "", safe_text(title).lower())
                normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
                return normalize_trace_source_type(source_type), normalized

            def safe_key(value):
                return re.sub(r"[^a-zA-Z0-9_-]+", "_", safe_text(value))[:80] or "item"

            source_documents = dataframe_records(results.get('source_documents_df', pd.DataFrame()))
            if not source_documents:
                source_documents = source_documents_from_raw_data(results.get('raw_data', pd.DataFrame()))

            source_by_id = {}
            source_by_title = {}
            for doc in source_documents:
                source_id = safe_text(doc.get('source_id')) or safe_text(doc.get('id'))
                if source_id:
                    source_by_id[source_id] = doc
                key = title_key(doc.get('source_type'), doc.get('title'))
                if key[1] and key not in source_by_title:
                    source_by_title[key] = doc

            links_by_signal = {}
            links_by_cluster = {}
            links_by_candidate = {}
            links_by_name = {}
            for link in dataframe_records(results.get('signal_evidence_links_df', pd.DataFrame())):
                signal_id = safe_text(link.get('signal_id'))
                if signal_id:
                    links_by_signal.setdefault(signal_id, []).append(link)
                cluster_id = safe_text(link.get('candidate_cluster_id'))
                if cluster_id:
                    links_by_cluster.setdefault(cluster_id, []).append(link)
                candidate_id = safe_text(link.get('candidate_id'))
                if candidate_id:
                    links_by_candidate.setdefault(candidate_id, []).append(link)
                candidate_name = safe_text(link.get('display_candidate_name'))
                if candidate_name:
                    links_by_name.setdefault(candidate_name, []).append(link)

            reliability_by_signal = {}
            for reliability in dataframe_records(results.get('signal_reliability_df', pd.DataFrame())):
                signal_id = safe_text(reliability.get('signal_id'))
                if signal_id:
                    reliability_by_signal[signal_id] = reliability

            def safe_bool(value):
                if isinstance(value, bool):
                    return value
                return safe_text(value).lower() in {'1', 'true', 'yes', 'y', '是', '通过'}

            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):
                        return default
                except Exception:
                    pass
                try:
                    return float(value)
                except Exception:
                    return default

            def parse_display_list(value):
                if isinstance(value, list):
                    return [safe_text(item) for item in value if safe_text(item)]
                text = safe_text(value)
                if not text:
                    return []
                if text[0] in '[{':
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return [safe_text(item) for item in parsed if safe_text(item)]
                    except Exception:
                        pass
                return [text]

            def split_display_terms(value):
                terms = []
                for item in parse_display_list(value):
                    pieces = re.split(r"[、,;；/|]+", item) if len(item) > 80 else [item]
                    for piece in pieces:
                        term = safe_text(piece)
                        if term and len(term) <= 80 and term not in terms:
                            terms.append(term)
                return terms

            def normalized_key(value):
                text = safe_text(value).lower()
                text = re.sub(r"[\s_\-]+", "", text)
                return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

            def term_matches_text(term, evidence_text, evidence_key):
                lower_term = safe_text(term).lower()
                if not lower_term:
                    return False
                if re.fullmatch(r"[a-z0-9.+#-]{1,4}", lower_term):
                    return re.search(rf"(?<![a-z0-9]){re.escape(lower_term)}(?![a-z0-9])", evidence_text) is not None
                if lower_term in evidence_text:
                    return True
                term_key = normalized_key(term)
                return len(term_key) >= 2 and term_key in evidence_key

            broad_support_terms = {
                normalized_key(term)
                for term in [
                    'ai', 'llm', 'robot', 'robots', 'robotics', 'technology', 'technologies',
                    'system', 'systems', 'method', 'methods', 'approach', 'application',
                    'control', 'planning', 'training', 'simulation', 'visual', 'vision',
                    '机器人', '控制', '技术', '方法', '系统', '视觉', '规划', '训练', '仿真',
                ]
            }
            specific_support_acronyms = {'slam', 'vla', 'rgb-d', 'lidar', 'imu', 'vr', 'ar'}
            support_alias_groups = [
                (
                    ['3D视觉', '三维视觉', '3d视觉', '3d', 'data:3d', 'point cloud', '点云'],
                    [
                        'SLAM',
                        'visual SLAM',
                        'point cloud',
                        '3D point cloud',
                        'visual feedback',
                        'virtual reality',
                        'viewpoint control',
                        'decoupled viewpoint',
                        'three-dimensional',
                        '3D vision',
                        'depth camera',
                        'RGB-D',
                        'LiDAR',
                    ],
                ),
                (
                    ['人形机器人', 'humanoid robot', 'humanoid'],
                    [
                        'immersive humanoid robot teleoperation',
                        'humanoid robot teleoperation',
                        'humanoid robot',
                        'humanoid',
                        'teleoperation',
                    ],
                ),
                (
                    ['遥操作', 'teleoperation'],
                    ['teleoperation', 'immersive teleoperation', 'viewpoint control', 'visual feedback'],
                ),
                (
                    ['点云', 'point cloud'],
                    ['point cloud', '3D point cloud', 'SLAM'],
                ),
            ]

            def is_broad_support_term(term):
                key = normalized_key(term)
                lower = safe_text(term).lower()
                if key in broad_support_terms:
                    return True
                return len(lower) <= 2 and lower not in specific_support_acronyms

            def support_specificity(term):
                text = safe_text(term)
                lower = text.lower()
                score = 0
                if is_broad_support_term(text):
                    score -= 100
                if lower in specific_support_acronyms:
                    score += 20
                if re.search(r"\d", text):
                    score += 8
                if " " in text or "-" in text:
                    score += 6
                if len(text) >= 10:
                    score += 4
                if any(marker in lower for marker in ['slam', 'point cloud', 'visual feedback', 'virtual reality', 'teleoperation', 'viewpoint']):
                    score += 10
                return score

            def finalize_support_terms(terms, limit=8):
                deduped = []
                for term in terms:
                    text = safe_text(term)
                    if text and text not in deduped:
                        deduped.append(text)
                specific = [term for term in deduped if not is_broad_support_term(term)]
                pool = specific if specific else deduped
                indexed = list(enumerate(pool))
                indexed.sort(key=lambda pair: (-support_specificity(pair[1]), pair[0]))
                return [term for _, term in indexed[:limit]]

            def signature_support_terms(value):
                terms = []
                for raw in split_display_terms(value):
                    for piece in re.split(r"\s*\|\|\s*|[;；、,，]+", raw):
                        piece = safe_text(piece)
                        if not piece:
                            continue
                        if ":" in piece:
                            piece = piece.split(":", 1)[1].strip()
                        for token in re.split(r"[|/]+", piece):
                            token = safe_text(token)
                            if token and token not in terms:
                                terms.append(token)
                return terms

            def semantic_support_terms(row, evidence, evidence_text, evidence_key, candidate_terms):
                context_text = " ".join([
                    safe_text(row.get('display_candidate_name')),
                    safe_text(row.get('tech_name')),
                    safe_text(row.get('technology')),
                    safe_text(row.get('canonical_candidate_name_en')),
                    safe_text(row.get('constraint_signature')),
                    safe_text(row.get('process_slot')),
                    safe_text(row.get('carrier_slot')),
                    safe_text(row.get('application_slot')),
                    safe_text(row.get('relation_data_modality')),
                    safe_text(evidence.get('raw_candidate_text')),
                    " ".join(candidate_terms),
                ]).lower()
                context_key = normalized_key(context_text)
                matched_terms = []
                for triggers, aliases in support_alias_groups:
                    active = any(
                        safe_text(trigger).lower() in context_text
                        or normalized_key(trigger) in context_key
                        for trigger in triggers
                    )
                    if not active:
                        continue
                    for alias in aliases:
                        if alias not in matched_terms and term_matches_text(alias, evidence_text, evidence_key):
                            matched_terms.append(alias)
                return matched_terms

            def derive_support_terms(row, evidence):
                terms = []

                def add(value):
                    for term in split_display_terms(value):
                        if term not in terms:
                            terms.append(term)
                        if len(terms) >= 32:
                            return

                for field in ['support_terms', 'matched_terms', 'matched_term', 'support_term']:
                    add(evidence.get(field))

                evidence_text = " ".join([
                    safe_text(evidence.get('title')),
                    safe_text(evidence.get('overview')),
                    safe_text(evidence.get('evidence_excerpt')),
                    safe_text(evidence.get('full_text'))[:2000],
                    safe_text(evidence.get('main_content'))[:2000],
                    safe_text(evidence.get('abstract')),
                    safe_text(evidence.get('keywords')),
                ]).lower()
                evidence_key = normalized_key(evidence_text)
                candidate_terms = []
                for field in [
                    'display_candidate_name',
                    'tech_name',
                    'technology',
                    'canonical_candidate_name_en',
                    'matched_term',
                    'mechanism_core',
                    'relation_target',
                    'relation_task',
                    'relation_method',
                    'relation_data_modality',
                    'constraint_signature',
                    'raw_candidate_text',
                    'tech_object_slot',
                    'capability_slot',
                    'process_slot',
                    'carrier_slot',
                    'application_slot',
                    'display_candidate_aliases',
                    'mechanism_core_tokens',
                    'task_constraint_tokens',
                    'object_modifier_tokens',
                    'data_modifier_tokens',
                    'method_modifier_tokens',
                ]:
                    candidate_terms.extend(split_display_terms(row.get(field)))
                candidate_terms.extend(signature_support_terms(row.get('constraint_signature')))
                candidate_terms.extend(split_display_terms(evidence.get('keywords')))

                for term in semantic_support_terms(row, evidence, evidence_text, evidence_key, candidate_terms):
                    if term not in terms:
                        terms.append(term)

                for term in candidate_terms:
                    if len(terms) >= 32:
                        break
                    if term not in terms and term_matches_text(term, evidence_text, evidence_key):
                        terms.append(term)

                finalized = finalize_support_terms(terms)
                if not finalized:
                    for field in ['display_candidate_name', 'mechanism_core']:
                        add(row.get(field))
                        if terms:
                            break
                    finalized = finalize_support_terms(terms)
                return finalized[:8]

            def to_json_safe(value):
                if isinstance(value, dict):
                    return {str(k): to_json_safe(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [to_json_safe(item) for item in value]
                if isinstance(value, tuple):
                    return [to_json_safe(item) for item in value]
                try:
                    if pd.isna(value):
                        return ""
                except Exception:
                    pass
                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except Exception:
                        pass
                return value

            def find_source_doc(evidence):
                source_id = (
                    safe_text(evidence.get('source_id'))
                    or safe_text(evidence.get('id'))
                    or safe_text(evidence.get('record_id'))
                )
                if source_id and source_id in source_by_id:
                    return source_by_id[source_id]
                key = title_key(evidence.get('source_type'), evidence.get('title'))
                return source_by_title.get(key)

            def evidence_links_for_signal(row):
                signal_id = safe_text(row.get('signal_id'))
                if signal_id and signal_id in links_by_signal:
                    return links_by_signal[signal_id]
                cluster_id = safe_text(row.get('candidate_cluster_id'))
                if cluster_id and cluster_id in links_by_cluster:
                    return links_by_cluster[cluster_id]
                candidate_id = safe_text(row.get('candidate_id'))
                if candidate_id and candidate_id in links_by_candidate:
                    return links_by_candidate[candidate_id]
                candidate_name = safe_text(row.get('display_candidate_name'))
                if candidate_name and candidate_name in links_by_name:
                    return links_by_name[candidate_name]
                return []

            def evidence_records_for_signal(row):
                link_records = evidence_links_for_signal(row)
                source_records = link_records if link_records else [
                    item for item in safe_list(row.get('evidence_items')) if isinstance(item, dict)
                ]
                records = []
                seen = set()
                for idx, item in enumerate(source_records, 1):
                    evidence = dict(item)
                    doc = find_source_doc(evidence) or {}
                    evidence_id = safe_text(evidence.get('evidence_id')) or f"{safe_text(row.get('signal_id')) or 'SIG'}-EV{idx:03d}"
                    source_id = (
                        safe_text(evidence.get('source_id'))
                        or safe_text(evidence.get('id'))
                        or safe_text(evidence.get('record_id'))
                        or safe_text(doc.get('source_id'))
                    )
                    title = safe_text(evidence.get('title')) or safe_text(doc.get('title'))
                    dedupe_key = (evidence_id, source_id, title)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    full_text = safe_text(doc.get('full_text')) or safe_text(evidence.get('full_text')) or safe_text(evidence.get('text'))
                    overview = (
                        safe_text(evidence.get('evidence_overview'))
                        or safe_text(evidence.get('snippet'))
                        or safe_text(doc.get('overview'))
                        or safe_text(evidence.get('text'))
                    )
                    evidence['evidence_id'] = evidence_id
                    evidence['source_id'] = source_id
                    evidence['source_type'] = safe_text(evidence.get('source_type')) or safe_text(doc.get('source_type'))
                    evidence['title'] = title
                    evidence['authors'] = safe_text(evidence.get('authors')) or safe_text(doc.get('authors'))
                    evidence['org'] = safe_text(evidence.get('org')) or safe_text(doc.get('org'))
                    evidence['affiliations'] = safe_text(evidence.get('affiliations')) or safe_text(doc.get('affiliations'))
                    evidence['date'] = safe_text(evidence.get('date')) or safe_text(doc.get('date'))
                    evidence['publish_time'] = safe_text(evidence.get('publish_time')) or safe_text(doc.get('publish_time'))
                    evidence['url'] = safe_text(evidence.get('url')) or safe_text(doc.get('url'))
                    evidence['keywords'] = safe_text(evidence.get('keywords')) or safe_text(doc.get('keywords'))
                    evidence['venue'] = safe_text(evidence.get('venue')) or safe_text(doc.get('venue'))
                    evidence['source_name'] = safe_text(evidence.get('source_name')) or safe_text(doc.get('source_name'))
                    evidence['abstract'] = safe_text(evidence.get('abstract')) or safe_text(doc.get('abstract'))
                    evidence['main_content'] = safe_text(evidence.get('main_content')) or safe_text(doc.get('main_content'))
                    evidence['industry'] = safe_text(evidence.get('industry')) or safe_text(doc.get('industry'))
                    evidence['document_code'] = safe_text(evidence.get('document_code')) or safe_text(doc.get('document_code'))
                    evidence['classification'] = safe_text(evidence.get('classification')) or safe_text(doc.get('classification'))
                    evidence['overview'] = safe_preview(overview, limit=420)
                    evidence['full_text'] = full_text
                    evidence['text_sha256'] = safe_text(evidence.get('text_sha256')) or safe_text(doc.get('text_sha256'))
                    evidence['traceability_status'] = safe_text(evidence.get('traceability_status')) or (
                        'traceable' if doc else 'partial_traceable'
                    )
                    support_terms = derive_support_terms(row, evidence)
                    if support_terms:
                        evidence['support_terms'] = support_terms
                    records.append(evidence)
                return records

            def infer_signal_reliability(row, evidence_records):
                signal_id = safe_text(row.get('signal_id'))
                existing = reliability_by_signal.get(signal_id)
                if existing:
                    source_types = parse_display_list(existing.get('source_types'))
                    if not source_types:
                        source_types = sorted({
                            safe_text(item.get('source_type'))
                            for item in evidence_records
                            if safe_text(item.get('source_type'))
                        })
                    return {
                        'evidence_gate_passed': safe_bool(existing.get('evidence_gate_passed')),
                        'evidence_gate_status': safe_text(existing.get('evidence_gate_status')) or 'not_evaluated',
                        'evidence_gate_reason': safe_text(existing.get('evidence_gate_reason')),
                        'reliable_evidence_count': int(safe_float(existing.get('reliable_evidence_count'))),
                        'traceable_evidence_count': int(safe_float(existing.get('traceable_evidence_count'))),
                        'low_quality_evidence_count': int(safe_float(existing.get('low_quality_evidence_count'))),
                        'missing_source_id_count': int(safe_float(existing.get('missing_source_id_count'))),
                        'missing_text_hash_count': int(safe_float(existing.get('missing_text_hash_count'))),
                        'source_types': source_types,
                        'top_evidence_ids': parse_display_list(existing.get('top_evidence_ids')),
                    }

                reliable_count = 0
                traceable_count = 0
                low_quality_count = 0
                missing_source_count = 0
                missing_hash_count = 0
                source_types = []
                top_evidence_ids = []
                for evidence in evidence_records:
                    source_id = safe_text(evidence.get('source_id'))
                    text_hash = safe_text(evidence.get('text_sha256'))
                    traceable = safe_text(evidence.get('traceability_status')) == 'traceable' and bool(source_id)
                    quality = safe_float(evidence.get('event_quality_score'))
                    has_text = bool(
                        safe_text(evidence.get('title'))
                        or safe_text(evidence.get('overview'))
                        or safe_text(evidence.get('evidence_excerpt'))
                    )
                    if traceable:
                        traceable_count += 1
                    if quality > 0 and quality < 4:
                        low_quality_count += 1
                    if not source_id:
                        missing_source_count += 1
                    if not text_hash:
                        missing_hash_count += 1
                    if traceable and quality >= 4 and has_text and text_hash:
                        reliable_count += 1
                        evidence_id = safe_text(evidence.get('evidence_id'))
                        if evidence_id:
                            top_evidence_ids.append(evidence_id)
                    source_type = safe_text(evidence.get('source_type'))
                    if source_type and source_type not in source_types:
                        source_types.append(source_type)

                is_weak_signal = safe_text(row.get('signal_type')) == 'weak_signal'
                gate_passed = reliable_count >= 1 if is_weak_signal else True
                if is_weak_signal and not gate_passed:
                    reason = '无可回连、质量达标且可校验的证据'
                    status = 'evidence_insufficient_candidate'
                elif is_weak_signal:
                    reason = '至少1条证据可回连源文本、质量达标且哈希可校验'
                    status = 'confirmed_weak_signal'
                else:
                    reason = ''
                    status = 'not_applicable'
                return {
                    'evidence_gate_passed': gate_passed,
                    'evidence_gate_status': status,
                    'evidence_gate_reason': reason,
                    'reliable_evidence_count': reliable_count,
                    'traceable_evidence_count': traceable_count,
                    'low_quality_evidence_count': low_quality_count,
                    'missing_source_id_count': missing_source_count,
                    'missing_text_hash_count': missing_hash_count,
                    'source_types': source_types,
                    'top_evidence_ids': top_evidence_ids[:10],
                }

            def render_source_detail(evidence, detail_key, signal_row=None):
                signal_row = signal_row if signal_row is not None else {}

                def text_key(value):
                    return normalized_key(value)

                def is_near_duplicate(left, right):
                    left_key = text_key(left)
                    right_key = text_key(right)
                    if not left_key or not right_key:
                        return False
                    if left_key == right_key:
                        return True
                    short_key, long_key = sorted([left_key, right_key], key=len)
                    return len(short_key) >= 30 and short_key in long_key and len(long_key) <= len(short_key) * 1.35

                def choose_rich_content(overview_text, abstract_text, main_text, full_text):
                    candidates = [
                        ("主要内容", main_text),
                        ("完整源文本", full_text),
                        ("摘要", abstract_text),
                    ]
                    for label, text in candidates:
                        clean = safe_text(text)
                        if len(clean) < 180:
                            continue
                        if is_near_duplicate(clean, overview_text):
                            continue
                        return label, clean
                    return "", ""

                def compact_paragraph(value, limit=460):
                    text = re.sub(r"\s+", " ", safe_text(value)).strip()
                    if len(text) <= limit:
                        return text
                    return text[:limit].rstrip(" ,，;；") + "..."

                def add_unique(items, value, limit=8):
                    for term in split_display_terms(value):
                        if term and term not in items:
                            items.append(term)
                        if len(items) >= limit:
                            return

                def readable_signature_terms(value):
                    terms = []
                    for raw in split_display_terms(value):
                        for piece in re.split(r"\s*\|\|\s*|[;；、,，]+", raw):
                            piece = safe_text(piece)
                            if not piece:
                                continue
                            if ":" in piece:
                                piece = piece.split(":", 1)[1].strip()
                            if piece and piece not in terms:
                                terms.append(piece)
                    return terms[:8]

                def relevant_sentences(texts, terms, limit=3):
                    normalized_terms = [safe_text(term).lower() for term in terms if safe_text(term)]
                    selected = []
                    for text in texts:
                        clean = re.sub(r"\s+", " ", safe_text(text)).strip()
                        if not clean:
                            continue
                        parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=[。！？!?；;])|(?<=[.])\s+", clean)
                        if len(parts) <= 1 and len(clean) > 260:
                            parts = re.split(r"(?<=,)\s+|(?<=，)", clean)
                        for sentence in parts:
                            sentence = safe_text(sentence)
                            if len(sentence) < 18:
                                continue
                            lower = sentence.lower()
                            matched = [
                                term
                                for term in normalized_terms
                                if term and (term in lower or normalized_key(term) in normalized_key(sentence))
                            ]
                            if matched and not any(is_near_duplicate(sentence, item) for item in selected):
                                selected.append(compact_paragraph(sentence, 300))
                            if len(selected) >= limit:
                                return selected
                    return selected

                def build_detail_review():
                    title = safe_text(evidence.get('title'))
                    candidate_name = (
                        safe_text(evidence.get('display_candidate_name'))
                        or safe_text(signal_row.get('display_candidate_name'))
                        or safe_text(signal_row.get('candidate_name'))
                    )
                    source_type = safe_text(evidence.get('source_type')) or "来源"
                    date = safe_text(evidence.get('publish_time') or evidence.get('date'))
                    org = safe_text(evidence.get('affiliations') or evidence.get('org'))
                    support_terms = (
                        split_display_terms(evidence.get('support_terms'))
                        or split_display_terms(evidence.get('matched_terms'))
                    )
                    keywords = split_display_terms(evidence.get('keywords'))
                    overview_text = safe_text(evidence.get('overview') or evidence.get('evidence_overview'))
                    excerpt = safe_text(evidence.get('evidence_excerpt'))
                    abstract_text = safe_text(evidence.get('abstract'))
                    main_text = safe_text(evidence.get('main_content'))
                    full_text = safe_text(evidence.get('full_text'))
                    raw_candidate = safe_text(evidence.get('raw_candidate_text')) or safe_text(signal_row.get('raw_candidate_text'))
                    mechanism = safe_text(signal_row.get('mechanism_core')) or safe_text(evidence.get('mechanism_core'))
                    relation_target = safe_text(signal_row.get('relation_target')) or safe_text(evidence.get('relation_target'))
                    relation_task = safe_text(signal_row.get('relation_task')) or safe_text(evidence.get('relation_task'))
                    relation_data = safe_text(signal_row.get('relation_data_modality')) or safe_text(evidence.get('relation_data_modality'))
                    relation_method = safe_text(signal_row.get('relation_method')) or safe_text(evidence.get('relation_method'))
                    constraint_terms = readable_signature_terms(signal_row.get('constraint_signature'))

                    meta_parts = []
                    if source_type:
                        meta_parts.append(source_type)
                    if date:
                        meta_parts.append(date)
                    if org:
                        meta_parts.append(org)
                    meta_text = " / ".join(meta_parts)

                    relation_terms = []
                    add_unique(relation_terms, candidate_name)
                    add_unique(relation_terms, raw_candidate)
                    add_unique(relation_terms, support_terms)
                    add_unique(relation_terms, mechanism)
                    add_unique(relation_terms, [relation_target, relation_task, relation_data, relation_method])
                    add_unique(relation_terms, constraint_terms)
                    add_unique(relation_terms, keywords[:4])

                    paragraphs = []
                    lead = f"这篇{source_type}被关联到"
                    lead += f"“{candidate_name}”" if candidate_name else "当前技术词条"
                    lead += "，不是因为来源分数，而是因为文本中出现了能支撑该词条的技术线索"
                    if support_terms:
                        lead += f"：{'、'.join(support_terms[:6])}"
                    elif raw_candidate:
                        lead += f"：{raw_candidate}"
                    lead += "。"
                    paragraphs.append(lead)

                    source_intro = f"文献来源：{meta_text or source_type}"
                    if title:
                        source_intro += f"，题名为“{title}”"
                    source_intro += "。"
                    paragraphs.append(source_intro)

                    content_sources = []
                    for text in [excerpt, abstract_text, overview_text, main_text, full_text]:
                        clean = safe_text(text)
                        if clean and not any(is_near_duplicate(clean, existing) for existing in content_sources):
                            content_sources.append(clean)
                    matched_sentences = relevant_sentences(content_sources, relation_terms, limit=3)
                    if matched_sentences:
                        paragraphs.append("文献内容线索：" + " ".join(matched_sentences))
                    elif content_sources:
                        paragraphs.append("文献内容线索：" + compact_paragraph(content_sources[0], 520))

                    mapping_bits = []
                    if raw_candidate and candidate_name and normalized_key(raw_candidate) != normalized_key(candidate_name):
                        mapping_bits.append(f"原文候选表达“{raw_candidate}”被归并到展示词条“{candidate_name}”")
                    elif raw_candidate:
                        mapping_bits.append(f"原文候选表达为“{raw_candidate}”")
                    if mechanism:
                        mapping_bits.append(f"其中“{mechanism}”体现了系统抽取到的核心技术机制")
                    role_bits = []
                    if relation_target:
                        role_bits.append(f"对象/目标是“{relation_target}”")
                    if relation_task:
                        role_bits.append(f"任务场景是“{relation_task}”")
                    if relation_data:
                        role_bits.append(f"数据或模态线索是“{relation_data}”")
                    if relation_method:
                        role_bits.append(f"方法线索是“{relation_method}”")
                    if role_bits:
                        mapping_bits.append("；".join(role_bits))
                    elif constraint_terms:
                        mapping_bits.append(f"系统还识别到这些限定信息：{'、'.join(constraint_terms[:6])}")
                    if mapping_bits:
                        paragraphs.append("关联方式：" + "；".join(mapping_bits) + "。")

                    if relation_terms:
                        paragraphs.append(
                            "这篇文献提供了一个具体研究场景或方法实例，说明“"
                            + (candidate_name or raw_candidate or relation_terms[0])
                            + "”并非孤立关键词，而是在源文本中由"
                            + "、".join(relation_terms[:6])
                            + "等线索共同支撑。"
                        )

                    return "\n\n".join(paragraphs)

                metadata_items = [
                    ('标题', evidence.get('title')),
                    ('作者/发明人', evidence.get('authors')),
                    ('机构/申请人', evidence.get('affiliations') or evidence.get('org')),
                    ('来源/刊物', evidence.get('venue') or evidence.get('source_name')),
                    ('来源类型', evidence.get('source_type')),
                    ('发布时间', evidence.get('publish_time') or evidence.get('date')),
                    ('关键词/标签', evidence.get('keywords')),
                    ('行业/对象', evidence.get('industry')),
                    ('文档编号/DOI', evidence.get('document_code')),
                    ('分类号/类型', evidence.get('classification')),
                    ('URL', evidence.get('url')),
                ]
                metadata_rows = [
                    {'字段': label, '内容': safe_text(value)}
                    for label, value in metadata_items
                    if safe_text(value)
                ]
                if metadata_rows:
                    st.markdown("**来源元数据**")
                    st.dataframe(pd.DataFrame(metadata_rows), use_container_width=True, hide_index=True)

                audit_items = [
                    ('source_id', evidence.get('source_id')),
                    ('追溯状态', evidence.get('traceability_status')),
                    ('证据质量分', evidence.get('event_quality_score')),
                    ('预见相关度', evidence.get('foresight_relevance_score')),
                    ('证据片段分', evidence.get('evidence_span_score')),
                    ('置信度分', evidence.get('confidence_score')),
                    ('text_sha256', evidence.get('text_sha256')),
                ]
                audit_rows = [
                    {'字段': label, '内容': safe_text(value)}
                    for label, value in audit_items
                    if safe_text(value)
                ]
                if audit_rows:
                    with st.expander("内部校验信息", expanded=False):
                        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

                abstract = safe_text(evidence.get('abstract'))
                if abstract:
                    st.markdown("**摘要/概述**")
                    st.write(abstract)

                overview_text = safe_text(evidence.get('overview') or evidence.get('evidence_overview'))
                main_content = safe_text(evidence.get('main_content'))
                full_text = safe_text(evidence.get('full_text'))
                source_content_label, source_content_text = choose_rich_content(overview_text, abstract, main_content, full_text)
                content_label = "关联说明"
                content_text = build_detail_review()
                if not content_text:
                    content_label = source_content_label or "主要内容"
                    content_text = source_content_text
                if content_text:
                    st.markdown("**主要内容**")
                    st.text_area(
                        content_label or "主要内容",
                        value=content_text,
                        height=260,
                        key=f"main_{detail_key}",
                        disabled=True,
                        label_visibility="collapsed",
                    )

                if (
                    source_content_text
                    and len(source_content_text) >= 180
                    and source_content_text != content_text
                    and not is_near_duplicate(source_content_text, content_text)
                ):
                    st.markdown(f"**{source_content_label or '源文本内容'}**")
                    st.text_area(
                        source_content_label or "源文本内容",
                        value=source_content_text,
                        height=260,
                        key=f"full_{detail_key}",
                        disabled=True,
                        label_visibility="collapsed",
                    )

            display_cols = [
                'signal_id',
                'display_candidate_name',
                'mechanism_core',
                'weak_signal_score',
                'candidate_evidence_quality',
                'candidate_evidence_foresight_relevance',
                'high_foresight_evidence_count',
                'signal_bucket',
                'source_count',
                'cluster_evidence_count',
                'evidence_gate_status',
                'reliable_evidence_count',
                'traceable_evidence_count',
            ]

            sorted_signals = sort_by_available_scores(
                weak_signals,
                ['weak_signal_score', 'hotspot_score', 'cluster_evidence_count', 'source_count']
            ).copy()
            if 'signal_id' not in sorted_signals.columns:
                sorted_signals['signal_id'] = [f"WS{i + 1:03d}" for i in range(len(sorted_signals))]
            else:
                sorted_signals['signal_id'] = [
                    safe_text(value) or f"WS{i + 1:03d}"
                    for i, value in enumerate(sorted_signals['signal_id'].tolist())
                ]

            def extract_year(value):
                match = re.search(r"(19|20)\d{2}", safe_text(value))
                return int(match.group(0)) if match else None

            signal_evidence_cache = {}
            signal_reliability_cache = {}
            meta_rows = []
            for row_index, row in sorted_signals.iterrows():
                signal_id = safe_text(row.get('signal_id')) or f"WS{len(meta_rows) + 1:03d}"
                evidence_records = evidence_records_for_signal(row)
                reliability = infer_signal_reliability(row, evidence_records)
                signal_evidence_cache[signal_id] = evidence_records
                signal_reliability_cache[signal_id] = reliability
                qualities = [
                    safe_float(item.get('event_quality_score'))
                    for item in evidence_records
                    if safe_float(item.get('event_quality_score')) > 0
                ]
                foresight_scores = [
                    safe_float(item.get('foresight_relevance_score'))
                    for item in evidence_records
                    if safe_float(item.get('foresight_relevance_score')) > 0
                ]
                years = [extract_year(item.get('date')) for item in evidence_records]
                years = [year for year in years if year is not None]
                trace_statuses = sorted({
                    safe_text(item.get('traceability_status')) or 'partial_traceable'
                    for item in evidence_records
                })
                source_type_values = sorted({
                    safe_text(item.get('source_type'))
                    for item in evidence_records
                    if safe_text(item.get('source_type'))
                })
                meta_rows.append({
                    '_row_index': row_index,
                    'evidence_gate_passed': reliability.get('evidence_gate_passed', False),
                    'evidence_gate_status': reliability.get('evidence_gate_status', 'not_evaluated'),
                    'evidence_gate_reason': reliability.get('evidence_gate_reason', ''),
                    'reliable_evidence_count': reliability.get('reliable_evidence_count', 0),
                    'traceable_evidence_count': reliability.get('traceable_evidence_count', 0),
                    'low_quality_evidence_count': reliability.get('low_quality_evidence_count', 0),
                    'missing_source_id_count': reliability.get('missing_source_id_count', 0),
                    'missing_text_hash_count': reliability.get('missing_text_hash_count', 0),
                    'evidence_source_types': source_type_values,
                    'evidence_traceability_statuses': trace_statuses,
                    'max_evidence_quality': max(qualities) if qualities else 0.0,
                    'max_evidence_foresight_relevance': max(foresight_scores) if foresight_scores else 0.0,
                    'min_evidence_year': min(years) if years else None,
                    'max_evidence_year': max(years) if years else None,
                    'has_evidence_url': any(bool(safe_text(item.get('url'))) for item in evidence_records),
                    'evidence_record_count': len(evidence_records),
                })

            if meta_rows:
                meta_df = pd.DataFrame(meta_rows).set_index('_row_index')
                for column in meta_df.columns:
                    sorted_signals[column] = meta_df[column]

            display_cols = [c for c in display_cols if c in sorted_signals.columns]
            confirmed_count = int(sorted_signals['evidence_gate_passed'].fillna(False).astype(bool).sum()) if 'evidence_gate_passed' in sorted_signals.columns else len(sorted_signals)
            pending_count = max(len(sorted_signals) - confirmed_count, 0)

            status_filter = st.selectbox(
                "证据门禁",
                options=["已确认弱信号", "证据不足/待复核", "全部"],
                index=0,
            )
            filter_cols = st.columns([1.3, 1.3, 1.0, 1.0, 1.0])
            all_source_types = sorted({
                source_type
                for items in sorted_signals.get('evidence_source_types', pd.Series(dtype=object)).tolist()
                for source_type in (items if isinstance(items, list) else [])
                if source_type
            })
            all_trace_statuses = sorted({
                status
                for items in sorted_signals.get('evidence_traceability_statuses', pd.Series(dtype=object)).tolist()
                for status in (items if isinstance(items, list) else [])
                if status
            })
            with filter_cols[0]:
                selected_source_types = st.multiselect("来源类型", all_source_types, default=[])
            with filter_cols[1]:
                selected_trace_statuses = st.multiselect("追溯状态", all_trace_statuses, default=[])
            with filter_cols[2]:
                min_quality_filter = st.slider("最低证据质量", 0.0, 10.0, 0.0, 0.5)
            with filter_cols[3]:
                min_foresight_filter = st.slider("最低预见相关度", 0.0, 10.0, 0.0, 0.5)
            with filter_cols[4]:
                only_with_url = st.checkbox("仅带URL证据", value=False)

            available_years = [
                int(year)
                for year in sorted_signals.get('min_evidence_year', pd.Series(dtype=object)).dropna().tolist()
            ] + [
                int(year)
                for year in sorted_signals.get('max_evidence_year', pd.Series(dtype=object)).dropna().tolist()
            ]
            year_range = None
            if available_years:
                min_year, max_year = min(available_years), max(available_years)
                if min_year < max_year:
                    year_range = st.slider("证据年份范围", min_year, max_year, (min_year, max_year), 1)

            filtered_signals = sorted_signals.copy()
            if status_filter == "已确认弱信号" and 'evidence_gate_passed' in filtered_signals.columns:
                filtered_signals = filtered_signals[filtered_signals['evidence_gate_passed'].fillna(False).astype(bool)]
            elif status_filter == "证据不足/待复核" and 'evidence_gate_passed' in filtered_signals.columns:
                filtered_signals = filtered_signals[~filtered_signals['evidence_gate_passed'].fillna(False).astype(bool)]
            if selected_source_types and 'evidence_source_types' in filtered_signals.columns:
                selected_set = set(selected_source_types)
                filtered_signals = filtered_signals[
                    filtered_signals['evidence_source_types'].apply(
                        lambda items: bool(selected_set & set(items if isinstance(items, list) else []))
                    )
                ]
            if selected_trace_statuses and 'evidence_traceability_statuses' in filtered_signals.columns:
                selected_set = set(selected_trace_statuses)
                filtered_signals = filtered_signals[
                    filtered_signals['evidence_traceability_statuses'].apply(
                        lambda items: bool(selected_set & set(items if isinstance(items, list) else []))
                    )
                ]
            if min_quality_filter > 0 and 'max_evidence_quality' in filtered_signals.columns:
                filtered_signals = filtered_signals[
                    pd.to_numeric(filtered_signals['max_evidence_quality'], errors='coerce').fillna(0) >= min_quality_filter
                ]
            if min_foresight_filter > 0 and 'max_evidence_foresight_relevance' in filtered_signals.columns:
                filtered_signals = filtered_signals[
                    pd.to_numeric(filtered_signals['max_evidence_foresight_relevance'], errors='coerce').fillna(0) >= min_foresight_filter
                ]
            if only_with_url and 'has_evidence_url' in filtered_signals.columns:
                filtered_signals = filtered_signals[filtered_signals['has_evidence_url'].fillna(False).astype(bool)]
            if year_range and 'min_evidence_year' in filtered_signals.columns and 'max_evidence_year' in filtered_signals.columns:
                start_year, end_year = year_range
                filtered_signals = filtered_signals[
                    (pd.to_numeric(filtered_signals['max_evidence_year'], errors='coerce').fillna(0) >= start_year)
                    & (pd.to_numeric(filtered_signals['min_evidence_year'], errors='coerce').fillna(9999) <= end_year)
                ]

            filtered_evidence_rows = []
            for _, signal_row in filtered_signals.iterrows():
                signal_id = safe_text(signal_row.get('signal_id'))
                for evidence in signal_evidence_cache.get(signal_id, []):
                    evidence_row = dict(evidence)
                    evidence_row['signal_id'] = signal_id
                    evidence_row['display_candidate_name'] = safe_text(signal_row.get('display_candidate_name'))
                    filtered_evidence_rows.append(evidence_row)

            total_signals = len(filtered_signals)
            page_size = 20
            total_pages = max(1, (total_signals + page_size - 1) // page_size)
            page_key = f"weak_signal_page_{safe_key(results.get('result_dir', 'active'))}"
            if page_key in st.session_state and int(st.session_state[page_key]) > total_pages:
                st.session_state[page_key] = total_pages

            page_cols = st.columns([1, 1, 1, 1, 2])
            with page_cols[0]:
                st.metric("确认弱信号", confirmed_count)
            with page_cols[1]:
                st.metric("待复核候选", pending_count)
            with page_cols[2]:
                st.metric("当前筛选", total_signals)
            with page_cols[3]:
                st.metric("每页展示", page_size)
            with page_cols[4]:
                page_number = int(st.number_input(
                    "页码",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key=page_key,
                ))

            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    label="导出当前筛选弱信号CSV",
                    data=filtered_signals[display_cols].to_csv(index=False).encode('utf-8-sig'),
                    file_name="filtered_weak_signals.csv",
                    mime="text/csv",
                    disabled=filtered_signals.empty,
                )
            with export_cols[1]:
                st.download_button(
                    label="导出当前筛选证据链JSON",
                    data=json.dumps(
                        [to_json_safe(item) for item in filtered_evidence_rows],
                        ensure_ascii=False,
                        indent=2,
                    ).encode('utf-8'),
                    file_name="filtered_signal_evidence_links.json",
                    mime="application/json",
                    disabled=not filtered_evidence_rows,
                )

            start = (page_number - 1) * page_size
            end = min(start + page_size, total_signals)
            page_signals = filtered_signals.iloc[start:end].copy()
            if total_signals:
                st.caption(f"第 {page_number}/{total_pages} 页，展示第 {start + 1}-{end} 条")
                st.dataframe(page_signals[display_cols], use_container_width=True)
            else:
                st.info("当前筛选条件下没有弱信号。")

            for idx, row in page_signals.iterrows():
                name = row.get('display_candidate_name', '未知')
                score = get_display_score(row)
                mechanism = row.get('mechanism_core', '')
                signal_id = safe_text(row.get('signal_id')) or f"WS{idx + 1:03d}"

                with st.expander(f"🎯 {signal_id} · {name} (得分: {score:.2f})"):
                    evidence_records = signal_evidence_cache.get(signal_id, evidence_records_for_signal(row))
                    reliability = signal_reliability_cache.get(signal_id, infer_signal_reliability(row, evidence_records))
                    gate_passed = bool(reliability.get('evidence_gate_passed'))
                    gate_status = safe_text(reliability.get('evidence_gate_status')) or 'not_evaluated'
                    gate_reason = safe_text(reliability.get('evidence_gate_reason'))
                    if gate_passed:
                        st.success(f"证据门禁通过：{gate_reason or '至少1条可靠证据可追溯'}")
                    else:
                        st.warning(f"证据不足/待复核：{gate_reason or '缺少可靠证据'}")

                    explain_cols = st.columns(4)
                    with explain_cols[0]:
                        st.metric("弱信号分", f"{score:.2f}")
                    with explain_cols[1]:
                        st.metric("可靠证据", int(reliability.get('reliable_evidence_count', 0)))
                    with explain_cols[2]:
                        st.metric("可追溯证据", int(reliability.get('traceable_evidence_count', 0)))
                    with explain_cols[3]:
                        st.metric("门禁状态", gate_status)

                    st.markdown(f"**机制核**: {mechanism}")
                    st.markdown(f"**来源数**: {row.get('source_count', 0)}")
                    st.markdown(f"**证据数**: {row.get('cluster_evidence_count', 0)}")

                    signal_bucket = safe_text(row.get('signal_bucket'))
                    stage_hypothesis = safe_text(row.get('stage_hypothesis'))
                    reverse_status = safe_text(row.get('reverse_validation_status'))
                    family_priority = safe_text(row.get('family_priority'))
                    family_consistency = safe_text(row.get('family_semantic_consistency'))
                    alignment_risk = safe_text(row.get('release_alignment_risk'))
                    evidence_quality = safe_text(row.get('candidate_evidence_quality'))
                    evidence_foresight = safe_text(row.get('candidate_evidence_foresight_relevance'))
                    constraint_signature = safe_preview(row.get('constraint_signature'), limit=120)
                    temporal_tier = safe_text(row.get('temporal_validation_tier'))
                    temporal_reason = safe_preview(row.get('temporal_validation_reason'), limit=180)
                    change_type = safe_text(row.get('weak_signal_change_type'))
                    change_reason = safe_preview(row.get('change_reason'), limit=180)

                    if signal_bucket:
                        st.markdown(f"**研究分桶**: {signal_bucket}")
                    if stage_hypothesis:
                        st.markdown(f"**阶段假设**: {stage_hypothesis}")
                    if reverse_status:
                        st.markdown(f"**反向验证**: {reverse_status}")
                    if family_priority or family_consistency:
                        family_parts = [part for part in [family_priority, family_consistency] if part]
                        st.markdown(f"**对象族承接**: {' / '.join(family_parts)}")
                    if alignment_risk:
                        st.markdown(f"**发布对齐风险**: {alignment_risk}")
                    if evidence_quality:
                        st.markdown(f"**证据质量**: {evidence_quality}")
                    if evidence_foresight:
                        st.markdown(f"**证据预见相关度**: {evidence_foresight}")
                    if constraint_signature:
                        st.markdown(f"**约束签名**: {constraint_signature}")
                    if temporal_tier or temporal_reason:
                        st.markdown(f"**时间验证**: {' / '.join([part for part in [temporal_tier, temporal_reason] if part])}")
                    if change_type or change_reason:
                        st.markdown(f"**弱信号变化**: {' / '.join([part for part in [change_type, change_reason] if part])}")

                    basis_rows = []
                    for label, column in [
                        ('弱信号事件占比', 'weak_signal_event_ratio'),
                        ('低关注线索', 'low_attention_ratio'),
                        ('小众主体线索', 'niche_actor_ratio'),
                        ('非主导表述', 'non_dominant_ratio'),
                        ('跨域融合线索', 'cross_domain_ratio'),
                        ('追溯载体占比', 'traceable_ratio'),
                        ('候选证据质量', 'candidate_evidence_quality'),
                        ('核心证据质量', 'candidate_core_evidence_quality'),
                        ('证据预见相关度', 'candidate_evidence_foresight_relevance'),
                        ('高预见证据数', 'high_foresight_evidence_count'),
                    ]:
                        value = safe_text(row.get(column))
                        if value:
                            basis_rows.append({'解释维度': label, '值': value})
                    if basis_rows:
                        st.markdown("**识别依据**")
                        st.dataframe(pd.DataFrame(basis_rows), use_container_width=True)

                    validation_rows = []
                    for label, value in [
                        ('反向验证', reverse_status),
                        ('对象族承接', ' / '.join([part for part in [family_priority, family_consistency] if part])),
                        ('证据门禁原因', gate_reason),
                        ('低质量证据数', reliability.get('low_quality_evidence_count', 0)),
                        ('缺失source_id证据数', reliability.get('missing_source_id_count', 0)),
                        ('缺失文本哈希证据数', reliability.get('missing_text_hash_count', 0)),
                    ]:
                        text = safe_text(value)
                        if text:
                            validation_rows.append({'校验项': label, '结果': text})
                    if validation_rows:
                        st.markdown("**可靠性校验**")
                        st.dataframe(pd.DataFrame(validation_rows), use_container_width=True)

                    signal_package = {
                        'signal': to_json_safe(row.to_dict()),
                        'reliability': to_json_safe(reliability),
                        'evidence': [to_json_safe(item) for item in evidence_records],
                    }
                    download_cols = st.columns(2)
                    with download_cols[0]:
                        st.download_button(
                            label="导出该弱信号证据包JSON",
                            data=json.dumps(signal_package, ensure_ascii=False, indent=2).encode('utf-8'),
                            file_name=f"{safe_key(signal_id)}_evidence_package.json",
                            mime="application/json",
                            key=f"download_json_{safe_key(signal_id)}_{idx}",
                        )
                    with download_cols[1]:
                        st.download_button(
                            label="导出该弱信号证据CSV",
                            data=pd.DataFrame(evidence_records).to_csv(index=False).encode('utf-8-sig') if evidence_records else b"",
                            file_name=f"{safe_key(signal_id)}_evidence.csv",
                            mime="text/csv",
                            key=f"download_csv_{safe_key(signal_id)}_{idx}",
                            disabled=not evidence_records,
                        )

                    show_evidence = st.checkbox(
                        f"展示关联证据（{len(evidence_records)} 条）",
                        key=f"show_evidence_{safe_key(signal_id)}_{idx}",
                    )
                    if show_evidence:
                        if not evidence_records:
                            st.warning("当前弱信号尚未找到可展示的关联证据。")
                        else:
                            overview_rows = []
                            for evidence in evidence_records:
                                overview_rows.append({
                                    '证据ID': safe_text(evidence.get('evidence_id')),
                                    'source_id': safe_text(evidence.get('source_id')),
                                    '来源': safe_text(evidence.get('source_type')),
                                    '标题': safe_preview(evidence.get('title'), limit=90),
                                    '日期': safe_text(evidence.get('date')),
                                    '机构': safe_preview(evidence.get('org'), limit=60),
                                    '质量分': safe_text(evidence.get('event_quality_score')),
                                    '预见分': safe_text(evidence.get('foresight_relevance_score')),
                                    '片段分': safe_text(evidence.get('evidence_span_score')),
                                    '置信分': safe_text(evidence.get('confidence_score')),
                                    '追溯状态': safe_text(evidence.get('traceability_status')) or 'partial_traceable',
                                    '支撑词': '、'.join(
                                        (
                                            parse_display_list(evidence.get('support_terms'))
                                            or parse_display_list(evidence.get('matched_terms'))
                                        )[:5]
                                    ),
                                })
                            st.markdown("**证据概述**")
                            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

                            for evidence_index, evidence in enumerate(evidence_records, 1):
                                evidence_id = safe_text(evidence.get('evidence_id')) or f"EV{evidence_index:03d}"
                                title = safe_text(evidence.get('title')) or "未命名来源"
                                st.markdown(f"**{evidence_index}. {title}**")
                                meta_parts = [
                                    safe_text(evidence.get('source_type')),
                                    safe_text(evidence.get('date')),
                                    safe_text(evidence.get('org')),
                                    f"质量分 {safe_text(evidence.get('event_quality_score'))}" if safe_text(evidence.get('event_quality_score')) else "",
                                    f"预见分 {safe_text(evidence.get('foresight_relevance_score'))}" if safe_text(evidence.get('foresight_relevance_score')) else "",
                                    safe_text(evidence.get('traceability_status')) or "partial_traceable",
                                ]
                                st.caption(" / ".join([part for part in meta_parts if part]))
                                url = safe_text(evidence.get('url'))
                                if url:
                                    st.markdown(f"[打开来源链接]({url})")
                                overview = safe_text(evidence.get('overview')) or safe_preview(evidence.get('evidence_excerpt'), limit=420)
                                if overview:
                                    st.write(overview)
                                support_terms = (
                                    parse_display_list(evidence.get('support_terms'))
                                    or parse_display_list(evidence.get('matched_terms'))
                                )
                                if support_terms:
                                    st.caption(f"支撑词: {'、'.join(support_terms[:8])}")
                                detail_key = f"detail_{safe_key(signal_id)}_{safe_key(evidence_id)}_{evidence_index}"
                                if st.checkbox("查看源文本详情", key=detail_key):
                                    has_detail = any(
                                        safe_text(evidence.get(field))
                                        for field in [
                                            'title',
                                            'authors',
                                            'affiliations',
                                            'org',
                                            'publish_time',
                                            'date',
                                            'keywords',
                                            'abstract',
                                            'main_content',
                                            'full_text',
                                            'url',
                                        ]
                                    )
                                    if has_detail:
                                        render_source_detail(evidence, detail_key, row)
                                    else:
                                        st.info("该历史证据只保留了概述或截断片段，未保存完整源文本。")
        else:
            st.info("未识别到弱信号")

    with tabs[6]:
        st.markdown("### ⏱ 弱信号持续跟踪与趋势观测")

        # 1. 弱信号趋势比对
        weak_signals_comparison_df = results.get('weak_signals_comparison_df', pd.DataFrame())
        if isinstance(weak_signals_comparison_df, pd.DataFrame) and not weak_signals_comparison_df.empty:
            st.markdown("#### 📈 弱信号变化趋势")
            col1, col2, col3, col4 = st.columns(4)
            total_ws = len(weak_signals_comparison_df)
            new_ws = int((weak_signals_comparison_df['trend'] == '新增').sum())
            growing_ws = int((weak_signals_comparison_df['trend'] == '增长').sum())
            declining_ws = int((weak_signals_comparison_df['trend'] == '减弱').sum())

            col1.metric("监测中弱信号总数", f"{total_ws} 个")
            col2.metric("新增弱信号 🆕", f"{new_ws} 个")
            col3.metric("提及增加 ↗️", f"{growing_ws} 个")
            col4.metric("提及减少 ↘️", f"{declining_ws} 个")

            comparison_cols = [
                'signal_id',
                'candidate_name',
                'trend_symbol',
                'current_mentions',
                'past_mentions',
                'temporal_momentum_score',
                'monitoring_priority',
                'monitoring_action',
                'temporal_validation_reason'
            ]
            comparison_cols = [c for c in comparison_cols if c in weak_signals_comparison_df.columns]

            render_paginated_dataframe(
                weak_signals_comparison_df.copy(),
                comparison_cols,
                "weak_signals_comparison_table",
                page_size=20,
                empty_message="暂无弱信号趋势比对数据。",
            )
            st.markdown("---")
        else:
            st.info("💡 当前运行结果或历史数据中未包含弱信号持续观测比对记录（需运行多次产生多次弱信号归档方可比对，或当前运行未识别到弱信号）。")
            st.markdown("---")

        # 2. 详细时间验证数据
        st.markdown("#### ⏱ 时间验证与窗口观测指标")
        temporal_validation_df = results.get('temporal_validation_df', pd.DataFrame())
        if isinstance(temporal_validation_df, pd.DataFrame) and not temporal_validation_df.empty:
            temporal_view_source = temporal_validation_df.copy()
            temporal_dedupe_cols = [
                column
                for column in ['candidate_id', 'candidate_name', 'temporal_validation_tier']
                if column in temporal_view_source.columns
            ]
            if temporal_dedupe_cols:
                temporal_view_source = temporal_view_source.drop_duplicates(subset=temporal_dedupe_cols, keep='first')
            if len(temporal_view_source) < len(temporal_validation_df):
                st.caption(f"已合并重复时间验证记录：展示 {len(temporal_view_source)} 条唯一弱信号，原始记录 {len(temporal_validation_df)} 条。")
            temporal_cols = [
                'candidate_name',
                'first_seen_date',
                'last_seen_date',
                'temporal_validation_tier',
                'temporal_validation_status',
                'temporal_validation_passed',
                'temporal_momentum_score',
                'growth_rate',
                'source_growth_rate',
                'org_growth_rate',
                'high_quality_growth_rate',
                'date_coverage_ratio',
                'monitoring_priority',
                'monitoring_action',
                'next_observation_window_start',
                'next_observation_window_end',
                'temporal_validation_reason',
            ]
            temporal_cols = [c for c in temporal_cols if c in temporal_view_source.columns]
            temporal_view = sort_by_available_scores(
                temporal_view_source,
                ['temporal_momentum_score', 'growth_rate', 'source_growth_rate', 'org_growth_rate']
            )
            render_paginated_dataframe(
                temporal_view,
                temporal_cols,
                "temporal_validation_table",
                page_size=20,
                empty_message="当前结果未包含弱信号时间验证。",
            )
            if 'temporal_validation_tier' in temporal_view_source.columns:
                st.markdown("##### 时间验证层级分布")
                render_horizontal_bar_chart(temporal_view_source['temporal_validation_tier'].value_counts(), "时间验证层级", "数量")
        else:
            st.info("当前结果未包含弱信号时间验证。")

        # 3. 弱信号时间验证候选字段
        validated_df = results.get('validated_df', pd.DataFrame())
        if isinstance(validated_df, pd.DataFrame) and not validated_df.empty:
            temporal_enriched_cols = [
                'display_candidate_name',
                'weak_signal_score',
                'temporal_validation_tier',
                'temporal_momentum_score',
                'date_coverage_ratio',
                'monitoring_priority',
                'monitoring_action',
                'temporal_validation_reason',
            ]
            temporal_enriched_cols = [c for c in temporal_enriched_cols if c in validated_df.columns]

            # 仅保留弱信号
            is_weak = pd.Series(False, index=validated_df.index)
            if "signal_type" in validated_df.columns:
                is_weak = is_weak | (validated_df["signal_type"] == "weak_signal")
            if "candidate_stage" in validated_df.columns:
                is_weak = is_weak | (validated_df["candidate_stage"] == "formed_candidate_strong")
            validated_df_weak = validated_df[is_weak].copy()

            if temporal_enriched_cols and not validated_df_weak.empty:
                st.markdown("##### 弱信号时间验证候选字段详情")
                temporal_candidates = sort_by_available_scores(
                    validated_df_weak,
                    ['temporal_momentum_score', 'weak_signal_score']
                )
                render_paginated_dataframe(
                    temporal_candidates,
                    temporal_enriched_cols,
                    "temporal_candidates_table",
                    page_size=20,
                    empty_message="当前结果未包含候选时间验证字段。",
                )

    with tabs[7]:
        st.markdown("### 📝 分析报告")
        report = results['report']
        if report:
            st.markdown(report)

            timestamp = os.path.basename(results['result_dir'])
            st.download_button(
                label="📥 下载报告",
                data=report,
                file_name=f"report_{timestamp}.txt",
                mime="text/plain"
            )

            result_dir = Path(results['result_dir'])
            family_report_path = result_dir / "family_evaluation_report.md"
            reverse_validation_path = result_dir / "reverse_validation.json"
            evidence_packets_path = result_dir / "report_evidence_packets.json"
            grounded_facts_path = result_dir / "report_grounded_facts.json"
            temporal_validation_path = result_dir / "temporal_validation.json"

            if family_report_path.exists():
                with st.expander("对象族评估报告"):
                    st.markdown(family_report_path.read_text(encoding='utf-8'))

            if reverse_validation_path.exists():
                with st.expander("反向验证明细"):
                    st.json(json.loads(reverse_validation_path.read_text(encoding='utf-8')))

            if evidence_packets_path.exists():
                with st.expander("证据包"):
                    st.json(json.loads(evidence_packets_path.read_text(encoding='utf-8')))

            if grounded_facts_path.exists():
                with st.expander("Grounded Facts"):
                    st.json(json.loads(grounded_facts_path.read_text(encoding='utf-8')))

            if temporal_validation_path.exists():
                with st.expander("时间验证与持续观测"):
                    st.json(json.loads(temporal_validation_path.read_text(encoding='utf-8')))

def _load_history_dataframe(folder_path: Path, stem: str) -> pd.DataFrame:
    json_path = folder_path / f"{stem}.json"
    csv_path = folder_path / f"{stem}.csv"

    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
        except Exception:
            pass

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _build_history_results(folder_path: Path) -> Dict[str, Any]:
    events_df = _load_history_dataframe(folder_path, "events")
    event_quality_df = _load_history_dataframe(folder_path, "event_quality")
    temporal_validation_df = _load_history_dataframe(folder_path, "temporal_validation")
    candidate_forms_df = _load_history_dataframe(folder_path, "candidate_forms")
    scored_df = _load_history_dataframe(folder_path, "scored")
    refined_df = _load_history_dataframe(folder_path, "refined")
    validated_df = _load_history_dataframe(folder_path, "validated")
    signals_df = _load_history_dataframe(folder_path, "signals")
    reverse_validation_df = _load_history_dataframe(folder_path, "reverse_validation")
    family_metrics_df = _load_history_dataframe(folder_path, "family_evaluation")
    source_documents_df = _load_history_dataframe(folder_path, "source_documents")
    signal_evidence_links_df = _load_history_dataframe(folder_path, "signal_evidence_links")
    signal_reliability_df = _load_history_dataframe(folder_path, "signal_reliability")
    weak_signals_comparison_df = _load_history_dataframe(folder_path, "weak_signals_comparison")

    report_path = folder_path / "report.txt"
    report = report_path.read_text(encoding='utf-8') if report_path.exists() else ""

    raw_data = events_df.copy()
    if raw_data.empty:
        raw_data = candidate_forms_df.copy()
    if raw_data.empty:
        raw_data = pd.DataFrame(columns=["source_type"])

    domain_context = None
    domain_pack_path = folder_path / "domain_pack.yaml"
    if domain_pack_path.exists():
        try:
            domain_context = DomainContext.from_pack(load_domain_pack(domain_pack_path))
        except Exception:
            domain_context = None

    near_strong_df = pd.DataFrame()
    if not signals_df.empty and "signal_type" in signals_df.columns:
        near_strong_df = signals_df[signals_df["signal_type"].isin(["near_strong", "hotspot"])].copy()

    return {
        'domain_context': domain_context,
        'events_df': events_df,
        'event_quality_df': event_quality_df,
        'temporal_validation_df': temporal_validation_df,
        'candidate_forms_df': candidate_forms_df,
        'scored_df': scored_df,
        'refined_df': refined_df,
        'reverse_validation_df': reverse_validation_df,
        'validated_df': validated_df,
        'signals_df': signals_df,
        'near_strong_df': near_strong_df,
        'family_metrics_df': family_metrics_df,
        'source_documents_df': source_documents_df,
        'signal_evidence_links_df': signal_evidence_links_df,
        'signal_reliability_df': signal_reliability_df,
        'weak_signals_comparison_df': weak_signals_comparison_df,
        'report': report,
        'result_dir': str(folder_path),
        'raw_data': raw_data,
    }


def render_history_page():
    """渲染历史结果页面"""
    st.markdown("## 📚 历史分析结果")

    result_dirs = sorted([item for item in Config.RESULT_DIR.iterdir() if item.is_dir()], key=lambda x: x.name, reverse=True)

    if not result_dirs:
        st.info("暂无历史结果")
        return

    selected = st.selectbox(
        "选择历史记录",
        options=[d.name for d in result_dirs],
        format_func=lambda x: f"📁 {x}"
    )

    if selected:
        folder_path = Config.RESULT_DIR / selected
        files = list(folder_path.glob("*"))

        st.markdown(f"**时间戳**: `{selected}`")
        st.markdown(f"**文件数量**: {len(files)} 个")

        history_results = _build_history_results(folder_path)
        if history_results.get('report') or not history_results.get('signals_df', pd.DataFrame()).empty:
            render_results(history_results)
        else:
            st.warning("该历史目录缺少可重建的分析结果文件。")

        with st.expander("原始产物文件"):
            for f in sorted(files):
                st.markdown(f"- `{f.name}`")

def main():
    """主函数"""
    init_analysis_state()

    page = st.sidebar.radio(
        "选择页面",
        options=["🔍 实时分析", "📚 历史结果"],
        index=0
    )

    if page == "🔍 实时分析":
        render_sidebar()
        render_header()

        if st.session_state.analysis_running:
            st.markdown("## 运行状态")
            render_live_analysis_panel()
            st.stop()

        st.markdown("## 分析来源")
        analysis_source = st.radio(
            "选择本次分析从哪里开始",
            options=[
                "数据库检索：完整流程",
                "原始数据 / CSV：完整流程",
                "已抽取事件：补齐质量评分后继续分析",
                "历史结果：仅重新生成报告",
            ],
            horizontal=False,
            disabled=st.session_state.analysis_running,
        )

        result_dirs = sorted(
            [item for item in Config.RESULT_DIR.iterdir() if item.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )
        events_result_dirs = [item for item in result_dirs if (item / "events.json").exists()]
        report_result_dirs = [item for item in result_dirs if (item / "signals.json").exists() or (item / "signals.csv").exists()]

        source_counts = {}
        custom_data_path = None
        custom_sample_size = None
        resume_mode = None
        resume_path = None
        custom_source_config = None
        domain_pack_ready = True
        selected_db_total = 0

        if analysis_source.startswith("数据库检索"):
            st.markdown("### 🔍 数据库检索条件配置")

            template_options = list(DOMAIN_SEARCH_TEMPLATES.keys())
            selected_template = st.selectbox(
                "技术领域检索模板",
                options=template_options,
                index=0,
                key="selected_domain_template",
            )
            template = DOMAIN_SEARCH_TEMPLATES[selected_template]
            default_id = template["field_id"]
            default_name = template["field_name"]
            default_keywords = template["keywords"]
            default_synonyms = template["synonyms"]
            default_exclude = template["exclude_terms"]

            col_id, col_name = st.columns(2)
            with col_id:
                tech_field_id = st.text_input("技术领域 ID", value=default_id)
            with col_name:
                tech_field_name = st.text_input("技术领域名称", value=default_name)

            keywords_input = st.text_area("检索词 (英文逗号分隔)", value=default_keywords)
            synonyms_input = st.text_area("同义词 / 扩展词 (英文逗号分隔)", value=default_synonyms)
            exclude_input = st.text_area("排除词 (英文逗号分隔)", value=default_exclude)

            col_start, col_end = st.columns(2)
            with col_start:
                start_date_val = st.date_input("开始日期", datetime(2023, 1, 1))
            with col_end:
                end_date_val = st.date_input("结束日期", datetime.now())

            # 转换日期为 YYYY-MM-DD
            start_date_str = start_date_val.strftime("%Y-%m-%d")
            end_date_str = end_date_val.strftime("%Y-%m-%d")

            keywords_list = split_csv_terms(keywords_input)
            synonyms_list = split_csv_terms(synonyms_input)
            exclude_list = split_csv_terms(exclude_input)

            # 获取或初始化数据库可用数量
            if "db_counts" not in st.session_state:
                st.session_state.db_counts = {"paper": 1000, "news": 1000, "policy": 1000, "report": 1000, "patent": 1000}

            # 构建查询参数用于获取统计或进行分析
            temp_query = SourceQuery(
                tech_field_id=tech_field_id,
                tech_field_name=tech_field_name,
                keywords=keywords_list,
                synonyms=synonyms_list,
                exclude_terms=exclude_list,
                start_date=start_date_str,
                end_date=end_date_str,
                source_types=["paper", "news", "policy", "report", "patent"],
                counts={}
            )

            st.markdown("#### 📊 可用总量统计")
            if st.button("📊 检索并获取实时数据库可用总量"):
                with st.spinner("正在向数据库发送 count 检索..."):
                    try:
                        repo = DataRepository.from_env()
                        st.session_state.db_counts = repo.get_source_counts(temp_query)
                        st.success("可用数据量统计更新成功！")
                    except Exception as ex:
                        st.error(f"查询数据库数量失败: {ex}")

            render_domain_pack_controls(
                tech_field_id,
                tech_field_name,
                keywords_list,
                synonyms_list,
                exclude_list,
            )
            selected_counts, domain_pack_ready = render_domain_pack_review_panel(st.session_state.db_counts)
            selected_db_total = sum(selected_counts.values())

            col_sort, col_topic = st.columns(2)
            with col_sort:
                sort_ui = st.selectbox("数据排序策略", ["相关性优先 (Relevance)", "时间优先 (Date)", "平衡轮转 (Balanced)"])
                sort_map = {
                    "相关性优先 (Relevance)": "relevance",
                    "时间优先 (Date)": "date",
                    "平衡轮转 (Balanced)": "balanced"
                }
                sort_mode = sort_map[sort_ui]
            with col_topic:
                use_topic_index = st.checkbox("启用话题词库自动扩展", value=True)

            source_query = SourceQuery(
                tech_field_id=tech_field_id,
                tech_field_name=tech_field_name,
                keywords=keywords_list,
                synonyms=synonyms_list,
                exclude_terms=exclude_list,
                start_date=start_date_str,
                end_date=end_date_str,
                source_types=[k for k, v in selected_counts.items() if v > 0],
                raw_source_types=DB_RAW_SOURCE_TYPES,
                counts=selected_counts,
                use_topic_index=use_topic_index,
                sort=sort_mode
            )

            custom_source_config = {
                "backend": "db",
                "query": source_query,
                "domain_context": DomainContext.from_pack(st.session_state.domain_pack_current) if domain_pack_ready else None,
            }
        elif analysis_source.startswith("原始数据"):
            sources = get_available_data_sources()
            source_counts, total = render_data_source_selection(sources)

            st.markdown("## 直接导入 CSV")
            uploaded_file = st.file_uploader("上传 CSV 数据文件", type=["csv"])
            if uploaded_file is not None:
                upload_dir = Config.MEMORY_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", uploaded_file.name).strip("_") or "uploaded.csv"
                upload_path = upload_dir / safe_name
                upload_path.write_bytes(uploaded_file.getbuffer())
                try:
                    uploaded_preview = pd.read_csv(upload_path, encoding='utf-8-sig')
                    st.caption(f"已导入 {len(uploaded_preview)} 行，后续分析将优先使用该 CSV。")
                    preview_cols = [c for c in ['id', 'source_type', 'title', 'text', 'date', 'org'] if c in uploaded_preview.columns]
                    if preview_cols:
                        st.dataframe(uploaded_preview[preview_cols].head(5), use_container_width=True)
                    custom_sample_size = st.number_input(
                        "CSV 采样数量",
                        min_value=1,
                        max_value=max(1, len(uploaded_preview)),
                        value=min(100, max(1, len(uploaded_preview))),
                        step=10,
                    )
                    custom_data_path = str(upload_path)
                except Exception as exc:
                    st.error(f"CSV 读取失败: {exc}")
        elif analysis_source.startswith("已抽取事件"):
            resume_mode = "events"
            event_source_mode = st.radio(
                "事件来源",
                options=["选择历史结果目录", "上传 events 文件"],
                horizontal=True,
                disabled=st.session_state.analysis_running,
            )
            if event_source_mode == "选择历史结果目录":
                if events_result_dirs:
                    selected_events_dir = st.selectbox(
                        "选择包含 events.json 的历史目录",
                        options=[item.name for item in events_result_dirs],
                        format_func=lambda name: f"📁 {name}",
                    )
                    resume_path = str(Config.RESULT_DIR / selected_events_dir / "events.json")
                    st.caption(f"将从 `{resume_path}` 补齐事件质量评分，并继续候选成形、评分、验证、时间验证、信号生成和报告生成。")
                else:
                    st.warning("result 目录下暂时没有可用的 events.json。")
            else:
                uploaded_events = st.file_uploader("上传 events.json 或 events.csv", type=["json", "csv"])
                if uploaded_events is not None:
                    upload_dir = Config.MEMORY_DIR / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", uploaded_events.name).strip("_") or "events.json"
                    upload_path = upload_dir / safe_name
                    upload_path.write_bytes(uploaded_events.getbuffer())
                    resume_path = str(upload_path)
                    try:
                        preview_df = pd.read_json(upload_path) if upload_path.suffix.lower() == ".json" else pd.read_csv(upload_path, encoding="utf-8-sig")
                        st.caption(f"已导入 {len(preview_df)} 条事件。")
                        st.dataframe(preview_df.head(5), use_container_width=True)
                    except Exception as exc:
                        st.error(f"事件文件读取失败: {exc}")
        else:
            resume_mode = "report"
            if report_result_dirs:
                selected_report_dir = st.selectbox(
                    "选择包含 signals 的历史目录",
                    options=[item.name for item in report_result_dirs],
                    format_func=lambda name: f"📁 {name}",
                )
                resume_path = str(Config.RESULT_DIR / selected_report_dir)
                st.caption("将复用该目录中的 signals 文件，只重新生成报告和报告证据包。")
            else:
                st.warning("result 目录下暂时没有可用于重生成报告的 signals 文件。")

        st.markdown("## 🚀 开始分析")
        start_disabled = st.session_state.analysis_running or (resume_mode in {"events", "report"} and not resume_path)
        if analysis_source.startswith("数据库检索"):
            start_disabled = start_disabled or (not domain_pack_ready) or selected_db_total <= 0

        col1, col2, col3 = st.columns(3)
        with col1:
            use_cache = st.checkbox("使用缓存", value=True)
        with col2:
            use_llm_strategy = st.checkbox("使用LLM策略", value=False)
        with col3:
            st.write("")
            start_button = st.button(
                "🚀 开始分析",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
            )

        if start_button:
            s_config = None
            if analysis_source.startswith("数据库检索"):
                s_config = custom_source_config
            start_analysis_task(
                source_counts,
                use_cache,
                use_llm_strategy,
                data_path=custom_data_path,
                sample_size=custom_sample_size,
                resume_mode=resume_mode,
                resume_path=resume_path,
                source_config=s_config,
            )
        render_live_analysis_panel()

        # 在 fragment 外部渲染结果，确保 tabs 内的分页/checkbox 等交互触发完整 rerun
        if not st.session_state.analysis_running and st.session_state.analysis_results:
            render_results(st.session_state.analysis_results)

    else:
        render_history_page()

if __name__ == "__main__":
    main()
