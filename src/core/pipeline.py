"""分析流水线 - 整合所有模块"""

import json
import re
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# 导入各模块
from ..extraction.event_extractor import process_events, load_event_cache, save_event_cache
from ..extraction.candidate_former import build_candidate_forms

from ..scoring.scorer import score_all_candidates, refresh_research_layers
from ..scoring.signal_generator import generate_candidate_outputs
from ..scoring.topic_refiner import refine_research_scored_candidates
from ..scoring.key_core_scorer import score_key_core_candidates, build_key_core_candidate_table

from ..validation.reverse_validator import build_reverse_validation_table, merge_reverse_validation_into_manual_review
from ..validation.object_family_canonicalizer import ObjectFamilyCanonicalizer
from ..validation.family_evaluator import evaluate_families, generate_family_report
from ..validation.final_shortlist import build_final_shortlist
from ..validation.baseline_compare import build_frequency_baseline_table, build_baseline_comparison_table
from ..validation.event_quality import (
    build_event_quality_table,
    merge_event_quality_into_events,
    merge_event_quality_into_raw_data,
    merge_event_quality_into_candidates,
)
from ..validation.tech_chain_mapper import (
    load_tech_chain_data,
    build_tech_chain_mapping_table,
    merge_tech_chain_mapping_into_candidates,
)
from ..validation.temporal_validator import (
    build_temporal_validation_table,
    merge_temporal_validation_into_candidates,
)

from ..utils.config import Config
from ..utils.llm_client import get_provider_and_client, chat_text
from ..utils.api_stats import record_call
from ..utils.env_config import ensure_env_loaded
import os

ensure_env_loaded()


class AnalysisPipeline:
    """分析流水线"""

    def __init__(self):
        self.canonicalizer = ObjectFamilyCanonicalizer()
        self.results = {}
        self.report_artifacts: Dict[str, Any] = {}
        self.latest_event_quality_df = pd.DataFrame()
        self.latest_tech_chain_mapping_df = pd.DataFrame()
        self.latest_temporal_validation_df = pd.DataFrame()
        self.latest_key_core_scored_df = pd.DataFrame()
        self.latest_key_core_candidates_df = pd.DataFrame()
        self.latest_reverse_validation_df = pd.DataFrame()
        self.latest_family_metrics_df = pd.DataFrame()
        self.latest_final_shortlist_df = pd.DataFrame()
        self.latest_shortlist_dedup_df = pd.DataFrame()
        self.latest_frequency_baseline_df = pd.DataFrame()
        self.latest_baseline_comparison_df = pd.DataFrame()

    def run_full_pipeline(
        self,
        data_path: Optional[Path] = None,
        sample_size: int = 100,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        source_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行完整分析流水线

        Args:
            data_path: 数据文件路径
            sample_size: 采样数量
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            source_config: 数据源配置

        Returns:
            分析结果字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Config.RESULT_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{timestamp}] 开始分析流水线...")

        # 阶段1: 数据加载
        print("\n[阶段1] 数据加载...")
        raw_data = self._load_data(data_path, sample_size, source_config)
        if raw_data.empty:
            print("[ERROR] 数据加载失败")
            return {}
        print(f"  加载了 {len(raw_data)} 条数据")

        # 阶段2: 事件抽取
        print("\n[阶段2] 事件抽取...")
        events_df = self._extract_events(raw_data, use_cache, cache_dir, data_path=data_path)
        if events_df is None or events_df.empty:
            print("[ERROR] 事件抽取失败")
            return {}
        print(f"  抽取了 {len(events_df)} 个事件")

        # 阶段2.5: 事件质量评分
        print("\n[阶段2.5] 事件质量评分...")
        event_quality_df = self._score_event_quality(events_df, raw_data)
        events_df = merge_event_quality_into_events(events_df, event_quality_df)
        raw_data = merge_event_quality_into_raw_data(raw_data, event_quality_df)
        print(f"  事件质量评分完成，共 {len(event_quality_df)} 条")

        # 阶段3: 候选成形
        print("\n[阶段3] 候选对象成形...")
        candidate_forms_df = self._form_candidates(events_df, raw_data)
        candidate_forms_df = self._apply_candidate_event_quality(candidate_forms_df, event_quality_df)
        candidate_count_before_dedupe = len(candidate_forms_df)
        candidate_forms_df = self._dedupe_candidate_flow(candidate_forms_df)
        print(f"  生成了 {candidate_count_before_dedupe} 个候选对象，流转去重后 {len(candidate_forms_df)} 个")

        # 阶段4: 评分
        print("\n[阶段4] 弱信号评分...")
        scored_df = self._score_candidates(candidate_forms_df)
        print(f"  评分完成，共 {len(scored_df)} 个候选")

        # 阶段5: 主题细化
        print("\n[阶段5] 主题细化...")
        refined_df = self._refine_topics(scored_df)
        print(f"  主题细化完成")

        # 阶段6: 反向验证
        print("\n[阶段6] 反向验证...")
        validated_df = self._validate_reverse(refined_df, candidate_forms_df)
        print(f"  反向验证完成")

        # 阶段6.5: 技术链映射
        print("\n[阶段6.5] 技术链映射...")
        validated_df = self._map_tech_chain(validated_df)
        print(f"  技术链映射完成，共 {len(self.latest_tech_chain_mapping_df)} 条")

        # 阶段6.6: 时间验证
        print("\n[阶段6.6] 时间验证...")
        temporal_validation_df = self._validate_temporal(validated_df, events_df, raw_data)
        validated_df = merge_temporal_validation_into_candidates(validated_df, temporal_validation_df)
        print(f"  时间验证完成，共 {len(temporal_validation_df)} 条")

        # 阶段6.7: 关键核心潜力评分
        print("\n[阶段6.7] 关键核心潜力评分...")
        validated_df = self._score_key_core_potential(validated_df, temporal_validation_df)
        self._build_research_validation_artifacts(validated_df)
        print(f"  关键核心潜力评分完成，共 {len(self.latest_key_core_scored_df)} 条")

        # 阶段7: 信号生成
        print("\n[阶段7] 信号生成...")
        signals_output = self._generate_signals(validated_df, raw_data)
        if isinstance(signals_output, dict):
            signals_df = signals_output.get("candidates_df", pd.DataFrame())
        elif isinstance(signals_output, pd.DataFrame):
            signals_df = signals_output
        else:
            signals_df = pd.DataFrame(signals_output if signals_output is not None else [])
        print(f"  生成了 {len(signals_df)} 个信号")

        # 阶段8: 报告生成
        print("\n[阶段8] 报告生成...")
        report = self._generate_report(signals_output)
        print(f"  报告生成完成")

        # 保存结果
        print("\n[保存结果] ...")
        self._save_results(result_dir, events_df, candidate_forms_df, scored_df, refined_df, validated_df, signals_output, report)

        print(f"\n[完成] 分析结果已保存到: {result_dir}")

        return {
            "result_dir": result_dir,
            "events_df": events_df,
            "event_quality_df": self.latest_event_quality_df,
            "candidate_forms_df": candidate_forms_df,
            "scored_df": scored_df,
            "refined_df": refined_df,
            "reverse_validation_df": self.latest_reverse_validation_df,
            "tech_chain_mapping_df": self.latest_tech_chain_mapping_df,
            "temporal_validation_df": self.latest_temporal_validation_df,
            "key_core_scored_df": self.latest_key_core_scored_df,
            "key_core_candidates_df": self.latest_key_core_candidates_df,
            "validated_df": validated_df,
            "signals_df": signals_df,
            "signals_output": signals_output,
            "family_metrics_df": self.latest_family_metrics_df,
            "final_shortlist_df": self.latest_final_shortlist_df,
            "frequency_baseline_df": self.latest_frequency_baseline_df,
            "baseline_comparison_df": self.latest_baseline_comparison_df,
            "report": report,
        }

    @staticmethod
    def _ensure_dataframe(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            if "candidates_df" in data and isinstance(data.get("candidates_df"), pd.DataFrame):
                return data.get("candidates_df").copy()
            return pd.DataFrame([data])
        return pd.DataFrame(data if data is not None else [])

    @staticmethod
    def _read_dataframe_file(path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            return pd.DataFrame()
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, encoding="utf-8-sig")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
        return pd.DataFrame()

    def _load_result_dataframe(self, result_dir: Path, stem: str) -> pd.DataFrame:
        result_dir = Path(result_dir)
        for suffix in [".json", ".csv", ".xlsx", ".xls"]:
            path = result_dir / f"{stem}{suffix}"
            if path.exists():
                try:
                    return self._read_dataframe_file(path)
                except Exception:
                    continue
        return pd.DataFrame()

    def _raw_data_from_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        events_df = self._ensure_dataframe(events_df)
        if events_df.empty:
            return pd.DataFrame(columns=["id", "source_type", "title", "text"])

        def event_text(row: pd.Series) -> str:
            technologies = row.get("technology", [])
            if isinstance(technologies, str):
                tech_text = technologies
            elif isinstance(technologies, list):
                tech_text = "、".join(str(item) for item in technologies if str(item).strip())
            else:
                tech_text = ""
            parts = [
                row.get("title", ""),
                row.get("subject", ""),
                row.get("action", ""),
                tech_text,
                row.get("scene", ""),
                row.get("time", ""),
            ]
            return " ".join(self._safe_report_text(part) for part in parts if self._safe_report_text(part)).strip()

        raw_data = pd.DataFrame()
        raw_data["id"] = events_df.get("id", pd.Series([f"event_{i}" for i in range(len(events_df))])).fillna("").astype(str)
        raw_data["source_type"] = events_df.get("source_type", pd.Series(["unknown"] * len(events_df))).fillna("unknown").astype(str)
        raw_data["title"] = events_df.get("title", pd.Series([""] * len(events_df))).fillna("").astype(str)
        raw_data["text"] = events_df.apply(event_text, axis=1)
        for column in ["org", "date", "url"]:
            if column in events_df.columns:
                raw_data[column] = events_df[column]
        raw_data = raw_data.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
        raw_data["text"] = raw_data["text"].where(raw_data["text"].str.strip() != "", raw_data["title"])
        return raw_data

    def _signals_output_from_signals_df(self, signals_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        signals_df = self._ensure_dataframe(signals_df)
        near_strong_df = pd.DataFrame()
        if not signals_df.empty and "signal_type" in signals_df.columns:
            near_strong_df = signals_df[signals_df["signal_type"].isin(["near_strong", "hotspot"])].copy()
        return {
            "candidates_df": signals_df,
            "near_strong_candidates_df": near_strong_df,
        }

    def run_from_events(
        self,
        events_path: Optional[Path] = None,
        events_df: Optional[pd.DataFrame] = None,
        raw_data_path: Optional[Path] = None,
        result_name_suffix: str = "from_events",
    ) -> Dict[str, Any]:
        """Continue analysis from already extracted events, skipping raw loading and extraction."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Config.RESULT_DIR / f"{timestamp}_{result_name_suffix}"
        result_dir.mkdir(parents=True, exist_ok=True)

        events_df = self._ensure_dataframe(events_df) if events_df is not None else self._read_dataframe_file(Path(events_path))
        if events_df.empty:
            print("[ERROR] 已抽取事件为空，无法继续分析")
            return {}

        raw_data = self._read_dataframe_file(Path(raw_data_path)) if raw_data_path else pd.DataFrame()
        if raw_data.empty:
            raw_data = self._raw_data_from_events(events_df)

        print(f"[{timestamp}] 从已抽取事件继续分析...")
        print(f"  事件数: {len(events_df)}，原始上下文记录数: {len(raw_data)}")

        print("\n[阶段2.5] 事件质量评分...")
        event_quality_df = self._score_event_quality(events_df, raw_data)
        events_df = merge_event_quality_into_events(events_df, event_quality_df)
        raw_data = merge_event_quality_into_raw_data(raw_data, event_quality_df)
        print(f"  事件质量评分完成，共 {len(event_quality_df)} 条")

        print("\n[阶段3] 候选对象成形...")
        candidate_forms_df = self._form_candidates(events_df, raw_data)
        candidate_forms_df = self._apply_candidate_event_quality(candidate_forms_df, event_quality_df)
        candidate_count_before_dedupe = len(candidate_forms_df)
        candidate_forms_df = self._dedupe_candidate_flow(candidate_forms_df)
        print(f"  生成了 {candidate_count_before_dedupe} 个候选对象，流转去重后 {len(candidate_forms_df)} 个")

        print("\n[阶段4] 弱信号评分...")
        scored_df = self._score_candidates(candidate_forms_df)
        print(f"  评分完成，共 {len(scored_df)} 个候选")

        print("\n[阶段5] 主题细化...")
        refined_df = self._refine_topics(scored_df)
        print("  主题细化完成")

        print("\n[阶段6] 反向验证...")
        validated_df = self._validate_reverse(refined_df, candidate_forms_df)
        print("  反向验证完成")

        print("\n[阶段6.5] 技术链映射...")
        validated_df = self._map_tech_chain(validated_df)
        print(f"  技术链映射完成，共 {len(self.latest_tech_chain_mapping_df)} 条")

        print("\n[阶段6.6] 时间验证...")
        temporal_validation_df = self._validate_temporal(validated_df, events_df, raw_data)
        validated_df = merge_temporal_validation_into_candidates(validated_df, temporal_validation_df)
        print(f"  时间验证完成，共 {len(temporal_validation_df)} 条")

        print("\n[阶段6.7] 关键核心潜力评分...")
        validated_df = self._score_key_core_potential(validated_df, temporal_validation_df)
        self._build_research_validation_artifacts(validated_df)
        print(f"  关键核心潜力评分完成，共 {len(self.latest_key_core_scored_df)} 条")

        print("\n[阶段7] 信号生成...")
        signals_output = self._generate_signals(validated_df, raw_data)
        signals_df = self._ensure_dataframe(signals_output)
        print(f"  生成了 {len(signals_df)} 个信号")

        print("\n[阶段8] 报告生成...")
        report = self._generate_report(signals_output)
        print("  报告生成完成")

        print("\n[保存结果] ...")
        self._save_results(result_dir, events_df, candidate_forms_df, scored_df, refined_df, validated_df, signals_output, report)

        return {
            "result_dir": result_dir,
            "events_df": events_df,
            "event_quality_df": self.latest_event_quality_df,
            "candidate_forms_df": candidate_forms_df,
            "scored_df": scored_df,
            "refined_df": refined_df,
            "reverse_validation_df": self.latest_reverse_validation_df,
            "tech_chain_mapping_df": self.latest_tech_chain_mapping_df,
            "temporal_validation_df": self.latest_temporal_validation_df,
            "key_core_scored_df": self.latest_key_core_scored_df,
            "key_core_candidates_df": self.latest_key_core_candidates_df,
            "validated_df": validated_df,
            "signals_df": signals_df,
            "signals_output": signals_output,
            "family_metrics_df": self.latest_family_metrics_df,
            "final_shortlist_df": self.latest_final_shortlist_df,
            "frequency_baseline_df": self.latest_frequency_baseline_df,
            "baseline_comparison_df": self.latest_baseline_comparison_df,
            "report": report,
            "raw_data": raw_data,
        }

    def regenerate_report_from_result(
        self,
        result_dir: Path,
        result_name_suffix: str = "report_refresh",
    ) -> Dict[str, Any]:
        """Regenerate report from a saved result directory without rerunning extraction/scoring."""
        source_dir = Path(result_dir)
        signals_df = self._load_result_dataframe(source_dir, "signals")
        if signals_df.empty:
            print("[ERROR] 历史目录缺少 signals 文件，无法重生成报告")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Config.RESULT_DIR / f"{timestamp}_{result_name_suffix}"
        target_dir.mkdir(parents=True, exist_ok=True)

        events_df = self._load_result_dataframe(source_dir, "events")
        self.latest_event_quality_df = self._load_result_dataframe(source_dir, "event_quality")
        candidate_forms_df = self._load_result_dataframe(source_dir, "candidate_forms")
        scored_df = self._load_result_dataframe(source_dir, "scored")
        refined_df = self._load_result_dataframe(source_dir, "refined")
        validated_df = self._load_result_dataframe(source_dir, "validated")
        self.latest_tech_chain_mapping_df = self._load_result_dataframe(source_dir, "tech_chain_mapping")
        self.latest_temporal_validation_df = self._load_result_dataframe(source_dir, "temporal_validation")
        self.latest_key_core_scored_df = self._load_result_dataframe(source_dir, "key_core_scored")
        self.latest_key_core_candidates_df = self._load_result_dataframe(source_dir, "key_core_candidates")
        self.latest_reverse_validation_df = self._load_result_dataframe(source_dir, "reverse_validation")
        self.latest_family_metrics_df = self._load_result_dataframe(source_dir, "family_evaluation")
        self.latest_final_shortlist_df = self._load_result_dataframe(source_dir, "final_shortlist")
        self.latest_shortlist_dedup_df = self._load_result_dataframe(source_dir, "final_shortlist_dedup_map")
        self.latest_frequency_baseline_df = self._load_result_dataframe(source_dir, "frequency_baseline")
        self.latest_baseline_comparison_df = self._load_result_dataframe(source_dir, "baseline_comparison")

        if self.latest_tech_chain_mapping_df.empty and not signals_df.empty:
            tech_chain_data = load_tech_chain_data(Config.DATA_DIR / "tech_chain")
            self.latest_tech_chain_mapping_df = build_tech_chain_mapping_table(signals_df, tech_chain_data)

        if not self.latest_tech_chain_mapping_df.empty:
            signals_df = merge_tech_chain_mapping_into_candidates(signals_df, self.latest_tech_chain_mapping_df)
            if not validated_df.empty:
                validated_df = merge_tech_chain_mapping_into_candidates(validated_df, self.latest_tech_chain_mapping_df)

        raw_data = events_df.copy() if not events_df.empty else pd.DataFrame(columns=["source_type"])
        if self.latest_temporal_validation_df.empty and not signals_df.empty:
            self.latest_temporal_validation_df = build_temporal_validation_table(signals_df, events_df, raw_data)

        if not self.latest_temporal_validation_df.empty:
            signals_df = merge_temporal_validation_into_candidates(signals_df, self.latest_temporal_validation_df)
            if not validated_df.empty:
                validated_df = merge_temporal_validation_into_candidates(validated_df, self.latest_temporal_validation_df)

        if self.latest_key_core_scored_df.empty and not signals_df.empty:
            self.latest_key_core_scored_df = score_key_core_candidates(signals_df, self.latest_temporal_validation_df)
            self.latest_key_core_candidates_df = build_key_core_candidate_table(self.latest_key_core_scored_df, top_k=20)
            signals_df = self.latest_key_core_scored_df
        elif not self.latest_key_core_scored_df.empty and "key_core_score" not in signals_df.columns:
            key_core_columns = [
                "candidate_id",
                "key_core_score",
                "key_core_tier",
                "weak_signal_component",
                "growth_validation_component",
                "tech_chain_bottleneck_component",
                "strategic_importance_component",
                "evidence_confidence_component",
                "asset_support_component",
                "quality_gate_passed",
                "mapping_gate_passed",
                "temporal_gate_passed",
                "object_gate_passed",
                "key_core_gate_passed",
                "key_core_reason",
                "key_core_risk",
                "recommended_action",
                "top_evidence_ids",
            ]
            available_columns = [column for column in key_core_columns if column in self.latest_key_core_scored_df.columns]
            if "candidate_id" in signals_df.columns and "candidate_id" in available_columns:
                signals_df = signals_df.merge(
                    self.latest_key_core_scored_df[available_columns],
                    on="candidate_id",
                    how="left",
                )

        signals_output = self._signals_output_from_signals_df(signals_df)
        report = self._generate_report(signals_output)
        self._save_results(target_dir, events_df, candidate_forms_df, scored_df, refined_df, validated_df, signals_output, report)

        return {
            "result_dir": target_dir,
            "events_df": events_df,
            "event_quality_df": self.latest_event_quality_df,
            "candidate_forms_df": candidate_forms_df,
            "scored_df": scored_df,
            "refined_df": refined_df,
            "reverse_validation_df": self.latest_reverse_validation_df,
            "tech_chain_mapping_df": self.latest_tech_chain_mapping_df,
            "temporal_validation_df": self.latest_temporal_validation_df,
            "key_core_scored_df": self.latest_key_core_scored_df,
            "key_core_candidates_df": self.latest_key_core_candidates_df,
            "validated_df": validated_df,
            "signals_df": signals_df,
            "signals_output": signals_output,
            "near_strong_df": signals_output.get("near_strong_candidates_df", pd.DataFrame()),
            "family_metrics_df": self.latest_family_metrics_df,
            "final_shortlist_df": self.latest_final_shortlist_df,
            "frequency_baseline_df": self.latest_frequency_baseline_df,
            "baseline_comparison_df": self.latest_baseline_comparison_df,
            "report": report,
            "raw_data": raw_data,
        }

    @staticmethod
    def _slugify_cache_part(value: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_")
        return text[:80] or "dataset"

    @staticmethod
    def _data_cache_fingerprint(raw_data: pd.DataFrame, data_path: Optional[Path] = None) -> str:
        if data_path:
            path = Path(data_path)
            try:
                stat = path.stat()
                raw = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
            except OSError:
                raw = str(path)
            return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]

        ids = raw_data.get("id", pd.Series(dtype="object")).astype(str).tolist()
        raw = "|".join(ids[:500]) + f"::{len(raw_data)}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def _event_cache_path(
        self,
        raw_data: pd.DataFrame,
        cache_dir: Optional[Path],
        data_path: Optional[Path] = None,
    ) -> Optional[Path]:
        if cache_dir is None:
            return None
        dataset_name = Path(data_path).stem if data_path else "selected_sources"
        slug = self._slugify_cache_part(dataset_name)
        fingerprint = self._data_cache_fingerprint(raw_data, data_path)
        return cache_dir / "events" / f"{slug}__{fingerprint}.json"

    def _load_data(self, data_path: Optional[Path], sample_size: int, source_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """加载数据并标准化列名"""
        def standardize_df(df: pd.DataFrame, source_type: str = "未知") -> pd.DataFrame:
            """标准化DataFrame，确保有text、id、source_type列"""
            # 可能的文本列名
            text_cols = ['text', 'content', 'abstract', '标题', '摘要', '内容', 
                        'title', '专利名称', '发明名称', 'name', '专利标题', '名称',
                        'title_cn', 'abstract_first', 'summary', 'snippet', 'description']
            # 可能的ID列名
            id_cols = ['id', 'ID', '编号', '专利号', '申请号', '公开号', 'doc_id', 'source_id']
            
            result = pd.DataFrame()
            
            available_text_cols = [col for col in text_cols if col in df.columns]
            title_source_col = next(
                (col for col in ['title', '标题', '专利名称', '发明名称', 'title_cn', 'name', '名称'] if col in df.columns),
                None,
            )
            abstract_source_col = next(
                (col for col in ['abstract', '摘要', 'abstract_first', 'summary', 'snippet', 'description'] if col in df.columns),
                None,
            )

            if available_text_cols:
                text_frame = df[available_text_cols].fillna("").astype(str)
                result['text'] = text_frame.apply(
                    lambda row: next((str(value).strip() for value in row if str(value).strip()), ""),
                    axis=1,
                )
            else:
                # 如果没有找到文本列，使用第一列
                result['text'] = df.iloc[:, 0].fillna("").astype(str)

            if title_source_col and abstract_source_col:
                title_text = df[title_source_col].fillna("").astype(str)
                abstract_text = df[abstract_source_col].fillna("").astype(str)
                combined_text = (title_text + "\n" + abstract_text).str.strip()
                result['text'] = result['text'].where(result['text'].str.strip() != "", combined_text)
            
            # 查找ID列
            id_col = None
            for col in id_cols:
                if col in df.columns:
                    id_col = col
                    break
            
            if id_col:
                result['id'] = df[id_col].fillna("").astype(str)
            else:
                # 生成默认ID
                result['id'] = [f"{source_type}_{i}" for i in range(len(df))]
            
            # 添加来源类型
            source_type_col = next(
                (col for col in ['source_type', 'type', '来源类型', '数据源类型'] if col in df.columns),
                None,
            )
            if source_type_col:
                result['source_type'] = df[source_type_col].fillna(source_type).astype(str)
            else:
                result['source_type'] = source_type

            # 标准化标题
            title_cols = ['title', '标题', '专利名称', '发明名称', 'title_cn', 'name', '名称']
            for col in title_cols:
                if col in df.columns:
                    result['title'] = df[col]
                    break

            # 标准化日期
            date_cols = ['date', 'time', 'publish_date', 'public_date', 'apply_date', 'priority_date', 'year', 'crawl_time', 'create_time', 'update_time']
            for col in date_cols:
                if col in df.columns:
                    result['date'] = df[col]
                    break

            # 标准化机构/来源
            org_cols = ['org', 'organization', 'institution', 'source', 'author_org', 'applicant', '申请人', '机构']
            for col in org_cols:
                if col in df.columns:
                    result['org'] = df[col]
                    break

            # 标准化链接
            url_cols = ['url', 'link', 'pdf_url', 'url_source', 'doi/arxiv_id', 'doi', 'source_url']
            for col in url_cols:
                if col in df.columns:
                    result['url'] = df[col]
                    break

            for optional_col in ['raw_candidate_text', 'display_candidate_name', 'snippet']:
                if optional_col in df.columns and optional_col not in result.columns:
                    result[optional_col] = df[optional_col]

            result['text'] = result['text'].fillna("").astype(str)
            result = result[result['text'].str.strip() != ""].reset_index(drop=True)
             
            return result
        
        def load_single_file(file_path: Path) -> Optional[pd.DataFrame]:
            """加载单个文件"""
            try:
                if file_path.suffix == '.csv':
                    return pd.read_csv(file_path, encoding='utf-8-sig')
                elif file_path.suffix in ['.xlsx', '.xls']:
                    return pd.read_excel(file_path)
                elif file_path.suffix == '.json':
                    return pd.read_json(file_path)
            except Exception as e:
                print(f"  警告: 无法加载 {file_path}: {e}")
            return None
        
        def infer_source_type(filename: str) -> tuple:
            """根据文件名推断来源类型，返回 (中文名, 英文key)"""
            fname = filename.lower()
            if '专利' in fname or 'patent' in fname:
                return ('专利', 'patent')
            elif '文献' in fname or 'literature' in fname or 'paper' in fname:
                return ('文献', 'literature')
            elif '研报' in fname or 'report' in fname:
                return ('研报', 'report')
            elif '资讯' in fname or 'news' in fname:
                return ('资讯', 'news')
            return ('未知', 'unknown')
        
        # 加载指定文件
        if data_path and data_path.exists():
            df = load_single_file(data_path)
            if df is not None:
                source_type, _ = infer_source_type(data_path.name)
                standardized = standardize_df(df, source_type)
                return standardized.head(sample_size) if sample_size else standardized
            return pd.DataFrame()

        # 默认加载data目录下所有数据
        data_files = list(Config.DATA_DIR.glob("*.csv")) + list(Config.DATA_DIR.glob("*.xlsx")) + list(Config.DATA_DIR.glob("*.json"))
        if not data_files:
            print("  警告: data目录中没有找到数据文件")
            return pd.DataFrame()

        # 按数据源类型分组文件
        source_files = {
            '专利': [],
            '文献': [],
            '研报': [],
            '资讯': [],
            '未知': []
        }
        
        for f in data_files:
            source_type, _ = infer_source_type(f.name)
            source_files[source_type].append(f)
        
        # 确定每个数据源的采样数量
        source_counts = {}
        if source_config:
            counts = source_config.get('counts', {})
            sources = source_config.get('sources')
            
            if counts:
                # 使用指定的数量
                source_key_map = {
                    'patent': '专利',
                    'literature': '文献',
                    'report': '研报',
                    'news': '资讯'
                }
                for key, cnt in counts.items():
                    cn_name = source_key_map.get(key, key)
                    source_counts[cn_name] = cnt
            elif sources:
                # 指定了数据源类型，平均分配
                source_key_map = {
                    'patent': '专利',
                    'literature': '文献',
                    'report': '研报',
                    'news': '资讯'
                }
                per_source = sample_size // len(sources)
                for src in sources:
                    cn_name = source_key_map.get(src, src)
                    source_counts[cn_name] = per_source
        else:
            # 默认从所有数据源平均分配
            active_sources = [k for k, v in source_files.items() if v and k != '未知']
            if active_sources:
                per_source = sample_size // len(active_sources)
                for src in active_sources:
                    source_counts[src] = per_source
        
        # 加载数据
        dfs = []
        for source_type, count in source_counts.items():
            files = source_files.get(source_type, [])
            if not files:
                continue
            
            # 加载该类型的数据文件
            for f in files:
                df = load_single_file(f)
                if df is not None:
                    standardized = standardize_df(df, source_type)
                    # 采样指定数量
                    sampled = standardized.head(count) if count < len(standardized) else standardized
                    dfs.append(sampled)
                    print(f"  加载 {f.name}: {len(sampled)} 条 ({source_type})")
        
        if not dfs:
            # 如果没有按配置加载，回退到默认行为
            for f in data_files[:5]:
                df = load_single_file(f)
                if df is not None:
                    source_type, _ = infer_source_type(f.name)
                    standardized = standardize_df(df, source_type)
                    dfs.append(standardized)
                    print(f"  加载 {f.name}: {len(standardized)} 条 ({source_type})")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        return combined.head(sample_size)

    def _extract_events(
        self,
        raw_data: pd.DataFrame,
        use_cache: bool,
        cache_dir: Optional[Path],
        data_path: Optional[Path] = None,
    ) -> Optional[pd.DataFrame]:
        """抽取事件"""
        cache_path = self._event_cache_path(raw_data, cache_dir, data_path)

        # 检查缓存
        if use_cache and cache_path:
            cached = load_event_cache(cache_path, expected_ids=raw_data.get('id', []).tolist())
            if cached is not None:
                print("  使用缓存的事件数据")
                return cached

        # 执行事件抽取
        events_df = process_events(
            raw_data,
            use_api=True,
            cache_path=cache_path if use_cache else None,
            refresh_cache=not use_cache,
        )

        # 保存缓存
        if use_cache and cache_path:
            save_event_cache(
                cache_path,
                events_df,
                metadata={
                    "rows": len(events_df),
                    "data_path": str(data_path) if data_path else "",
                    "cache_fingerprint": cache_path.stem,
                },
            )

        return events_df

    def _score_event_quality(self, events_df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """为每条事件生成质量分。"""
        event_quality_df = build_event_quality_table(events_df, raw_data)
        self.latest_event_quality_df = event_quality_df.copy()
        return event_quality_df

    def _form_candidates(self, events_df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """候选成形"""
        return build_candidate_forms(events_df, raw_data)

    def _apply_candidate_event_quality(
        self,
        candidate_df: pd.DataFrame,
        event_quality_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """将事件质量聚合到候选层和证据项。"""
        return merge_event_quality_into_candidates(candidate_df, event_quality_df)

    def _candidate_flow_key(self, row: pd.Series, index: int) -> str:
        identity_parts = [
            row.get("candidate_stage"),
            row.get("stable_object_id"),
            row.get("stable_object_label"),
            row.get("display_candidate_name"),
            row.get("topic_summary_name"),
            row.get("canonical_candidate_name_en"),
            row.get("mechanism_core"),
            row.get("relation_target"),
            row.get("relation_task"),
            row.get("relation_data_modality"),
            row.get("relation_method"),
            row.get("constraint_signature"),
            row.get("primary_scope"),
            row.get("scope_name"),
        ]
        normalized = []
        for part in identity_parts:
            text = self._safe_report_text(part).lower()
            if not text:
                continue
            text = re.sub(r"[\s_\-]+", "", text)
            text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
            if text:
                normalized.append(text)
        if normalized:
            return "::".join(normalized)

        for field in ["candidate_cluster_id", "candidate_id", "id"]:
            text = self._safe_report_text(row.get(field))
            if text:
                return f"{field}::{text}"
        return f"candidate_row_{index + 1}"

    def _dedupe_scalar_preferred(self, rows: List[pd.Series], field: str) -> Any:
        for row in rows:
            value = row.get(field)
            if self._safe_report_text(value):
                return value
        return rows[0].get(field) if rows else ""

    def _dedupe_list_values(self, rows: List[pd.Series], field: str, limit: Optional[int] = None) -> List[Any]:
        values = []
        seen = set()
        for row in rows:
            for item in self._safe_report_list(row.get(field)):
                key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else self._safe_report_text(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                values.append(item)
                if limit is not None and len(values) >= limit:
                    return values
        return values

    def _dedupe_evidence_items(self, rows: List[pd.Series], limit: int = 30) -> List[Dict[str, Any]]:
        items = []
        seen = set()
        for row in rows:
            for item in self._safe_report_list(row.get("evidence_items")):
                if not isinstance(item, dict):
                    continue
                key = (
                    self._safe_report_text(item.get("id") or item.get("event_id") or item.get("source_id")),
                    self._safe_report_text(item.get("title")),
                    self._safe_report_text(item.get("date")),
                    self._safe_report_text(item.get("source_type")),
                )
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
                if len(items) >= limit:
                    return items
        return items

    def _merge_candidate_flow_group(self, rows: List[pd.Series], flow_key: str) -> Dict[str, Any]:
        def metric(row: pd.Series) -> tuple:
            stage_rank = {
                "formed_candidate_strong": 0,
                "formed_candidate": 1,
                "scope_overview": 2,
            }.get(self._safe_report_text(row.get("candidate_stage")), 3)
            return (
                stage_rank,
                -self._safe_report_float(row.get("quality_adjusted_rank_score"), 0.0),
                -self._safe_report_float(row.get("candidate_core_evidence_quality"), 0.0),
                -self._safe_report_float(row.get("cluster_evidence_count"), 0.0),
                -self._safe_report_float(row.get("source_count"), 0.0),
            )

        ordered_rows = sorted(rows, key=metric)
        representative = ordered_rows[0].copy().to_dict()
        mention_ids = self._dedupe_list_values(ordered_rows, "mention_ids")
        mention_dates = self._dedupe_list_values(ordered_rows, "mention_dates")
        source_types = self._dedupe_list_values(ordered_rows, "source_types")
        orgs = self._dedupe_list_values(ordered_rows, "orgs")
        aliases = self._dedupe_list_values(ordered_rows, "display_candidate_aliases")
        evidence_titles = self._dedupe_list_values(ordered_rows, "evidence_titles", limit=30)
        evidence_items = self._dedupe_evidence_items(ordered_rows, limit=30)

        total_mentions = len(mention_ids) if mention_ids else int(max(self._safe_report_float(row.get("total_mentions"), 0.0) for row in ordered_rows))
        source_count = len(source_types) if source_types else int(max(self._safe_report_float(row.get("source_count"), 0.0) for row in ordered_rows))
        org_count = len(orgs) if orgs else int(max(self._safe_report_float(row.get("org_count"), 0.0) for row in ordered_rows))
        evidence_count = len(mention_ids) if mention_ids else len(evidence_items)
        evidence_count = max(evidence_count, int(max(self._safe_report_float(row.get("cluster_evidence_count"), 0.0) for row in ordered_rows)))

        ratio_fields = [
            "weak_signal_event_ratio",
            "low_attention_ratio",
            "niche_actor_ratio",
            "non_dominant_ratio",
            "cross_domain_ratio",
            "traceable_ratio",
            "low_quality_evidence_ratio",
        ]
        for field in ratio_fields:
            weighted_sum = 0.0
            weight_total = 0.0
            for row in ordered_rows:
                weight = max(self._safe_report_float(row.get("total_mentions"), 0.0), 1.0)
                weighted_sum += self._safe_report_float(row.get(field), 0.0) * weight
                weight_total += weight
            if weight_total > 0:
                representative[field] = round(weighted_sum / weight_total, 3)

        max_fields = [
            "candidate_evidence_quality",
            "candidate_core_evidence_quality",
            "high_quality_evidence_count",
            "quality_adjusted_rank_score",
            "object_like_score",
            "cluster_object_specificity",
            "specific_anchor_strength",
            "non_scope_constraint_count",
            "template_variant_count",
            "alias_count",
            "cluster_item_count",
            "weak_signal_event_count",
        ]
        for field in max_fields:
            values = [self._safe_report_float(row.get(field), 0.0) for row in ordered_rows]
            if values:
                representative[field] = max(values)

        preferred_text_fields = [
            "candidate_cluster_id",
            "display_candidate_name",
            "topic_summary_name",
            "final_research_object_name",
            "representative_candidate_name",
            "current_representative_name",
            "stable_object_label",
            "mechanism_core",
            "relation_target",
            "relation_task",
            "relation_data_modality",
            "relation_method",
            "constraint_signature",
            "primary_scope",
            "scope_name",
            "display_tier",
            "candidate_stage",
        ]
        for field in preferred_text_fields:
            representative[field] = self._dedupe_scalar_preferred(ordered_rows, field)

        representative["candidate_flow_key"] = flow_key
        representative["candidate_id"] = self._safe_report_text(representative.get("candidate_cluster_id")) or f"flow::{hashlib.md5(flow_key.encode('utf-8')).hexdigest()[:12]}"
        representative["dedupe_count"] = len(ordered_rows)
        representative["dedupe_source_row_count"] = len(ordered_rows)
        representative["dedupe_merged_candidate_ids"] = self._dedupe_list_values(ordered_rows, "candidate_cluster_id", limit=50)
        representative["dedupe_reason"] = "pipeline_candidate_flow_dedupe" if len(ordered_rows) > 1 else ""
        representative["mention_ids"] = mention_ids
        representative["mention_dates"] = mention_dates
        representative["source_types"] = source_types
        representative["source_count"] = source_count
        representative["total_mentions"] = total_mentions
        representative["orgs"] = orgs
        representative["org_count"] = org_count
        representative["display_candidate_aliases"] = aliases
        representative["alias_count"] = max(len(aliases), int(self._safe_report_float(representative.get("alias_count"), 0.0)))
        representative["evidence_titles"] = evidence_titles
        representative["evidence_items"] = evidence_items
        representative["cluster_evidence_count"] = evidence_count
        return representative

    def _dedupe_candidate_flow(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """候选流转前置去重，减少后续评分、验证和映射的重复消耗。"""
        if candidate_df is None or candidate_df.empty:
            return candidate_df.copy() if isinstance(candidate_df, pd.DataFrame) else pd.DataFrame()

        groups: Dict[str, List[pd.Series]] = {}
        for index, row in candidate_df.reset_index(drop=True).iterrows():
            groups.setdefault(self._candidate_flow_key(row, index), []).append(row)

        merged_rows = [
            self._merge_candidate_flow_group(rows, flow_key)
            for flow_key, rows in groups.items()
        ]
        deduped = pd.DataFrame(merged_rows)
        for column in candidate_df.columns:
            if column not in deduped.columns:
                deduped[column] = ""
        return deduped.reset_index(drop=True)

    def _score_candidates(self, candidate_forms_df: pd.DataFrame) -> pd.DataFrame:
        """评分"""
        return score_all_candidates(candidate_forms_df)

    def _refine_topics(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """主题细化"""
        return refine_research_scored_candidates(scored_df, top_k=20)

    def _validate_reverse(
        self,
        scored_df: pd.DataFrame,
        candidate_forms_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """反向验证"""
        reverse_df = build_reverse_validation_table(scored_df, candidate_forms_df, top_k=15)
        self.latest_reverse_validation_df = reverse_df.copy()
        validated_df = merge_reverse_validation_into_manual_review(scored_df, reverse_df)
        return refresh_research_layers(validated_df)

    def _map_tech_chain(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """将候选对象映射到轻量技术链先验。"""
        if candidate_df is None or candidate_df.empty:
            self.latest_tech_chain_mapping_df = pd.DataFrame()
            return candidate_df.copy() if isinstance(candidate_df, pd.DataFrame) else pd.DataFrame()
        tech_chain_data = load_tech_chain_data(Config.DATA_DIR / "tech_chain")
        mapping_df = build_tech_chain_mapping_table(candidate_df, tech_chain_data)
        self.latest_tech_chain_mapping_df = mapping_df.copy()
        return merge_tech_chain_mapping_into_candidates(candidate_df, mapping_df)

    def _validate_temporal(
        self,
        candidate_df: pd.DataFrame,
        events_df: pd.DataFrame,
        raw_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """生成候选级时间验证表。"""
        temporal_validation_df = build_temporal_validation_table(candidate_df, events_df, raw_data)
        self.latest_temporal_validation_df = temporal_validation_df.copy()
        return temporal_validation_df

    def _score_key_core_potential(
        self,
        candidate_df: pd.DataFrame,
        temporal_validation_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """生成关键核心潜力评分并缓存关键核心候选表。"""
        key_core_scored_df = score_key_core_candidates(candidate_df, temporal_validation_df)
        self.latest_key_core_scored_df = key_core_scored_df.copy()
        self.latest_key_core_candidates_df = build_key_core_candidate_table(key_core_scored_df, top_k=20)
        return key_core_scored_df

    def _build_key_core_candidates(self, key_core_scored_df: pd.DataFrame) -> pd.DataFrame:
        """构建关键核心潜力候选短名单。"""
        key_core_candidates_df = build_key_core_candidate_table(key_core_scored_df, top_k=20)
        self.latest_key_core_candidates_df = key_core_candidates_df.copy()
        return key_core_candidates_df

    def _generate_signals(self, candidate_df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """信号生成"""
        return generate_candidate_outputs(candidate_df, raw_data)

    def _build_research_validation_artifacts(self, validated_df: pd.DataFrame) -> None:
        """Build shortlist and baseline comparison artifacts after reverse validation."""
        if validated_df is None or validated_df.empty:
            self.latest_final_shortlist_df = pd.DataFrame()
            self.latest_shortlist_dedup_df = pd.DataFrame()
            self.latest_frequency_baseline_df = pd.DataFrame()
            self.latest_baseline_comparison_df = pd.DataFrame()
            return

        final_shortlist_df, shortlist_dedup_df = build_final_shortlist(validated_df, top_k=20)
        frequency_baseline_df = build_frequency_baseline_table(
            validated_df,
            reverse_validation_df=self.latest_reverse_validation_df,
            top_k=20,
        )
        baseline_comparison_df = build_baseline_comparison_table(
            validated_df,
            frequency_baseline_df,
            reverse_validation_df=self.latest_reverse_validation_df,
            top_k=20,
        )

        self.latest_final_shortlist_df = final_shortlist_df
        self.latest_shortlist_dedup_df = shortlist_dedup_df
        self.latest_frequency_baseline_df = frequency_baseline_df
        self.latest_baseline_comparison_df = baseline_comparison_df
        self.report_artifacts["final_shortlist_summary"] = {
            "shortlist_count": int(len(final_shortlist_df)),
            "frequency_baseline_count": int(len(frequency_baseline_df)),
            "baseline_comparison_count": int(len(baseline_comparison_df)),
        }

    @staticmethod
    def _family_priority_rank(value: Any) -> int:
        mapping = {"high": 0, "medium": 1, "low": 2}
        return mapping.get(str(value).strip().lower(), 3)

    def _build_family_summary(self, family_metrics_df: pd.DataFrame) -> Dict[str, Any]:
        if family_metrics_df is None or family_metrics_df.empty or "family_coverage" not in family_metrics_df.columns:
            return {"covered_family_count": 0, "high_priority_covered_count": 0, "top_families": []}

        covered = family_metrics_df[family_metrics_df["family_coverage"].fillna(0) > 0].copy()
        if covered.empty:
            return {"covered_family_count": 0, "high_priority_covered_count": 0, "top_families": []}

        covered["priority_rank"] = covered.get("priority", "").apply(self._family_priority_rank)
        covered = covered.sort_values(
            by=["priority_rank", "family_cross_source_count", "family_coverage", "family_semantic_validation_score"],
            ascending=[True, False, False, False],
            na_position="last",
        )

        top_families = []
        for _, row in covered.head(5).iterrows():
            top_families.append(
                {
                    "family_id": self._safe_report_text(row.get("family_id")),
                    "canonical_term": self._safe_report_text(row.get("canonical_term")),
                    "priority": self._safe_report_text(row.get("priority")) or "unknown",
                    "family_coverage": int(row.get("family_coverage", 0) or 0),
                    "family_cross_source_count": int(row.get("family_cross_source_count", 0) or 0),
                    "family_reverse_validation_status": self._safe_report_text(row.get("family_reverse_validation_status")),
                    "family_semantic_consistency": self._safe_report_text(row.get("family_semantic_consistency")),
                    "family_semantic_validation_score": round(float(row.get("family_semantic_validation_score", 0.0) or 0.0), 3),
                }
            )

        return {
            "covered_family_count": int(len(covered)),
            "high_priority_covered_count": int((covered.get("priority", "").astype(str).str.lower() == "high").sum()),
            "top_families": top_families,
        }

    def _evaluate_family_metrics(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        if candidates_df is None or candidates_df.empty:
            self.latest_family_metrics_df = pd.DataFrame()
            self.report_artifacts["family_evaluation_metrics"] = []
            self.report_artifacts["family_evaluation_report"] = ""
            self.report_artifacts["family_evaluation_summary"] = self._build_family_summary(pd.DataFrame())
            return self.latest_family_metrics_df

        family_metrics_df = evaluate_families(candidates_df, self.canonicalizer)
        self.latest_family_metrics_df = family_metrics_df
        self.report_artifacts["family_evaluation_metrics"] = family_metrics_df.to_dict(orient="records")
        self.report_artifacts["family_evaluation_report"] = generate_family_report(family_metrics_df)
        self.report_artifacts["family_evaluation_summary"] = self._build_family_summary(family_metrics_df)
        return family_metrics_df

    def _compose_family_metrics_summary(self) -> str:
        summary = self.report_artifacts.get("family_evaluation_summary", {}) or {}
        top_families = [item for item in summary.get("top_families", []) if isinstance(item, dict)]
        if not top_families:
            return "对象族评估尚未形成可稳定承接的候选骨架，当前仍以单点弱信号跟踪为主。"

        lead_parts = []
        first = top_families[0]
        first_name = self._safe_report_text(first.get("canonical_term")) or "当前主干对象族"
        first_coverage = int(first.get("family_coverage", 0) or 0)
        first_cross = int(first.get("family_cross_source_count", 0) or 0)
        lead_parts.append(
            f"对象族评估显示，{first_name} 当前具备最强承接骨架，已覆盖 {first_coverage} 个候选，其中 {first_cross} 个具备跨源支撑。"
        )

        followup = []
        for item in top_families[1:3]:
            name = self._safe_report_text(item.get("canonical_term"))
            if not name:
                continue
            consistency = self._safe_report_text(item.get("family_semantic_consistency")) or "unknown"
            followup.append(f"{name} 也已形成初步对象族承接，语义一致性为 {consistency}。")

        high_count = int(summary.get("high_priority_covered_count", 0) or 0)
        if high_count > 0:
            followup.append(f"当前已有 {high_count} 个高优先级对象族被真实候选覆盖。")

        paragraph = " ".join(part for part in lead_parts + followup if part).strip()
        return paragraph or "对象族评估已完成，但尚未形成足够清晰的承接主干。"

    @staticmethod
    def _safe_report_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
        else:
            try:
                if pd.isna(value):
                    return ""
            except Exception:
                pass
            text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none", "null", "nat"} else text

    @staticmethod
    def _safe_report_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_report_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = AnalysisPipeline._safe_report_text(value).lower()
        return text in {"1", "true", "yes", "y", "是", "通过"}

    @staticmethod
    def _truncate_report_text(value: Any, limit: int = 240) -> str:
        text = AnalysisPipeline._safe_report_text(value)
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)].rstrip() + "..."

    @staticmethod
    def _safe_report_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text[0] in "[{":
                try:
                    parsed = json.loads(text)
                except Exception:
                    return []
                return AnalysisPipeline._safe_report_list(parsed)
            return []
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        return []

    def _signal_display_name(self, row: pd.Series) -> str:
        return (
            self._safe_report_text(row.get("display_candidate_name"))
            or self._safe_report_text(row.get("canonical_candidate_name_en"))
            or self._safe_report_text(row.get("tech_name"))
            or "Unknown"
        )

    def _signal_score(self, row: pd.Series) -> float:
        for key in ["weak_signal_score", "hotspot_score", "score"]:
            value = row.get(key, 0.0)
            try:
                return float(value or 0.0)
            except Exception:
                continue
        return 0.0

    def _sort_report_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        sort_cols = []
        for column in ["weak_signal_score", "quality_adjusted_rank_score", "hotspot_score", "cluster_evidence_count", "source_count", "total_mentions"]:
            if column in df.columns:
                sort_cols.append(column)
        if not sort_cols:
            return df.reset_index(drop=True)
        return df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)

    def _normalize_report_inputs(self, signals_output):
        if isinstance(signals_output, dict):
            candidates_df = signals_output.get("candidates_df", pd.DataFrame())
            near_strong_df = signals_output.get("near_strong_candidates_df", pd.DataFrame())
        elif isinstance(signals_output, pd.DataFrame):
            candidates_df = signals_output
            near_strong_df = pd.DataFrame()
        else:
            candidates_df = pd.DataFrame(signals_output if signals_output is not None else [])
            near_strong_df = pd.DataFrame()

        if not isinstance(candidates_df, pd.DataFrame):
            candidates_df = pd.DataFrame(candidates_df if candidates_df is not None else [])
        if not isinstance(near_strong_df, pd.DataFrame):
            near_strong_df = pd.DataFrame(near_strong_df if near_strong_df is not None else [])

        if not candidates_df.empty and "signal_type" in candidates_df.columns:
            weak_signals = candidates_df[candidates_df["signal_type"] == "weak_signal"].copy()
            scope_overviews = (
                candidates_df[candidates_df["candidate_stage"] == "scope_overview"].copy()
                if "candidate_stage" in candidates_df.columns
                else pd.DataFrame()
            )
        else:
            weak_signals = pd.DataFrame()
            scope_overviews = pd.DataFrame()

        return (
            candidates_df,
            self._sort_report_rows(weak_signals),
            self._sort_report_rows(near_strong_df),
            self._sort_report_rows(scope_overviews),
        )

    def _build_signal_packet(self, row: pd.Series, signal_id: str, signal_kind: str, max_evidence: int = 4) -> Dict[str, Any]:
        evidence_items = []
        seen = set()
        raw_evidence_items = [
            item for item in self._safe_report_list(row.get("evidence_items", []))
            if isinstance(item, dict)
        ]
        low_quality_evidence_count = sum(
            1
            for item in raw_evidence_items
            if 0 < self._safe_report_float(item.get("event_quality_score"), 0.0) < 4.0
        )
        report_evidence_items = [
            item
            for item in raw_evidence_items
            if self._safe_report_float(item.get("event_quality_score"), 0.0) <= 0
            or self._safe_report_float(item.get("event_quality_score"), 0.0) >= 4.0
        ]
        report_evidence_items = sorted(
            report_evidence_items,
            key=lambda item: (
                self._safe_report_float(item.get("event_quality_score"), -1.0),
                self._safe_report_text(item.get("source_type")).lower() in {"paper", "patent", "report"},
            ),
            reverse=True,
        )
        for item in report_evidence_items:
            if not isinstance(item, dict):
                continue
            source_type = self._safe_report_text(item.get("source_type")).lower() or "unknown"
            title = self._safe_report_text(item.get("title"))
            snippet = self._truncate_report_text(item.get("snippet") or item.get("text"), 280)
            if not snippet and not title:
                continue
            dedupe_key = (source_type, title, snippet)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            evidence_items.append(
                {
                    "evidence_id": f"{signal_id}-E{len(evidence_items) + 1}",
                    "record_id": self._safe_report_text(item.get("id")),
                    "source_type": source_type,
                    "org": self._safe_report_text(item.get("org")) or "Unknown",
                    "date": self._safe_report_text(item.get("date")),
                    "url": self._safe_report_text(item.get("url")),
                    "title": title or self._signal_display_name(row),
                    "snippet": snippet,
                    "text": self._truncate_report_text(item.get("text") or item.get("snippet"), 520),
                    "raw_candidate_text": self._safe_report_text(item.get("raw_candidate_text")),
                    "event_quality_score": round(self._safe_report_float(item.get("event_quality_score"), 0.0), 2),
                    "event_quality_tier": self._safe_report_text(item.get("event_quality_tier")),
                    "event_quality_reason": self._truncate_report_text(item.get("event_quality_reason"), 180),
                }
            )
            if len(evidence_items) >= max_evidence:
                break

        source_types = [self._safe_report_text(item).lower() for item in self._safe_report_list(row.get("source_types", []))]
        source_types = [item for item in source_types if item]
        orgs = [self._safe_report_text(item) for item in self._safe_report_list(row.get("orgs", []))]
        orgs = [item for item in orgs if item]

        packet = {
            "signal_id": signal_id,
            "signal_kind": signal_kind,
            "name": self._signal_display_name(row),
            "score": round(self._signal_score(row), 3),
            "mechanism_core": self._safe_report_text(row.get("mechanism_core")) or "unknown",
            "signal_bucket": self._safe_report_text(
                row.get("signal_bucket")
                or row.get("final_research_bucket")
                or row.get("signal_type")
            ),
            "stage_hypothesis": self._safe_report_text(row.get("stage_hypothesis")),
            "source_types": source_types,
            "source_count": int(row.get("source_count", len(source_types)) or len(source_types)),
            "evidence_count": int(row.get("cluster_evidence_count", len(evidence_items)) or len(evidence_items)),
            "org_count": int(row.get("org_count", len(orgs)) or len(orgs)),
            "orgs": orgs[:8],
            "time_validation": {
                "early_count": int(row.get("time_validation_early_count", 0) or 0),
                "recent_count": int(row.get("time_validation_recent_count", 0) or 0),
                "ratio": float(row.get("time_validation_ratio", 0.0) or 0.0),
                "validated": bool(row.get("time_validated", False)),
            },
            "validation": {
                "multi_source_validated": bool(row.get("multi_source_validated", False)),
                "semantic_validated": bool(row.get("semantic_validated", False)),
                "semantic_score": float(row.get("semantic_validation_score", 0.0) or 0.0),
                "event_doc_semantic_validated": bool(row.get("event_doc_semantic_validated", False)),
                "event_doc_validation_score": float(row.get("event_doc_validation_score", 0.0) or 0.0),
                "traceable_ratio": float(row.get("traceable_ratio", 0.0) or 0.0),
            },
            "reverse_validation": {
                "status": self._safe_report_text(row.get("reverse_validation_status")),
                "semantic_consistency": self._safe_report_text(row.get("source_semantic_consistency")),
                "object_alignment": self._safe_report_text(row.get("object_semantic_alignment")),
                "alignment_confidence": self._safe_report_text(row.get("alignment_confidence")),
                "release_alignment_pass": bool(row.get("release_alignment_pass", False)),
                "release_alignment_risk": self._safe_report_text(row.get("release_alignment_risk")),
                "note": self._safe_report_text(row.get("reverse_validation_note")),
            },
            "family": {
                "family_id": self._safe_report_text(row.get("family_id")),
                "priority": self._safe_report_text(row.get("family_priority")),
                "semantic_consistency": self._safe_report_text(row.get("family_semantic_consistency")),
                "semantic_validation_score": float(row.get("family_semantic_validation_score", 0.0) or 0.0),
            },
            "evidence_quality": {
                "candidate_evidence_quality": round(self._safe_report_float(row.get("candidate_evidence_quality"), 0.0), 2),
                "candidate_core_evidence_quality": round(self._safe_report_float(row.get("candidate_core_evidence_quality"), 0.0), 2),
                "low_quality_evidence_ratio": round(self._safe_report_float(row.get("low_quality_evidence_ratio"), 0.0), 3),
                "high_quality_evidence_count": int(self._safe_report_float(row.get("high_quality_evidence_count"), 0.0)),
                "quality_risk_flag": self._safe_report_text(row.get("quality_risk_flag")),
                "excluded_low_quality_evidence_count": low_quality_evidence_count,
            },
            "tech_chain_mapping": {
                "tech_chain_node_id": self._safe_report_text(row.get("tech_chain_node_id")),
                "tech_chain_name": self._safe_report_text(row.get("tech_chain_name")),
                "tech_chain_official_name": self._safe_report_text(row.get("tech_chain_official_name")),
                "mapping_relation": self._safe_report_text(row.get("mapping_relation")),
                "mapping_confidence": round(self._safe_report_float(row.get("mapping_confidence"), 0.0), 3),
                "mapping_method": self._safe_report_text(row.get("mapping_method")),
                "matched_term": self._safe_report_text(row.get("matched_term")),
                "parent_technology": self._safe_report_text(row.get("parent_technology")),
                "bottleneck_level": self._safe_report_text(row.get("bottleneck_level")),
                "strategic_importance_level": self._safe_report_text(row.get("strategic_importance_level")),
                "mapping_reason": self._safe_report_text(row.get("mapping_reason")),
                "mapping_risk": self._safe_report_text(row.get("tech_chain_mapping_risk")),
                "coverage_flag": self._safe_report_text(row.get("tech_chain_mapping_coverage_flag")),
            },
            "temporal_validation": {
                "tier": self._safe_report_text(row.get("temporal_validation_tier")),
                "status": self._safe_report_text(row.get("temporal_validation_status")),
                "passed": self._safe_report_bool(row.get("temporal_validation_passed", False)),
                "momentum_score": round(self._safe_report_float(row.get("temporal_momentum_score"), 0.0), 2),
                "growth_rate": round(self._safe_report_float(row.get("growth_rate"), 0.0), 3),
                "source_growth_rate": round(self._safe_report_float(row.get("source_growth_rate"), 0.0), 3),
                "org_growth_rate": round(self._safe_report_float(row.get("org_growth_rate"), 0.0), 3),
                "date_coverage_ratio": round(self._safe_report_float(row.get("date_coverage_ratio"), 0.0), 3),
                "reason": self._truncate_report_text(row.get("temporal_validation_reason"), 220),
            },
            "key_core_potential": {
                "key_core_score": round(self._safe_report_float(row.get("key_core_score"), 0.0), 2),
                "key_core_tier": self._safe_report_text(row.get("key_core_tier")),
                "weak_signal_component": round(self._safe_report_float(row.get("weak_signal_component"), 0.0), 2),
                "growth_validation_component": round(self._safe_report_float(row.get("growth_validation_component"), 0.0), 2),
                "tech_chain_bottleneck_component": round(self._safe_report_float(row.get("tech_chain_bottleneck_component"), 0.0), 2),
                "strategic_importance_component": round(self._safe_report_float(row.get("strategic_importance_component"), 0.0), 2),
                "evidence_confidence_component": round(self._safe_report_float(row.get("evidence_confidence_component"), 0.0), 2),
                "asset_support_component": round(self._safe_report_float(row.get("asset_support_component"), 0.0), 2),
                "quality_gate_passed": self._safe_report_bool(row.get("quality_gate_passed", False)),
                "mapping_gate_passed": self._safe_report_bool(row.get("mapping_gate_passed", False)),
                "temporal_gate_passed": self._safe_report_bool(row.get("temporal_gate_passed", False)),
                "object_gate_passed": self._safe_report_bool(row.get("object_gate_passed", False)),
                "key_core_gate_passed": self._safe_report_bool(row.get("key_core_gate_passed", False)),
                "reason": self._truncate_report_text(row.get("key_core_reason"), 260),
                "risk": self._safe_report_text(row.get("key_core_risk")),
                "recommended_action": self._safe_report_text(row.get("recommended_action")),
            },
            "evidence_items": evidence_items,
            "primary_evidence_ids": [item["evidence_id"] for item in evidence_items[:2]],
        }
        packet["chain_summary"] = self._build_packet_chain_summary(packet)
        packet["timeline_summary"] = self._build_packet_timeline_summary(packet)
        packet["organization_summary"] = self._build_packet_org_summary(packet)
        packet["evidence_highlights"] = self._build_packet_evidence_highlights(packet)
        return packet

    def _build_key_core_report_candidates(self, candidates_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if candidates_df is None or candidates_df.empty or "key_core_score" not in candidates_df.columns:
            return []

        df = candidates_df.copy()
        weak_score_series = (
            df["weak_signal_score"]
            if "weak_signal_score" in df.columns
            else pd.Series([0.0] * len(df), index=df.index)
        )
        df["_key_core_score"] = pd.to_numeric(df["key_core_score"], errors="coerce").fillna(0.0)
        df["_weak_signal_score"] = pd.to_numeric(weak_score_series, errors="coerce").fillna(0.0)
        preferred_tiers = {"core_key_candidate", "strong_key_potential", "watchlist_key_potential"}
        tier_order = {
            "core_key_candidate": 0,
            "strong_key_potential": 1,
            "watchlist_key_potential": 2,
            "weak_signal_only": 3,
            "insufficient_evidence": 4,
            "not_key_core_candidate": 5,
        }
        tier_series = (
            df["key_core_tier"]
            if "key_core_tier" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        gate_series = (
            df["key_core_gate_passed"]
            if "key_core_gate_passed" in df.columns
            else pd.Series([False] * len(df), index=df.index)
        )
        df["_tier_order"] = tier_series.map(
            lambda value: tier_order.get(self._safe_report_text(value), 9)
        )

        candidate_subset = df[
            tier_series.astype(str).isin(preferred_tiers)
            | (df["_key_core_score"] >= 50)
            | gate_series.map(self._safe_report_bool)
        ].copy()
        if candidate_subset.empty:
            candidate_subset = df[df["_key_core_score"] > 0].copy()
        if candidate_subset.empty:
            return []

        dedupe_column = next(
            (
                column
                for column in ["candidate_cluster_id", "candidate_id", "display_candidate_name", "candidate_name"]
                if column in candidate_subset.columns
            ),
            None,
        )
        candidate_subset = candidate_subset.sort_values(
            by=["_tier_order", "_key_core_score", "_weak_signal_score"],
            ascending=[True, False, False],
            na_position="last",
        )
        if dedupe_column:
            candidate_subset = candidate_subset.drop_duplicates(subset=[dedupe_column], keep="first")

        candidates = []
        for rank, (_, row) in enumerate(candidate_subset.head(10).iterrows(), 1):
            candidates.append(
                {
                    "rank": rank,
                    "name": (
                        self._safe_report_text(row.get("candidate_name"))
                        or self._safe_report_text(row.get("display_candidate_name"))
                        or self._safe_report_text(row.get("final_research_object_name"))
                        or "Unknown"
                    ),
                    "key_core_score": round(self._safe_report_float(row.get("key_core_score"), 0.0), 2),
                    "key_core_tier": self._safe_report_text(row.get("key_core_tier")),
                    "key_core_gate_passed": self._safe_report_bool(row.get("key_core_gate_passed", False)),
                    "weak_signal_score": round(self._safe_report_float(row.get("weak_signal_score"), 0.0), 2),
                    "tech_chain_name": self._safe_report_text(row.get("tech_chain_name")),
                    "mapping_relation": self._safe_report_text(row.get("mapping_relation")),
                    "bottleneck_level": self._safe_report_text(row.get("bottleneck_level")),
                    "strategic_importance_level": self._safe_report_text(row.get("strategic_importance_level")),
                    "temporal_validation_tier": self._safe_report_text(row.get("temporal_validation_tier")),
                    "temporal_momentum_score": round(self._safe_report_float(row.get("temporal_momentum_score"), 0.0), 2),
                    "candidate_core_evidence_quality": round(
                        self._safe_report_float(
                            row.get("candidate_core_evidence_quality"),
                            self._safe_report_float(row.get("candidate_evidence_quality"), 0.0),
                        ),
                        2,
                    ),
                    "source_count": int(self._safe_report_float(row.get("source_count"), 0.0)),
                    "evidence_count": int(self._safe_report_float(row.get("cluster_evidence_count"), 0.0)),
                    "reason": self._truncate_report_text(row.get("key_core_reason"), 260),
                    "risk": self._safe_report_text(row.get("key_core_risk")),
                    "recommended_action": self._safe_report_text(row.get("recommended_action")),
                    "components": {
                        "weak_signal": round(self._safe_report_float(row.get("weak_signal_component"), 0.0), 2),
                        "growth_validation": round(self._safe_report_float(row.get("growth_validation_component"), 0.0), 2),
                        "tech_chain_bottleneck": round(
                            self._safe_report_float(row.get("tech_chain_bottleneck_component"), 0.0),
                            2,
                        ),
                        "strategic_importance": round(
                            self._safe_report_float(row.get("strategic_importance_component"), 0.0),
                            2,
                        ),
                        "evidence_confidence": round(
                            self._safe_report_float(row.get("evidence_confidence_component"), 0.0),
                            2,
                        ),
                        "asset_support": round(self._safe_report_float(row.get("asset_support_component"), 0.0), 2),
                    },
                }
            )
        return candidates

    def _build_grounded_report_packets(
        self,
        candidates_df: pd.DataFrame,
        weak_signals: pd.DataFrame,
        near_strong_df: pd.DataFrame,
        scope_overviews: pd.DataFrame,
    ) -> Dict[str, Any]:
        weak_evidence_limit = 2 if len(weak_signals) > 50 else 4
        weak_packets = [
            self._build_signal_packet(row, f"WS{idx}", "weak_signal", max_evidence=weak_evidence_limit)
            for idx, (_, row) in enumerate(weak_signals.iterrows(), 1)
        ]
        near_packets = [
            self._build_signal_packet(row, f"NS{idx}", "near_strong")
            for idx, (_, row) in enumerate(near_strong_df.head(3).iterrows(), 1)
        ]
        scope_packets = []
        for idx, (_, row) in enumerate(scope_overviews.head(5).iterrows(), 1):
            scope_packets.append(
                {
                    "scope_id": f"SC{idx}",
                    "name": self._safe_report_text(row.get("display_candidate_name")) or self._safe_report_text(row.get("scope_name")) or "Unknown",
                    "evidence_count": int(row.get("cluster_evidence_count", 0) or 0),
                    "source_count": int(row.get("source_count", 0) or 0),
                    "mechanism_core": self._safe_report_text(row.get("mechanism_core")),
                }
            )

        weak_signal_clusters = self._build_weak_signal_clusters(weak_packets)
        key_core_candidates = self._build_key_core_report_candidates(candidates_df)

        evidence_lookup = {}
        signal_lookup = {}
        for packet in weak_packets + near_packets:
            signal_lookup[packet["signal_id"]] = {
                "signal_id": packet["signal_id"],
                "signal_kind": packet["signal_kind"],
                "name": packet["name"],
            }
            for evidence in packet["evidence_items"]:
                evidence_lookup[evidence["evidence_id"]] = evidence

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "candidate_count": int(len(candidates_df)),
                "weak_signal_count": int(len(weak_signals)),
                "near_strong_count": int(len(near_strong_df)),
                "scope_count": int(len(scope_overviews)),
                "weak_signal_cluster_count": int(len(weak_signal_clusters)),
                "key_core_candidate_count": int(len(key_core_candidates)),
            },
            "family_summary": self.report_artifacts.get("family_evaluation_summary", {}),
            "key_core_candidates": key_core_candidates,
            "weak_signals": weak_packets,
            "weak_signal_clusters": weak_signal_clusters,
            "near_strong_signals": near_packets,
            "scope_overviews": scope_packets,
            "signal_lookup": signal_lookup,
            "evidence_lookup": evidence_lookup,
        }

    def _weak_cluster_key(self, packet: Dict[str, Any]) -> tuple:
        family = packet.get("family", {}) or {}
        family_id = self._safe_report_text(family.get("family_id"))
        if family_id:
            return f"family:{family_id}", "对象族"

        mechanism = self._safe_report_text(packet.get("mechanism_core"))
        if mechanism and mechanism.lower() not in {"unknown", "none", "nan"}:
            return f"mechanism:{mechanism.lower()}", "机制核"

        bucket = self._safe_report_text(packet.get("signal_bucket"))
        if bucket and bucket.lower() not in {"unknown", "none", "nan"}:
            return f"bucket:{bucket.lower()}", "研究分桶"

        return "uncategorized", "综合线索"

    def _append_unique_text(self, values: List[str], value: Any, limit: Optional[int] = None) -> None:
        text = self._safe_report_text(value)
        if not text or text in values:
            return
        if limit is not None and len(values) >= limit:
            return
        values.append(text)

    def _build_weak_signal_clusters(self, weak_packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for packet in weak_packets:
            key, basis = self._weak_cluster_key(packet)
            group = groups.setdefault(key, {"basis": basis, "packets": []})
            group["packets"].append(packet)

        clusters = []
        for key, group in groups.items():
            packets = sorted(
                group["packets"],
                key=lambda item: (
                    float(item.get("score", 0.0) or 0.0),
                    int(item.get("source_count", 0) or 0),
                    int(item.get("evidence_count", 0) or 0),
                ),
                reverse=True,
            )
            if not packets:
                continue

            covered_names: List[str] = []
            covered_signals: List[Dict[str, Any]] = []
            mechanisms: List[str] = []
            buckets: List[str] = []
            source_types: List[str] = []
            for packet in packets:
                self._append_unique_text(covered_names, packet.get("name"))
                covered_signals.append(
                    {
                        "signal_id": packet.get("signal_id"),
                        "name": packet.get("name"),
                    }
                )
                self._append_unique_text(mechanisms, packet.get("mechanism_core"), limit=4)
                self._append_unique_text(buckets, packet.get("signal_bucket"), limit=4)
                for source_type in packet.get("source_types", []):
                    self._append_unique_text(source_types, source_type)

            scores = [float(packet.get("score", 0.0) or 0.0) for packet in packets]
            representative_packets = packets[:3]
            main_mechanism = mechanisms[0] if mechanisms else ""
            if group["basis"] == "机制核" and main_mechanism:
                cluster_name = f"{main_mechanism}相关弱信号簇"
            elif group["basis"] == "研究分桶" and buckets:
                cluster_name = f"{buckets[0]}弱信号簇"
            else:
                cluster_name = covered_names[0] if covered_names else "未命名弱信号簇"

            clusters.append(
                {
                    "cluster_key": key,
                    "cluster_name": cluster_name,
                    "cluster_basis": group["basis"],
                    "signal_count": len(packets),
                    "covered_signals": covered_signals,
                    "covered_signal_names": covered_names,
                    "top_signal_names": covered_names[:8],
                    "source_types": source_types,
                    "source_count_total": sum(int(packet.get("source_count", 0) or 0) for packet in packets),
                    "evidence_count_total": sum(int(packet.get("evidence_count", 0) or 0) for packet in packets),
                    "score_max": max(scores) if scores else 0.0,
                    "score_avg": sum(scores) / len(scores) if scores else 0.0,
                    "mechanisms": mechanisms,
                    "buckets": buckets,
                    "representative_signals": [
                        {
                            "signal_id": packet.get("signal_id"),
                            "name": packet.get("name"),
                            "score": packet.get("score"),
                            "primary_evidence_ids": packet.get("primary_evidence_ids", []),
                        }
                        for packet in representative_packets
                    ],
                    "signal_ids": [packet.get("signal_id") for packet in packets if packet.get("signal_id")],
                }
            )

        clusters.sort(
            key=lambda item: (
                int(item.get("signal_count", 0) or 0),
                float(item.get("score_max", 0.0) or 0.0),
                int(item.get("evidence_count_total", 0) or 0),
            ),
            reverse=True,
        )
        for idx, cluster in enumerate(clusters, 1):
            cluster["cluster_id"] = f"WC{idx}"
        return clusters

    def _build_prompt_ready_packets(self, packets: Dict[str, Any]) -> Dict[str, Any]:
        def shrink(packet_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            compact = []
            for packet in packet_list:
                compact.append(
                    {
                        "signal_id": packet["signal_id"],
                        "signal_kind": packet["signal_kind"],
                        "name": packet["name"],
                        "score": packet["score"],
                        "mechanism_core": packet["mechanism_core"],
                        "signal_bucket": packet.get("signal_bucket", ""),
                        "stage_hypothesis": packet.get("stage_hypothesis", ""),
                        "source_types": packet["source_types"],
                        "source_count": packet["source_count"],
                        "evidence_count": packet["evidence_count"],
                        "org_count": packet["org_count"],
                        "time_validation": packet["time_validation"],
                        "validation": packet["validation"],
                        "reverse_validation": packet.get("reverse_validation", {}),
                        "family": packet.get("family", {}),
                        "evidence_quality": packet.get("evidence_quality", {}),
                        "tech_chain_mapping": packet.get("tech_chain_mapping", {}),
                        "temporal_validation": packet.get("temporal_validation", {}),
                        "key_core_potential": packet.get("key_core_potential", {}),
                        "chain_summary": packet.get("chain_summary", ""),
                        "timeline_summary": packet.get("timeline_summary", ""),
                        "organization_summary": packet.get("organization_summary", ""),
                        "evidence_highlights": packet.get("evidence_highlights", []),
                        "evidence_items": [
                            {
                                "evidence_id": item["evidence_id"],
                                "source_type": item["source_type"],
                                "org": item["org"],
                                "date": item["date"],
                                "title": item["title"],
                                "snippet": item["snippet"],
                                "raw_candidate_text": item["raw_candidate_text"],
                                "event_quality_score": item.get("event_quality_score", 0.0),
                                "event_quality_tier": item.get("event_quality_tier", ""),
                            }
                            for item in packet["evidence_items"]
                        ],
                    }
                )
            return compact

        def shrink_clusters(cluster_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            compact = []
            for cluster in cluster_list:
                compact.append(
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "cluster_name": cluster.get("cluster_name"),
                        "cluster_basis": cluster.get("cluster_basis"),
                        "signal_count": cluster.get("signal_count"),
                        "covered_signals": cluster.get("covered_signals", []),
                        "top_signal_names": cluster.get("top_signal_names", []),
                        "covered_signal_names": cluster.get("covered_signal_names", []),
                        "source_types": cluster.get("source_types", []),
                        "evidence_count_total": cluster.get("evidence_count_total", 0),
                        "source_count_total": cluster.get("source_count_total", 0),
                        "score_max": cluster.get("score_max", 0.0),
                        "mechanisms": cluster.get("mechanisms", []),
                        "representative_signals": cluster.get("representative_signals", []),
                    }
                )
            return compact

        return {
            "generated_at": packets.get("generated_at"),
            "summary": packets.get("summary", {}),
            "family_summary": packets.get("family_summary", {}),
            "key_core_candidates": packets.get("key_core_candidates", []),
            "weak_signals": shrink(packets.get("weak_signals", [])[:12]),
            "weak_signal_clusters": shrink_clusters(packets.get("weak_signal_clusters", [])),
            "near_strong_signals": shrink(packets.get("near_strong_signals", [])),
            "scope_overviews": packets.get("scope_overviews", []),
        }

    def _extract_first_json(self, text: str) -> Dict[str, Any]:
        cleaned = self._safe_report_text(text)
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        raise ValueError("LLM response does not contain a valid JSON object")

    def _grounded_extract_facts(self, packets: Dict[str, Any]) -> Dict[str, Any]:
        prompt_packets = self._build_prompt_ready_packets(packets)
        prompt = (
            "请仅基于下面的证据包生成一份严格 grounded 的 JSON。\n"
            "要求：\n"
            "1. 只能使用证据包中的信息，禁止补充新的机构、时间、数字、产品名、政策名或结论。\n"
            "2. executive_summary、overall_findings、signal_analyses.key_facts、signal_analyses.why_it_matters、trend_implications 中的每条陈述都必须绑定 evidence_ids。\n"
            "3. 优先总结证据链体现的环节变化、时间延续、代表性场景或动作、参与主体，但不要机械罗列全部证据。\n"
            "4. 如果证据不足，请写“无法仅凭现有证据判断”。\n"
            "5. uncertainties 和 next_observations 可以不带 evidence_ids，但必须保持保守。\n"
            "6. 只输出 JSON，不要输出 Markdown。\n\n"
            "输出 JSON schema：\n"
            "{\n"
            '  "executive_summary": {"summary": "string", "signal_ids": ["WS1"], "evidence_ids": ["WS1-E1"]},\n'
            '  "overall_findings": [{"statement": "string", "signal_ids": ["WS1"], "evidence_ids": ["WS1-E1"]}],\n'
            '  "signal_analyses": [\n'
            "    {\n"
            '      "signal_id": "WS1",\n'
            '      "key_facts": [{"statement": "string", "evidence_ids": ["WS1-E1"]}],\n'
            '      "why_it_matters": [{"statement": "string", "evidence_ids": ["WS1-E1"]}],\n'
            '      "uncertainties": ["string"],\n'
            '      "next_observations": ["string"]\n'
            "    }\n"
            "  ],\n"
            '  "trend_implications": [{"statement": "string", "signal_ids": ["WS1"], "evidence_ids": ["WS1-E1"]}],\n'
            '  "next_steps": ["string"],\n'
            '  "report_limits": ["string"]\n'
            "}\n\n"
            "证据包：\n"
            f"{json.dumps(prompt_packets, ensure_ascii=False, indent=2)}"
        )
        report_model = os.getenv("REPORT_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
        print(f"  [报告生成] 使用模型: {report_model}")
        print("  [报告生成] 正在提取 grounded facts...")
        result, usage_info, _ = chat_text(
            prompt,
            system="你是一名严格遵守证据边界的技术情报分析师。任何未被证据支持的内容都必须舍弃。",
            model=report_model,
            temperature=0.0,
            max_tokens=1800,
            timeout=90,
        )
        print(f"  [报告生成] LLM响应长度: {len(result)} 字符")
        print(f"  [报告生成] Token使用: prompt={usage_info['prompt_tokens']}, completion={usage_info['completion_tokens']}")
        if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
            record_call(
                call_type="报告生成",
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
                success=True,
            )
        self.report_artifacts["report_llm_raw_response"] = result
        return self._extract_first_json(result)

    def _sanitize_text_items(self, values: Any, max_items: int = 4) -> List[str]:
        cleaned = []
        for value in self._safe_report_list(values):
            text = self._truncate_report_text(value, 180)
            if text and text not in cleaned:
                cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _sanitize_statement_items(
        self,
        entries: Any,
        valid_signal_ids: set,
        valid_evidence_ids: set,
        signal_evidence_ids: Optional[set] = None,
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        cleaned = []
        seen = set()
        for entry in self._safe_report_list(entries):
            if not isinstance(entry, dict):
                continue
            statement = self._truncate_report_text(
                entry.get("statement") or entry.get("claim") or entry.get("finding") or entry.get("summary"),
                220,
            )
            evidence_ids = [
                self._safe_report_text(item)
                for item in self._safe_report_list(entry.get("evidence_ids", []))
            ]
            evidence_ids = [item for item in evidence_ids if item in valid_evidence_ids]
            if signal_evidence_ids is not None:
                evidence_ids = [item for item in evidence_ids if item in signal_evidence_ids]
            signal_ids = [
                self._safe_report_text(item)
                for item in self._safe_report_list(entry.get("signal_ids", []))
            ]
            signal_ids = [item for item in signal_ids if item in valid_signal_ids]
            if not statement or not evidence_ids:
                continue
            dedupe_key = (statement, tuple(evidence_ids))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned.append(
                {
                    "statement": statement,
                    "signal_ids": signal_ids,
                    "evidence_ids": evidence_ids,
                }
            )
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _grounded_build_default_facts(self, packets: Dict[str, Any]) -> Dict[str, Any]:
        weak_packets = packets.get("weak_signals", [])
        near_packets = packets.get("near_strong_signals", [])
        top_names = [packet["name"] for packet in weak_packets[:3]]
        summary_signal_ids = [packet["signal_id"] for packet in weak_packets[:3]]
        summary_evidence_ids = []
        for packet in weak_packets[:2]:
            summary_evidence_ids.extend(packet.get("primary_evidence_ids", [])[:1])

        if weak_packets:
            summary_text = (
                f"本轮识别出 {len(weak_packets)} 个弱信号，另有 {len(near_packets)} 个近强信号。"
                f"当前最值得优先跟踪的方向包括：{'、'.join(top_names) if top_names else '暂无明确方向'}。"
            )
            summary_text += self._compose_cross_signal_chain_summary(weak_packets + near_packets)
        elif near_packets:
            summary_text = f"本轮尚未形成明确弱信号，但筛出 {len(near_packets)} 个近强信号，可继续观察其证据累积。"
        else:
            summary_text = "本轮未形成可支撑报告的弱信号或近强信号。"

        overall_findings = []
        for packet in weak_packets[:3]:
            evidence_ids = packet.get("primary_evidence_ids") or [item["evidence_id"] for item in packet.get("evidence_items", [])[:1]]
            if not evidence_ids:
                continue
            finding_parts = []
            chain_summary = self._safe_report_text(packet.get("chain_summary")).rstrip("。")
            timeline_summary = self._safe_report_text(packet.get("timeline_summary")).rstrip("。")
            org_summary = self._safe_report_text(packet.get("organization_summary")).rstrip("。")
            if chain_summary:
                finding_parts.append(f"{packet['name']} {chain_summary}")
            else:
                finding_parts.append(f"{packet['name']} 已进入值得持续跟踪的观察范围")
            if timeline_summary:
                finding_parts.append(timeline_summary)
            elif org_summary:
                finding_parts.append(org_summary)
            overall_findings.append(
                {
                    "statement": "，".join(part for part in finding_parts if part) + "。",
                    "signal_ids": [packet["signal_id"]],
                    "evidence_ids": evidence_ids,
                }
            )

        signal_analyses = []
        for packet in weak_packets:
            signal_evidence_ids = [item["evidence_id"] for item in packet.get("evidence_items", [])]
            primary_ids = packet.get("primary_evidence_ids") or signal_evidence_ids[:2]
            key_facts = []
            if primary_ids:
                source_label = self._source_type_text(packet.get("source_types", []), fallback="相关来源")
                key_facts.append(
                    {
                        "statement": (
                            f"当前线索主要指向{packet['mechanism_core']}，并已在{source_label}等来源中出现；"
                            f"{self._safe_report_text(packet.get('chain_summary')).rstrip('。')}。"
                        ),
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )
            highlight_text = self._compact_report_paragraph(packet.get("evidence_highlights", []), max_items=2)
            if highlight_text and primary_ids:
                key_facts.append(
                    {
                        "statement": highlight_text,
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )
            timeline_summary = self._safe_report_text(packet.get("timeline_summary"))
            if timeline_summary and primary_ids:
                key_facts.append(
                    {
                        "statement": timeline_summary,
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )
            organization_summary = self._safe_report_text(packet.get("organization_summary"))
            if organization_summary and primary_ids:
                key_facts.append(
                    {
                        "statement": organization_summary,
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )
            quality_statement = self._safe_report_text(self._compose_packet_quality_text(packet))
            if quality_statement and primary_ids:
                key_facts.append(
                    {
                        "statement": quality_statement,
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )
            tech_chain_statement = self._safe_report_text(self._compose_packet_tech_chain_text(packet))
            if tech_chain_statement and primary_ids:
                key_facts.append(
                    {
                        "statement": tech_chain_statement,
                        "signal_ids": [packet["signal_id"]],
                        "evidence_ids": primary_ids,
                    }
                )

            why_it_matters = []
            if primary_ids:
                if packet["validation"]["multi_source_validated"] or packet["validation"]["semantic_validated"] or packet["validation"]["event_doc_semantic_validated"]:
                    why_it_matters.append(
                        {
                            "statement": "现有线索已不只是单点讨论，而是开始出现研究、产业或工程环节之间的联动，更适合作为重点跟踪对象。",
                            "signal_ids": [packet["signal_id"]],
                            "evidence_ids": primary_ids,
                        }
                    )
                else:
                    why_it_matters.append(
                        {
                            "statement": "当前信号仍处于早期观察阶段，但已具备可追溯文本证据，可作为后续验证的起点。",
                            "signal_ids": [packet["signal_id"]],
                            "evidence_ids": primary_ids,
                        }
                    )

            uncertainties = []
            if packet["source_count"] < 2:
                uncertainties.append("当前跨源支撑仍偏弱，后续需要补充更多不同来源的证据。")
            if "patent" not in packet.get("source_types", []) and packet["source_count"] >= 2:
                uncertainties.append("目前尚未看到足够明确的工程布局或专利侧证据。")
            if not packet["time_validation"]["validated"]:
                uncertainties.append("时间延续性尚未充分验证，暂不宜将其判断为稳定趋势。")
            if packet["validation"]["traceable_ratio"] < 0.5:
                uncertainties.append("可追溯证据占比有限，需要继续补充带时间或来源标识的文本。")
            mapping = packet.get("tech_chain_mapping", {}) or {}
            if self._safe_report_text(mapping.get("mapping_relation")) in {"no_match", "broader_match"}:
                uncertainties.append("技术链映射仍需复核，暂不宜作为关键核心技术结论。")
            if not uncertainties:
                uncertainties.append("现有证据已形成初步支撑，但仍需继续观察其后续扩散情况。")

            next_observations = [self._build_signal_followup_priority(packet)]
            if "news" not in packet.get("source_types", []):
                next_observations.append("继续观察是否出现资讯或产业动态侧佐证。")
            next_observations.append("跟踪后续月份中是否继续出现相同主题的新增证据。")

            signal_analyses.append(
                {
                    "signal_id": packet["signal_id"],
                    "key_facts": key_facts,
                    "why_it_matters": why_it_matters,
                    "uncertainties": uncertainties[:4],
                    "next_observations": next_observations[:4],
                }
            )

        trend_implications = []
        for packet in weak_packets[:2]:
            evidence_ids = packet.get("primary_evidence_ids") or [item["evidence_id"] for item in packet.get("evidence_items", [])[:1]]
            if not evidence_ids:
                continue
            source_set = {item for item in packet.get("source_types", []) if item}
            if "paper" in source_set and "patent" in source_set and ("news" in source_set or "report" in source_set):
                statement = (
                    f"{packet['name']} 的证据链已覆盖研究验证、产业关注与工程布局，"
                    "后续若继续扩散到更多主体和场景，可能进一步演化为稳定热点。"
                )
            elif "paper" in source_set and ("news" in source_set or "report" in source_set):
                statement = (
                    f"{packet['name']} 已出现由研究验证向产业关注扩散的迹象，"
                    "后续需观察是否补齐工程布局或规模化应用证据。"
                )
            elif "news" in source_set or "report" in source_set:
                statement = (
                    f"{packet['name']} 当前更多体现为产业讨论与场景观察升温，"
                    "后续需关注是否获得更强的研究或工程侧支撑。"
                )
            else:
                statement = f"{packet['name']} 可作为当前技术观察范围中的前沿方向之一，但其趋势判断仍应以新增证据持续出现为前提。"
            trend_implications.append(
                {
                    "statement": statement,
                    "signal_ids": [packet["signal_id"]],
                    "evidence_ids": evidence_ids,
                }
            )

        next_steps = [self._build_signal_followup_priority(packet) for packet in weak_packets[:2]]
        next_steps.append("将高优先级方向回流到人工复核流程，核查是否存在误并类或主题过宽问题。")
        deduped_next_steps = []
        for step in next_steps:
            if step and step not in deduped_next_steps:
                deduped_next_steps.append(step)

        return {
            "executive_summary": {
                "summary": summary_text,
                "signal_ids": summary_signal_ids,
                "evidence_ids": summary_evidence_ids,
            },
            "overall_findings": overall_findings,
            "signal_analyses": signal_analyses,
            "trend_implications": trend_implications,
            "next_steps": deduped_next_steps or [
                "优先补充弱信号对应的跨源文本，尤其是能提供时间与来源标识的证据。",
                "对排名靠前的弱信号建立连续监测，观察未来数周或数月的新增证据密度。",
                "将高优先级方向回流到人工复核流程，核查是否存在误并类或主题过宽问题。",
            ],
            "report_limits": [
                "本报告仅基于当前输入数据与已保留的证据片段生成，不代表该方向已经进入确定性趋势阶段。",
                "如果某一信号缺乏跨源、跨时间或工程实现侧证据，应将其视为待观察对象而非既定结论。",
            ],
        }

    def _grounded_sanitize_facts(self, extracted: Dict[str, Any], packets: Dict[str, Any]) -> Dict[str, Any]:
        fallback = self._grounded_build_default_facts(packets)
        if not isinstance(extracted, dict):
            return fallback

        all_signal_ids = set(packets.get("signal_lookup", {}).keys())
        valid_evidence_ids = set(packets.get("evidence_lookup", {}).keys())
        weak_signal_ids = [packet["signal_id"] for packet in packets.get("weak_signals", [])]
        packet_by_id = {packet["signal_id"]: packet for packet in packets.get("weak_signals", [])}

        summary_entry = extracted.get("executive_summary", {}) if isinstance(extracted.get("executive_summary", {}), dict) else {}
        summary_text = self._truncate_report_text(summary_entry.get("summary"), 260)
        summary_evidence_ids = [
            self._safe_report_text(item)
            for item in self._safe_report_list(summary_entry.get("evidence_ids", []))
        ]
        summary_evidence_ids = [item for item in summary_evidence_ids if item in valid_evidence_ids]
        if summary_text and summary_evidence_ids:
            sanitized_summary = {
                "summary": summary_text,
                "signal_ids": [
                    self._safe_report_text(item)
                    for item in self._safe_report_list(summary_entry.get("signal_ids", []))
                    if self._safe_report_text(item) in all_signal_ids
                ],
                "evidence_ids": summary_evidence_ids,
            }
        else:
            sanitized_summary = fallback["executive_summary"]

        overall_findings = self._sanitize_statement_items(
            extracted.get("overall_findings", []),
            all_signal_ids,
            valid_evidence_ids,
            max_items=6,
        ) or fallback["overall_findings"]

        signal_analyses = {}
        for entry in self._safe_report_list(extracted.get("signal_analyses", [])):
            if not isinstance(entry, dict):
                continue
            signal_id = self._safe_report_text(entry.get("signal_id"))
            packet = packet_by_id.get(signal_id)
            if packet is None:
                continue
            signal_evidence_ids = {item["evidence_id"] for item in packet.get("evidence_items", [])}
            signal_analyses[signal_id] = {
                "signal_id": signal_id,
                "key_facts": self._sanitize_statement_items(
                    entry.get("key_facts", []),
                    all_signal_ids,
                    valid_evidence_ids,
                    signal_evidence_ids=signal_evidence_ids,
                    max_items=4,
                ),
                "why_it_matters": self._sanitize_statement_items(
                    entry.get("why_it_matters", []),
                    all_signal_ids,
                    valid_evidence_ids,
                    signal_evidence_ids=signal_evidence_ids,
                    max_items=3,
                ),
                "uncertainties": self._sanitize_text_items(entry.get("uncertainties", []), max_items=4),
                "next_observations": self._sanitize_text_items(entry.get("next_observations", []), max_items=4),
            }
        for entry in fallback["signal_analyses"]:
            signal_analyses.setdefault(entry["signal_id"], entry)

        trend_implications = self._sanitize_statement_items(
            extracted.get("trend_implications", []),
            all_signal_ids,
            valid_evidence_ids,
            max_items=4,
        ) or fallback["trend_implications"]

        return {
            "executive_summary": sanitized_summary,
            "overall_findings": overall_findings,
            "signal_analyses": [signal_analyses[signal_id] for signal_id in weak_signal_ids if signal_id in signal_analyses],
            "trend_implications": trend_implications,
            "next_steps": self._sanitize_text_items(extracted.get("next_steps", []), max_items=5) or fallback["next_steps"],
            "report_limits": self._sanitize_text_items(extracted.get("report_limits", []), max_items=5) or fallback["report_limits"],
        }

    def _grounded_build_check(self, facts: Dict[str, Any], packets: Dict[str, Any]) -> Dict[str, Any]:
        weak_signal_ids = [packet["signal_id"] for packet in packets.get("weak_signals", [])]
        evidence_lookup = packets.get("evidence_lookup", {})
        referenced_evidence = set()
        signals_with_cited_facts = set()

        referenced_evidence.update(facts.get("executive_summary", {}).get("evidence_ids", []))
        for entry in facts.get("overall_findings", []):
            referenced_evidence.update(entry.get("evidence_ids", []))
        for entry in facts.get("trend_implications", []):
            referenced_evidence.update(entry.get("evidence_ids", []))
        for analysis in facts.get("signal_analyses", []):
            signal_id = analysis.get("signal_id")
            fact_count = 0
            for block in analysis.get("key_facts", []) + analysis.get("why_it_matters", []):
                evidence_ids = [item for item in block.get("evidence_ids", []) if item in evidence_lookup]
                if evidence_ids:
                    referenced_evidence.update(evidence_ids)
                    fact_count += 1
            if signal_id and fact_count > 0:
                signals_with_cited_facts.add(signal_id)

        missing_signal_ids = [signal_id for signal_id in weak_signal_ids if signal_id not in signals_with_cited_facts]
        return {
            "weak_signal_count": len(weak_signal_ids),
            "signal_analyses_count": len(facts.get("signal_analyses", [])),
            "referenced_evidence_count": len(referenced_evidence),
            "missing_signal_ids": missing_signal_ids,
            "passed": (not weak_signal_ids) or (not missing_signal_ids and len(referenced_evidence) > 0),
        }

    @staticmethod
    def _format_citations(evidence_ids: List[str]) -> str:
        ids = [str(item).strip() for item in evidence_ids if str(item).strip()]
        return f" [证据: {', '.join(ids)}]" if ids else ""

    @staticmethod
    def _source_type_label(value: Any) -> str:
        key = str(value).strip().lower()
        mapping = {
            "paper": "论文",
            "literature": "论文",
            "report": "研报",
            "news": "资讯",
            "patent": "专利",
            "company": "企业",
            "unknown": "多源文本",
        }
        return mapping.get(key, str(value).strip() or "多源文本")

    def _source_type_text(self, values: List[Any], fallback: str = "多源文本") -> str:
        labels = []
        for value in values:
            label = self._source_type_label(value)
            if label and label not in labels:
                labels.append(label)
        return "、".join(labels) if labels else fallback

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in str(text))

    @staticmethod
    def _ascii_ratio(text: str) -> float:
        text = str(text or "")
        if not text:
            return 0.0
        ascii_count = sum(1 for char in text if ord(char) < 128)
        return ascii_count / max(len(text), 1)

    def _is_noisy_report_phrase(self, text: Any) -> bool:
        normalized = self._safe_report_text(text)
        if not normalized:
            return True
        lowered = normalized.lower()
        tokens = [token for token in re.split(r"[\s/_-]+", lowered) if token]
        generic_tokens = {
            "robot", "control", "planning", "training", "simulation", "visual", "sensor",
            "network", "deep", "policy", "environment", "perception", "system",
        }
        if "..." in normalized:
            return True
        if not self._contains_cjk(normalized) and self._ascii_ratio(normalized) >= 0.85 and len(tokens) >= 3:
            return True
        if len(tokens) >= 3 and sum(1 for token in tokens if token in generic_tokens) >= 2:
            return True
        return False

    def _is_generic_news_lead(self, text: Any) -> bool:
        normalized = self._safe_report_text(text)
        if not normalized:
            return True
        patterns = [
            r"^\d{4}年\d{1,2}月\d{1,2}日",
            r"^\d{1,2}月\d{1,2}日",
            r"^\d{1,2}月\d{1,2}日下午",
            r"^(近日|日前|近年来|当前|在.+当下|随着|会上|作为|据了解|【编者按】)",
        ]
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _normalize_org_names(self, values: List[Any], max_items: int = 4) -> List[str]:
        cleaned = []
        for value in values:
            text = self._safe_report_text(value)
            if not text or text.lower() == "unknown" or text in cleaned:
                continue
            cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    def _parse_report_date(self, value: Any) -> Optional[datetime]:
        text = self._safe_report_text(value)
        if not text:
            return None
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        try:
            return parsed.to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _format_report_date(value: Optional[datetime]) -> str:
        if value is None:
            return ""
        return value.strftime("%Y年%m月")

    def _report_mechanism_text(self, packet: Dict[str, Any]) -> str:
        mechanism = self._safe_report_text(packet.get("mechanism_core"))
        mapping = {
            "control": "控制能力",
            "planning": "任务规划能力",
            "training": "训练方法",
            "simulation": "仿真能力",
            "retrieval": "检索能力",
            "memory": "记忆能力",
            "alignment": "对齐能力",
            "reasoning": "推理能力",
            "perception": "感知能力",
            "grounding": "具身落地能力",
        }
        if not mechanism:
            return ""
        lowered = mechanism.lower()
        if lowered in mapping:
            return mapping[lowered]
        if self._is_noisy_report_phrase(mechanism):
            return ""
        return mechanism

    def _extract_evidence_fragment(self, item: Dict[str, Any], limit: int = 34) -> str:
        source_type = self._safe_report_text(item.get("source_type")).lower()
        if source_type in {"news", "report"}:
            key_order = ["title", "snippet", "raw_candidate_text", "text"]
        elif source_type in {"paper", "patent"}:
            key_order = ["title", "snippet", "text", "raw_candidate_text"]
        else:
            key_order = ["snippet", "title", "raw_candidate_text", "text"]

        for key in key_order:
            text = self._safe_report_text(item.get(key))
            if not text:
                continue
            text = re.sub(r"\s+", " ", text).strip().strip("“”\"'")
            clause = re.split(r"[。；;!?！？]", text)[0].strip() or text
            if len(clause) > limit + 8:
                shorter = re.split(r"[，,:：]", clause)[0].strip()
                if shorter:
                    clause = shorter
            clause = self._truncate_report_text(clause, limit).rstrip("。；;，, ")
            if self._is_generic_news_lead(clause):
                continue
            if clause and not self._is_noisy_report_phrase(clause):
                return clause
        return ""

    def _build_packet_evidence_highlights(self, packet: Dict[str, Any], max_items: int = 2) -> List[str]:
        candidates = []
        for item in packet.get("evidence_items", []):
            fragment = self._extract_evidence_fragment(item)
            if not fragment:
                continue
            source_label = self._source_type_label(item.get("source_type"))
            candidates.append((source_label, f"{source_label}侧线索提到“{fragment}”"))

        selected = []
        seen = set()
        used_sources = set()
        for source_label, text in candidates:
            if text in seen or source_label in used_sources:
                continue
            selected.append(text)
            seen.add(text)
            used_sources.add(source_label)
            if len(selected) >= max_items:
                return selected
        for _, text in candidates:
            if text in seen:
                continue
            selected.append(text)
            seen.add(text)
            if len(selected) >= max_items:
                break
        return selected

    def _build_packet_chain_summary(self, packet: Dict[str, Any]) -> str:
        source_set = {item for item in packet.get("source_types", []) if item}
        stages = []
        if "paper" in source_set:
            stages.append("研究验证")
        if "news" in source_set or "report" in source_set:
            stages.append("产业关注")
        if "patent" in source_set:
            stages.append("工程布局")

        if len(stages) >= 3:
            summary = "证据链已覆盖研究验证、产业关注与工程布局三个环节"
        elif len(stages) == 2:
            summary = f"证据链已覆盖{stages[0]}与{stages[1]}两个环节"
        elif len(stages) == 1:
            summary = f"当前证据主要集中在{stages[0]}环节"
        else:
            summary = "当前证据仍以零散线索累积为主"

        validation = packet.get("validation", {})
        if validation.get("event_doc_semantic_validated"):
            summary += "，研究与产业文本之间已形成较好的相互印证"
        elif validation.get("multi_source_validated") or validation.get("semantic_validated"):
            summary += "，跨源信息之间已形成初步印证"

        reverse_validation = packet.get("reverse_validation", {})
        reverse_status = self._safe_report_text(reverse_validation.get("status"))
        if reverse_status == "natural_topic_in_source":
            summary += "，原始证据中已能找到较自然的小主题表达"
        elif reverse_status == "only_upper_topic_in_source":
            summary += "，但原始证据仍偏向上层母主题表述"

        family_info = packet.get("family", {})
        family_consistency = self._safe_report_text(family_info.get("semantic_consistency"))
        if family_consistency == "high":
            summary += "，对象族语义承接较稳定"
        elif family_consistency == "medium":
            summary += "，对象族承接已初步稳定"

        return summary + "。"

    def _build_packet_timeline_summary(self, packet: Dict[str, Any]) -> str:
        parsed_dates = []
        for item in packet.get("evidence_items", []):
            parsed = self._parse_report_date(item.get("date"))
            if parsed is not None:
                parsed_dates.append(parsed)
        if parsed_dates:
            start = min(parsed_dates)
            end = max(parsed_dates)
            start_label = self._format_report_date(start)
            end_label = self._format_report_date(end)
            if start_label and end_label and start_label != end_label:
                if packet.get("time_validation", {}).get("validated"):
                    return f"可追溯时间线已从{start_label}延续到{end_label}。"
                return f"相关线索主要分布在{start_label}至{end_label}期间。"
            if end_label:
                return f"当前可追溯线索主要集中在{end_label}。"

        time_validation = packet.get("time_validation", {})
        if time_validation.get("validated"):
            return "从时间分布看，该方向已出现持续迹象。"
        if time_validation.get("recent_count", 0) > 0:
            return "当前线索更多集中在近期，后续仍需观察其延续性。"
        return ""

    def _build_packet_org_summary(self, packet: Dict[str, Any], max_items: int = 3) -> str:
        orgs = self._normalize_org_names(packet.get("orgs", []), max_items=max_items)
        if not orgs:
            return ""
        return f"相关线索已涉及{'、'.join(orgs)}等主体。"

    def _collect_packet_orgs(self, packets: List[Dict[str, Any]], max_items: int = 4) -> List[str]:
        orgs = []
        for packet in packets:
            for org in self._normalize_org_names(packet.get("orgs", []), max_items=max_items):
                if org in orgs:
                    continue
                orgs.append(org)
                if len(orgs) >= max_items:
                    return orgs
        return orgs

    def _compose_cross_signal_chain_summary(self, packets: List[Dict[str, Any]]) -> str:
        if not packets:
            return ""
        source_sets = [{item for item in packet.get("source_types", []) if item} for packet in packets]
        if any("paper" in source_set and "patent" in source_set and ("news" in source_set or "report" in source_set) for source_set in source_sets):
            return "从证据链结构看，部分方向已同时出现研究验证、产业关注与工程布局信号。"
        if any("paper" in source_set and ("news" in source_set or "report" in source_set) for source_set in source_sets):
            return "从证据链结构看，部分方向已呈现研究验证向产业关注扩散的迹象。"
        if any("patent" in source_set and ("news" in source_set or "report" in source_set) for source_set in source_sets):
            return "从证据链结构看，部分方向已出现产业关注与工程布局并行的特征。"
        return "从证据链结构看，当前仍以早期线索累积和多源补证为主。"

    def _build_signal_followup_priority(self, packet: Dict[str, Any]) -> str:
        source_set = {item for item in packet.get("source_types", []) if item}
        if "patent" not in source_set and ("paper" in source_set or "news" in source_set or "report" in source_set):
            return f"重点补充{packet['name']}的专利或工程实现侧证据，观察其是否从讨论走向布局。"
        if not packet.get("time_validation", {}).get("validated"):
            return f"持续跟踪{packet['name']}在后续月份是否连续出现，确认其是否具备稳定延续性。"
        if packet.get("source_count", 0) < 2:
            return f"继续观察{packet['name']}是否扩散到更多来源，避免仅凭单源线索放大判断。"
        return f"围绕{packet['name']}持续补充跨源文本与主体信息，提升趋势判断的稳定性。"

    def _compose_report_intro(self, packets: Dict[str, Any]) -> str:
        weak_packets = packets.get("weak_signals", [])
        weak_clusters = packets.get("weak_signal_clusters", [])
        near_packets = packets.get("near_strong_signals", [])
        summary = packets.get("summary", {})
        top_names = [packet["name"] for packet in weak_packets[:3]]
        if weak_packets:
            cluster_text = (
                f"这些线索进一步归并为 {len(weak_clusters)} 个弱信号簇，报告正文将以簇为主线覆盖全部弱信号。"
                if weak_clusters and len(weak_packets) > 8
                else ""
            )
            parts = [
                "本期围绕多源文本中识别出的前沿线索进行梳理。",
                f"综合比对结果显示，{'、'.join(top_names) if top_names else '相关方向'}已形成较清晰的观察对象。",
                cluster_text,
                self._compose_cross_signal_chain_summary(weak_packets + near_packets),
            ]
            return " ".join(part for part in parts if part).strip()
        if near_packets:
            return (
                f"本期尚未形成稳定弱信号，但已筛出 {summary.get('near_strong_count', len(near_packets))} 个近强信号。"
                "从现有证据看，相关方向已具备继续跟踪的现实基础。"
            )
        return "本期样本尚不足以支撑形成稳定的前沿技术判断，后续仍需继续补充跨源、跨时间的可追溯证据。"

    def _compose_signal_position(self, packet: Dict[str, Any]) -> str:
        source_set = {item for item in packet.get("source_types", []) if item}
        if "paper" in source_set and "patent" in source_set and ("news" in source_set or "report" in source_set):
            return "从证据链结构看，该方向已由研究验证延伸至产业关注，并开始出现工程布局信号。"
        if "paper" in source_set and ("news" in source_set or "report" in source_set):
            return "从证据链结构看，该方向正由研究验证向产业场景扩散。"
        if "patent" in source_set and ("news" in source_set or "report" in source_set):
            return "从证据链结构看，该方向已出现产业关注与工程布局并行推进的特征。"
        if "news" in source_set or "report" in source_set:
            return "从证据链结构看，当前产业侧对相关场景和能力边界的关注正在升温。"
        if "paper" in source_set:
            return "从证据链结构看，当前仍以研究验证和技术探索信号为主。"
        if "patent" in source_set:
            return "从证据链结构看，当前以工程布局和方法申请信号为主。"
        return "从证据链结构看，当前仍以早期线索累积为主。"

    def _compose_packet_evidence_narrative(self, packet: Dict[str, Any], max_items: int = 3) -> str:
        statements = []
        seen_sources = set()
        for item in packet.get("evidence_items", []):
            source_label = self._source_type_label(item.get("source_type"))
            if source_label in seen_sources:
                continue
            fragment = self._extract_evidence_fragment(item, limit=26)
            if fragment:
                if source_label == "论文":
                    statements.append(f"论文线索主要围绕“{fragment}”展开")
                elif source_label == "资讯":
                    statements.append(f"产业资讯已出现“{fragment}”等动向")
                elif source_label == "研报":
                    statements.append(f"研报侧已开始围绕“{fragment}”进行跟踪判断")
                elif source_label == "专利":
                    statements.append(f"专利文本显示“{fragment}”等工程布局信号")
                else:
                    statements.append(f"{source_label}线索提到“{fragment}”")
            else:
                if source_label == "论文":
                    statements.append(f"论文侧已出现与{packet['name']}相关的研究验证线索")
                elif source_label == "资讯":
                    statements.append(f"产业资讯已出现与{packet['name']}相关的应用或部署动向")
                elif source_label == "研报":
                    statements.append(f"研报侧已将{packet['name']}纳入持续观察范围")
                elif source_label == "专利":
                    statements.append(f"专利文本中已出现与{packet['name']}相关的工程布局信号")
                else:
                    statements.append(f"{source_label}侧已出现与{packet['name']}相关的线索")
            seen_sources.add(source_label)
            if len(statements) >= max_items:
                break
        if not statements:
            return ""
        return "；".join(statements) + "。"

    def _compose_packet_quality_text(self, packet: Dict[str, Any]) -> str:
        quality = packet.get("evidence_quality", {}) or {}
        average = self._safe_report_float(quality.get("candidate_evidence_quality"), 0.0)
        core_average = self._safe_report_float(quality.get("candidate_core_evidence_quality"), 0.0)
        low_ratio = self._safe_report_float(quality.get("low_quality_evidence_ratio"), 0.0)
        excluded_count = int(self._safe_report_float(quality.get("excluded_low_quality_evidence_count"), 0.0))
        if average <= 0 and excluded_count <= 0:
            return ""

        parts = []
        if average > 0:
            parts.append(f"证据质量均值约 {average:.1f}")
        if core_average > 0:
            parts.append(f"核心证据均值约 {core_average:.1f}")
        if low_ratio > 0:
            parts.append(f"低质量证据占比约 {low_ratio:.0%}")
        text = "，".join(parts)
        if excluded_count > 0:
            text = (text + "；" if text else "") + f"{excluded_count} 条低于 4 分的证据未作为报告核心证据"
        return text + "。" if text else ""

    def _compose_packet_tech_chain_text(self, packet: Dict[str, Any]) -> str:
        mapping = packet.get("tech_chain_mapping", {}) or {}
        relation = self._safe_report_text(mapping.get("mapping_relation"))
        method = self._safe_report_text(mapping.get("mapping_method"))
        name = self._safe_report_text(mapping.get("tech_chain_name") or mapping.get("tech_chain_official_name"))
        if not name or relation == "no_match" or method == "no_match":
            return "技术链映射尚未命中明确节点，后续应优先补充术语表或人工复核。"

        confidence = self._safe_report_float(mapping.get("mapping_confidence"), 0.0)
        bottleneck = self._safe_report_text(mapping.get("bottleneck_level")) or "unknown"
        strategic = self._safe_report_text(mapping.get("strategic_importance_level")) or "unknown"
        relation_text = {
            "exact_match": "精确匹配",
            "close_match": "近似匹配",
            "broader_match": "上位方向匹配",
            "narrower_match": "下位对象匹配",
            "related_match": "相关匹配",
            "manual_override": "人工指定",
        }.get(relation, relation)
        text = (
            f"技术链上可映射到“{name}”，关系为{relation_text}，"
            f"置信度约 {confidence:.2f}，卡点等级为 {bottleneck}、战略重要性为 {strategic}"
        )
        if relation == "broader_match" or (method == "semantic_match" and confidence < 0.7):
            text += "；该映射只表示方向相关，不宜写成已命中确定关键节点"
        return text + "。"

    def _build_frontier_subheading(self, packet: Dict[str, Any]) -> str:
        source_set = {item for item in packet.get("source_types", []) if item}
        if "paper" in source_set and "patent" in source_set and ("news" in source_set or "report" in source_set):
            suffix = "研究验证向工程布局延伸"
        elif "paper" in source_set and ("news" in source_set or "report" in source_set):
            suffix = "研究线索向产业场景扩散"
        elif "news" in source_set or "report" in source_set:
            suffix = "产业关注热度上升"
        elif "paper" in source_set:
            suffix = "研究验证持续推进"
        elif "patent" in source_set:
            suffix = "工程布局信号显现"
        else:
            suffix = "进入持续观察区间"
        return f"{packet['name']}：{suffix}"

    def _build_industry_subheading(self, scope_packets: List[Dict[str, Any]], weak_packets: List[Dict[str, Any]]) -> str:
        scope_names = [packet["name"] for packet in scope_packets[:2] if packet.get("name")]
        if len(scope_names) >= 2:
            return f"{scope_names[0]}与{scope_names[1]}方向热度上升"
        if len(scope_names) == 1:
            return f"{scope_names[0]}方向动向梳理"
        if weak_packets:
            top_names = [packet["name"] for packet in weak_packets[:2] if packet.get("name")]
            if len(top_names) >= 2:
                return f"{top_names[0]}等方向带动产业关注"
            if len(top_names) == 1:
                return f"{top_names[0]}带动相关产业动向"
        return "重点领域产业动向"

    def _build_near_signal_subheading(self, near_packets: List[Dict[str, Any]]) -> str:
        if not near_packets:
            return "产业扩散仍处早期阶段"
        top_names = [packet["name"] for packet in near_packets[:2] if packet.get("name")]
        if len(top_names) >= 2:
            return f"{top_names[0]}等方向进入持续跟踪区间"
        return f"{top_names[0]}进入持续跟踪区间"

    def _build_hotspot_subheading(self, weak_packets: List[Dict[str, Any]]) -> str:
        if not weak_packets:
            return "热点方向判断"
        return f"{weak_packets[0]['name']}具备热点演化潜力"

    def _compose_near_signal_summary(self, packet: Dict[str, Any]) -> str:
        mechanism_text = self._report_mechanism_text(packet)
        parts = [
            (
                f"{packet['name']} 已进入持续观察区间，当前主要体现为{mechanism_text}相关信号。"
                if mechanism_text
                else f"{packet['name']} 已进入持续观察区间。"
            ),
            self._compose_signal_position(packet),
            packet.get("timeline_summary", ""),
            self._compose_packet_evidence_narrative(packet, max_items=2),
            packet.get("organization_summary", ""),
        ]
        summary = " ".join(part for part in parts if part).strip()
        return summary or f"{packet['name']} 已进入持续观察区间，建议继续关注其后续变化。"

    def _compose_hotspot_focus(self, packets: List[Dict[str, Any]]) -> str:
        if not packets:
            return ""
        top_packet = packets[0]
        source_set = {item for item in top_packet.get("source_types", []) if item}
        if "paper" in source_set and "patent" in source_set and ("news" in source_set or "report" in source_set):
            return f"{top_packet['name']} 的证据链较完整，后续可重点观察其是否继续向更多主体和应用场景扩散。"
        if "paper" in source_set and ("news" in source_set or "report" in source_set):
            return f"{top_packet['name']} 已出现由研究验证向产业关注扩散的迹象，值得持续纳入热点跟踪。"
        if "news" in source_set or "report" in source_set:
            return f"{top_packet['name']} 当前以产业讨论与场景观察为主，后续需关注是否出现更强的研究或工程证据。"
        return f"{top_packet['name']} 值得纳入持续跟踪清单。"

    def _compose_cluster_overview(self, clusters: List[Dict[str, Any]], weak_count: int) -> str:
        if not clusters:
            return ""
        top_clusters = clusters[:4]
        top_texts = []
        for cluster in top_clusters:
            names = cluster.get("top_signal_names", [])[:3]
            name_text = "、".join(names) if names else cluster.get("cluster_name", "未命名弱信号簇")
            top_texts.append(f"{cluster.get('cluster_name')}覆盖{cluster.get('signal_count', 0)}个弱信号，代表线索包括{name_text}")
        return (
            f"本轮共识别出 {weak_count} 个弱信号，已按对象族、机制核和研究分桶归并为 "
            f"{len(clusters)} 个弱信号簇。"
            f"{'；'.join(top_texts)}。"
        )

    def _compose_cluster_summary_line(self, cluster: Dict[str, Any]) -> str:
        names = cluster.get("top_signal_names", [])[:4]
        source_text = self._source_type_text(cluster.get("source_types", []), fallback="多源文本")
        mechanism_text = "、".join(cluster.get("mechanisms", [])[:3])
        evidence_count = int(cluster.get("evidence_count_total", 0) or 0)
        name_text = "、".join(names) if names else cluster.get("cluster_name", "相关方向")
        mechanism_part = f"，主要机制指向{mechanism_text}" if mechanism_text else ""
        return (
            f"{cluster.get('cluster_name')}：覆盖 {cluster.get('signal_count', 0)} 个弱信号"
            f"{mechanism_part}，证据侧主要来自{source_text}，累计保留 {evidence_count} 条可追溯证据；"
            f"代表线索包括{name_text}。"
        )

    def _compose_cluster_coverage_lines(self, clusters: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for idx, cluster in enumerate(clusters, 1):
            covered_signals = [
                item for item in cluster.get("covered_signals", [])
                if isinstance(item, dict) and (item.get("signal_id") or item.get("name"))
            ]
            if covered_signals:
                signal_texts = [
                    f"{item.get('signal_id')}: {item.get('name')}"
                    if item.get("signal_id")
                    else str(item.get("name"))
                    for item in covered_signals
                ]
            else:
                signal_texts = [name for name in cluster.get("covered_signal_names", []) if name]
            if not signal_texts:
                continue
            lines.append(
                f"{idx}. {cluster.get('cluster_name')}（{cluster.get('cluster_basis', '聚类')}，"
                f"覆盖 {len(signal_texts)} 个）：{'、'.join(signal_texts)}。"
            )
        return lines

    def _compose_cluster_industry_text(self, clusters: List[Dict[str, Any]]) -> str:
        if not clusters:
            return ""
        parts = []
        for cluster in clusters[:3]:
            source_text = self._source_type_text(cluster.get("source_types", []), fallback="多源文本")
            names = "、".join(cluster.get("top_signal_names", [])[:2])
            suffix = f"，其中{names}等方向更适合作为产业侧跟踪入口" if names else ""
            parts.append(f"{cluster.get('cluster_name')}在{source_text}中形成连续线索{suffix}")
        return "；".join(parts) + "。"

    def _compose_cluster_followup_text(self, clusters: List[Dict[str, Any]]) -> str:
        if not clusters:
            return ""
        focus = []
        for cluster in clusters[:3]:
            names = cluster.get("top_signal_names", [])[:2]
            name_text = "、".join(names) if names else cluster.get("cluster_name", "相关方向")
            focus.append(f"围绕{cluster.get('cluster_name')}继续跟踪{name_text}的新增证据")
        return "；".join(focus) + "。"

    def _key_core_tier_label(self, tier: Any) -> str:
        return {
            "core_key_candidate": "核心候选",
            "strong_key_potential": "强潜力候选",
            "watchlist_key_potential": "观察候选",
            "weak_signal_only": "弱信号保留",
            "insufficient_evidence": "证据不足",
            "not_key_core_candidate": "暂不列入",
        }.get(self._safe_report_text(tier), self._safe_report_text(tier) or "未分层")

    def _compose_key_core_candidate_lines(self, candidates: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
        lines = []
        for item in candidates[:max_items]:
            name = self._safe_report_text(item.get("name")) or "Unknown"
            score = self._safe_report_float(item.get("key_core_score"), 0.0)
            tier = self._key_core_tier_label(item.get("key_core_tier"))
            chain_name = self._safe_report_text(item.get("tech_chain_name")) or "未命中明确节点"
            bottleneck = self._safe_report_text(item.get("bottleneck_level")) or "unknown"
            strategic = self._safe_report_text(item.get("strategic_importance_level")) or "unknown"
            temporal = self._safe_report_text(item.get("temporal_validation_tier")) or "unknown"
            action = self._safe_report_text(item.get("recommended_action"))
            risk = self._safe_report_text(item.get("risk"))
            suffix = f"建议：{action}" if action else "建议继续补充跨源、跨时间证据。"
            if risk:
                suffix += f" 风险：{risk}。"
            lines.append(
                f"{int(self._safe_report_float(item.get('rank'), len(lines) + 1))}. {name}：关键核心潜力得分 {score:.1f}，"
                f"层级为{tier}，技术链映射到“{chain_name}”，卡点/战略等级为 {bottleneck}/{strategic}，"
                f"时间验证为 {temporal}。{suffix}"
            )
        return lines

    @staticmethod
    def _cn_marker(index: int) -> str:
        numerals = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 0 <= index < len(numerals):
            return f"（{numerals[index]}）"
        return f"（{index + 1}）"

    def _compact_report_paragraph(self, texts: List[Any], max_items: int = 2) -> str:
        cleaned = []
        for text in texts:
            value = self._safe_report_text(text)
            if not value:
                continue
            value = re.sub(r"\s+", " ", value).strip().rstrip("。；; ")
            if not value or value in cleaned:
                continue
            cleaned.append(value)
            if len(cleaned) >= max_items:
                break
        if not cleaned:
            return ""
        paragraph = "；".join(cleaned)
        return paragraph + ("。" if paragraph[-1] not in "。！？!?" else "")

    def _compose_signal_summary(self, packet: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        source_label = self._source_type_text(packet.get("source_types", []))
        mechanism_text = self._report_mechanism_text(packet)
        lead = (
            (
                f"从{source_label}等来源看，{packet['name']} 在{mechanism_text}方面已形成较清晰的前沿观察画像。"
                if mechanism_text
                else f"从{source_label}等来源看，{packet['name']} 已形成较清晰的前沿观察画像。"
            )
        )

        detail_candidates = []
        for entry in analysis.get("key_facts", []):
            statement = self._safe_report_text(entry.get("statement"))
            if not statement:
                continue
            if (
                "当前线索主要指向" in statement
                or "当前可直接追溯的代表性证据包括" in statement
                or "时间分布" in statement
                or "可追溯时间线" in statement
                or "当前可追溯线索主要集中" in statement
                or "相关线索主要分布在" in statement
                or "相关线索已涉及" in statement
                or "侧线索提到" in statement
                or "证据质量均值" in statement
                or "技术链上可映射" in statement
                or "技术链映射尚未命中" in statement
            ):
                continue
            detail_candidates.append(statement)
        details = self._compact_report_paragraph(detail_candidates, max_items=1)
        insight = self._compose_signal_position(packet)
        evidence_narrative = self._compose_packet_evidence_narrative(packet, max_items=2)
        quality_text = self._compose_packet_quality_text(packet)
        tech_chain_text = self._compose_packet_tech_chain_text(packet)
        caution = self._compact_report_paragraph(analysis.get("uncertainties", []), max_items=1)
        return " ".join(
            part
            for part in [
                lead,
                insight,
                self._safe_report_text(packet.get("timeline_summary")),
                evidence_narrative,
                quality_text,
                tech_chain_text,
                self._safe_report_text(packet.get("organization_summary")),
                details,
                caution,
            ]
            if part
        ).strip()

    def _grounded_render_report(self, facts: Dict[str, Any], packets: Dict[str, Any]) -> str:
        weak_packets = packets.get("weak_signals", [])
        weak_clusters = packets.get("weak_signal_clusters", [])
        near_packets = packets.get("near_strong_signals", [])
        scope_packets = packets.get("scope_overviews", [])
        key_core_candidates = packets.get("key_core_candidates", [])
        summary = packets.get("summary", {})
        weak_count = int(summary.get("weak_signal_count", len(weak_packets)) or len(weak_packets))
        analysis_by_id = {item["signal_id"]: item for item in facts.get("signal_analyses", [])}
        intro = self._compose_report_intro(packets)

        report_lines = [
            "# 科技前沿与产业观察",
            "",
            f"生成时间：{packets.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
            "",
            intro,
            "",
            "一、前沿技术摘要",
            "",
        ]

        frontier_packets = weak_packets if 0 < weak_count <= 8 else weak_packets[:2]
        if weak_clusters and weak_count > 8:
            report_lines.append("（一）弱信号簇总体画像")
            report_lines.append(self._compose_cluster_overview(weak_clusters, weak_count))
            report_lines.append("")

            report_lines.append("（二）重点弱信号簇")
            for cluster in weak_clusters[:6]:
                report_lines.append(self._compose_cluster_summary_line(cluster))
            report_lines.append("")

            report_lines.append("（三）弱信号覆盖清单")
            report_lines.extend(self._compose_cluster_coverage_lines(weak_clusters))
            report_lines.append("")
        elif frontier_packets:
            for idx, packet in enumerate(frontier_packets):
                analysis = analysis_by_id.get(packet["signal_id"], {})
                report_lines.append(f"{self._cn_marker(idx)} {self._build_frontier_subheading(packet)}")
                report_lines.append(self._compose_signal_summary(packet, analysis))
                report_lines.append("")
            if weak_clusters:
                report_lines.append(f"{self._cn_marker(len(frontier_packets))} 弱信号覆盖清单")
                report_lines.extend(self._compose_cluster_coverage_lines(weak_clusters))
                report_lines.append("")
        else:
            report_lines.append("（一）暂无形成稳定前沿画像的技术方向")
            report_lines.append("当前输入样本尚不足以支撑明确的前沿技术摘要，后续需要继续补充跨源且可追溯的文本证据。")
            report_lines.append("")

        report_lines.extend(["二、关键核心技术候选", ""])
        if key_core_candidates:
            core_count = sum(
                1
                for item in key_core_candidates
                if self._safe_report_text(item.get("key_core_tier")) == "core_key_candidate"
            )
            strong_count = sum(
                1
                for item in key_core_candidates
                if self._safe_report_text(item.get("key_core_tier")) == "strong_key_potential"
            )
            watch_count = sum(
                1
                for item in key_core_candidates
                if self._safe_report_text(item.get("key_core_tier")) == "watchlist_key_potential"
            )
            report_lines.append("（一）候选总体判断")
            report_lines.append(
                f"v2.7 已在弱信号评分之外单独生成关键核心潜力评分。本轮形成 {len(key_core_candidates)} 个候选，"
                f"其中核心候选 {core_count} 个、强潜力候选 {strong_count} 个、观察候选 {watch_count} 个。"
                "该结论仍是潜力排序，不等同于最终关键核心技术认定。"
            )
            report_lines.append("")
            report_lines.append("（二）优先复核清单")
            report_lines.extend(self._compose_key_core_candidate_lines(key_core_candidates))
            report_lines.append("")
        else:
            report_lines.append("（一）暂无可提升的关键核心候选")
            report_lines.append(
                "当前样本尚未同时满足时间增长、技术链卡点、战略重要性和证据质量等门槛，建议继续保留为弱信号观察。"
            )
            report_lines.append("")

        report_lines.extend(["三、产业科技动态", ""])

        scope_names = [packet["name"] for packet in scope_packets[:2]]
        source_pool = []
        for packet in weak_packets + near_packets:
            source_pool.extend(packet.get("source_types", []))
        source_pool = [item for item in source_pool if item]
        source_text = self._source_type_text(sorted(dict.fromkeys(source_pool))[:4])
        overall_text = self._compact_report_paragraph(
            [entry.get("statement") for entry in facts.get("overall_findings", [])],
            max_items=2,
        )
        chain_text = self._compose_cross_signal_chain_summary(weak_packets + near_packets)
        orgs = self._collect_packet_orgs(weak_packets + near_packets, max_items=4)
        org_text = f"相关线索已涉及{'、'.join(orgs)}等主体。" if orgs else ""
        dynamic_intro = (
            f"从当前样本看，产业科技动态主要围绕{('、'.join(scope_names) if scope_names else '重点技术对象')}展开，"
            f"来源结构以 {source_text} 为主。"
        )
        cluster_industry_text = self._compose_cluster_industry_text(weak_clusters)
        dynamic_paragraph = " ".join(
            part for part in [dynamic_intro, chain_text, cluster_industry_text, overall_text, org_text] if part
        ).strip()
        report_lines.append(f"（一）{self._build_industry_subheading(scope_packets, weak_packets)}")
        report_lines.append(dynamic_paragraph or "当前产业科技动态仍以早期线索累积为主，多源扩散态势有待继续确认。")
        report_lines.append("")

        family_summary_text = self._compose_family_metrics_summary()
        report_lines.append("（二）对象族承接格局")
        report_lines.append(family_summary_text)
        report_lines.append("")

        report_lines.append(f"（三）{self._build_near_signal_subheading(near_packets)}")
        if near_packets:
            near_texts = []
            for packet in near_packets[:2]:
                near_texts.append(self._compose_near_signal_summary(packet))
            report_lines.append(self._compact_report_paragraph(near_texts, max_items=2))
        else:
            report_lines.append("当前尚未形成需要单列展开的近强信号，产业侧变化仍以前沿弱信号的早期扩散为主。")
        report_lines.append("")

        report_lines.extend(["四、热点观察", ""])

        trend_text = self._compact_report_paragraph(
            [entry.get("statement") for entry in facts.get("trend_implications", [])],
            max_items=2,
        )
        cluster_followup_text = self._compose_cluster_followup_text(weak_clusters)
        if cluster_followup_text:
            trend_text = self._compact_report_paragraph([trend_text, cluster_followup_text], max_items=2)
        if not trend_text and weak_packets:
            trend_text = self._compact_report_paragraph(
                [f"{packet['name']} 值得纳入持续跟踪清单" for packet in weak_packets[:2]],
                max_items=2,
            )
        hotspot_focus = self._compose_hotspot_focus(weak_packets)
        if not trend_text:
            trend_text = hotspot_focus
        report_lines.append(f"（一）{self._build_hotspot_subheading(weak_packets)}")
        report_lines.append(trend_text or "当前热点方向仍处于早期形成阶段，建议继续依据新增证据判断其是否演化为稳定趋势。")
        report_lines.append("")

        next_steps_text = self._compact_report_paragraph(facts.get("next_steps", []), max_items=2)
        limit_text = self._compact_report_paragraph(facts.get("report_limits", []), max_items=1)
        followup_text = " ".join(part for part in [cluster_followup_text, next_steps_text, limit_text] if part).strip()
        report_lines.append("（二）后续关注重点")
        report_lines.append(followup_text or "建议继续补充跨源、跨时间的可追溯文本证据，并结合人工复核持续更新热点判断。")

        return "\n".join(report_lines).strip() + "\n"

    def _generate_grounded_report(self, signals_output) -> str:
        self.report_artifacts = {}
        candidates_df, weak_signals, near_strong_df, scope_overviews = self._normalize_report_inputs(signals_output)
        self._evaluate_family_metrics(candidates_df)
        packets = self._build_grounded_report_packets(candidates_df, weak_signals, near_strong_df, scope_overviews)
        self.report_artifacts["report_evidence_packets"] = packets

        provider, client = get_provider_and_client()
        print(f"  [报告生成] LLM客户端状态: provider={provider}, client={'已初始化' if client else '未初始化'}")

        facts = None
        llm_used = False
        if client is not None and provider is not None and packets.get("weak_signals"):
            try:
                extracted = self._grounded_extract_facts(packets)
                facts = self._grounded_sanitize_facts(extracted, packets)
                check = self._grounded_build_check(facts, packets)
                self.report_artifacts["report_grounding_check"] = check
                llm_used = bool(check.get("passed", False))
                if not check.get("passed", False):
                    print("  [报告生成] grounding check 未通过，回退到规则渲染")
                    facts = None
                else:
                    print("  [报告生成] grounded facts 提取完成")
            except Exception as e:
                import traceback
                print(f"[WARNING] grounded report 生成失败: {e}")
                print(f"[DEBUG] 错误详情: {traceback.format_exc()}")
                print("  [报告生成] 使用规则回退生成")
                facts = None
        else:
            print("  [报告生成] 跳过LLM grounded 提取，使用规则回退生成")

        if facts is None:
            facts = self._grounded_build_default_facts(packets)
            self.report_artifacts["report_grounding_check"] = self._grounded_build_check(facts, packets)

        self.report_artifacts["report_grounded_facts"] = facts
        self.report_artifacts["report_generation_mode"] = "llm_grounded" if llm_used and facts is not None else "fallback_grounded"
        return self._grounded_render_report(facts, packets)

    def _generate_report(self, signals_output) -> str:
        """生成 grounded 报告。"""
        return self._generate_grounded_report(signals_output)
    
    def _generate_report_with_llm(self, candidates_df, weak_signals, near_strong_df) -> str:
        """使用LLM生成报告"""
        # 构建信号信息
        signals_info = []
        
        # 添加弱信号
        if not weak_signals.empty:
            for idx, row in weak_signals.head(10).iterrows():
                name = row.get('display_candidate_name', row.get('canonical_candidate_name_en', row.get('tech_name', 'Unknown')))
                score = row.get('weak_signal_score', row.get('hotspot_score', 0))
                mechanism = row.get('mechanism_core', 'N/A')
                evidence_count = row.get('cluster_evidence_count', 0)
                org_count = row.get('org_count', 0)
                signals_info.append(f"弱信号{len(signals_info)+1}: {name}")
                signals_info.append(f"  - 得分: {score:.2f}")
                signals_info.append(f"  - 机制: {mechanism}")
                signals_info.append(f"  - 证据数: {evidence_count}")
                signals_info.append(f"  - 机构数: {org_count}")
        
        # 添加近强信号
        if not near_strong_df.empty:
            for _, row in near_strong_df.head(10).iterrows():
                name = row.get('display_candidate_name', row.get('canonical_candidate_name_en', row.get('tech_name', 'Unknown')))
                score = row.get('weak_signal_score', row.get('hotspot_score', 0))
                mechanism = row.get('mechanism_core', 'N/A')
                evidence_count = row.get('cluster_evidence_count', 0)
                signals_info.append(f"近强信号{len(signals_info)+1}: {name}")
                signals_info.append(f"  - 得分: {score:.2f}")
                signals_info.append(f"  - 机制: {mechanism}")
                signals_info.append(f"  - 证据数: {evidence_count}")
        
        # 添加观察范围概览
        if not candidates_df.empty and 'candidate_stage' in candidates_df.columns:
            scope_overviews = candidates_df[candidates_df['candidate_stage'] == 'scope_overview']
            if not scope_overviews.empty:
                signals_info.append("")
                signals_info.append("观察范围概览:")
                for _, row in scope_overviews.head(5).iterrows():
                    name = row.get('display_candidate_name', row.get('scope_name', 'Unknown'))
                    count = row.get('cluster_evidence_count', 0)
                    signals_info.append(f"  - {name} (证据数: {count})")
        
        signals_text = "\n".join(signals_info) if signals_info else "暂无有效信号"
        
        prompt = f"""你是一位AI技术趋势分析师，请基于以下数据生成一份结构化的技术趋势分析报告。

**分析数据：**

{signals_text}

**统计信息：**
- 候选对象总数: {len(candidates_df)}
- 弱信号数量: {len(weak_signals)}
- 近强信号数量: {len(near_strong_df)}

**报告要求：**
请生成一份专业、连贯的报告，包含以下部分：
1. 执行摘要
2. 主要发现
3. 弱信号详细分析
4. 技术趋势洞察
5. 建议与下一步

请用中文撰写报告，语言专业、简洁、有洞察力。
"""
        
        # 使用配置的报告生成模型
        report_model = os.getenv("REPORT_MODEL", os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
        print(f"  [报告生成] 使用模型: {report_model}")
        print(f"  [报告生成] 正在发送请求...")
        
        result, usage_info, _ = chat_text(
            prompt,
            model=report_model,
            temperature=0.1,
            max_tokens=1000,
            timeout=60,
        )
        
        print(f"  [报告生成] LLM响应长度: {len(result)} 字符")
        print(f"  [报告生成] Token使用: prompt={usage_info['prompt_tokens']}, completion={usage_info['completion_tokens']}")
        
        if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
            record_call(
                call_type="报告生成",
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
                success=True,
            )
        
        # 构建完整报告
        report_lines = [
            "=" * 60,
            "产业技术弱信号分析报告",
            "=" * 60,
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"候选对象总数: {len(candidates_df)}",
            f"弱信号数量: {len(weak_signals)}",
            f"近强信号数量: {len(near_strong_df)}",
            "",
            "-" * 60,
            "LLM生成的分析报告",
            "-" * 60,
            "",
        ]
        
        report_lines.append(result)
        
        report_lines.extend([
            "",
            "=" * 60,
            "报告结束",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)
    
    def _generate_report_fallback(self, candidates_df, weak_signals, near_strong_df) -> str:
        """回退生成报告（规则方式）"""
        report_lines = [
            "# 产业技术弱信号分析报告",
            "",
            f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 候选对象总数: {len(candidates_df)}",
            f"- 弱信号数量: {len(weak_signals)}",
            f"- 近强信号数量: {len(near_strong_df)}",
            "",
            "## 1. 执行摘要",
            "",
        ]

        if len(weak_signals) > 0:
            top_names = weak_signals.head(3)["display_candidate_name"].fillna("").astype(str).tolist()
            summary = f"本轮共识别出 {len(weak_signals)} 个弱信号、{len(near_strong_df)} 个近强信号。当前较值得关注的方向包括：{', '.join([name for name in top_names if name][:3]) or '暂无明显方向'}。"
        else:
            summary = f"本轮未识别出明确弱信号，但筛出了 {len(near_strong_df)} 个近强信号，可继续跟踪其证据积累情况。"
        report_lines.extend([summary, "", "## 2. 弱信号列表", ""])

        if len(weak_signals) > 0:
            sorted_signals = weak_signals.sort_values(
                by=['weak_signal_score', 'hotspot_score'],
                ascending=[False, False]
            ) if 'weak_signal_score' in weak_signals.columns else weak_signals
            
            for idx, (_, row) in enumerate(sorted_signals.head(20).iterrows(), 1):
                name = row.get('display_candidate_name', row.get('canonical_candidate_name_en', row.get('tech_name', 'Unknown')))
                score = row.get('weak_signal_score', row.get('hotspot_score', 0))
                signal_type = row.get('signal_type', 'unknown')
                source_count = row.get('source_count', 0)
                evidence_count = row.get('cluster_evidence_count', 0)
                mechanism = row.get('mechanism_core', 'N/A')
                report_lines.append(f"{idx}. **{name}**")
                report_lines.append(f"   - 得分: {score:.2f}")
                report_lines.append(f"   - 类型: {signal_type}")
                report_lines.append(f"   - 来源数: {source_count}")
                report_lines.append(f"   - 证据数: {evidence_count}")
                report_lines.append(f"   - 机制: {mechanism}")
        else:
            report_lines.append("- 暂无弱信号")
        
        # 添加观察范围概览
        if not candidates_df.empty and 'candidate_stage' in candidates_df.columns:
            scope_overviews = candidates_df[candidates_df['candidate_stage'] == 'scope_overview']
        else:
            scope_overviews = pd.DataFrame()
        
        report_lines.extend(["", "## 3. 观察范围概览", ""])
        if len(scope_overviews) > 0:
            for idx, (_, row) in enumerate(scope_overviews.head(10).iterrows(), 1):
                name = row.get('display_candidate_name', row.get('scope_name', 'Unknown'))
                count = row.get('cluster_evidence_count', 0)
                report_lines.append(f"{idx}. **{name}** (证据数: {count})")
        else:
            report_lines.append("- 暂无观察范围概览")
        
        # 添加近强信号
        report_lines.extend(["", "## 4. 近强信号列表", ""])
        if len(near_strong_df) > 0:
            sort_cols = []
            if 'cluster_evidence_count' in near_strong_df.columns:
                sort_cols.append('cluster_evidence_count')
            if 'weak_signal_score' in near_strong_df.columns:
                sort_cols.append('weak_signal_score')
            
            if sort_cols:
                sorted_near_strong = near_strong_df.sort_values(
                    by=sort_cols,
                    ascending=[False] * len(sort_cols)
                )
            else:
                sorted_near_strong = near_strong_df
            
            for idx, (_, row) in enumerate(sorted_near_strong.head(10).iterrows(), 1):
                name = row.get('display_candidate_name', row.get('canonical_candidate_name_en', row.get('tech_name', 'Unknown')))
                count = row.get('cluster_evidence_count', 0)
                mechanism = row.get('mechanism_core', 'N/A')
                score = row.get('weak_signal_score', row.get('hotspot_score', row.get('score', 0)))
                report_lines.append(f"{idx}. **{name}**")
                report_lines.append(f"   - 得分: {score:.2f}")
                report_lines.append(f"   - 证据数: {count}")
                report_lines.append(f"   - 机制: {mechanism}")
        else:
            report_lines.append("- 暂无近强信号")

        return "\n".join(report_lines)

    def _save_results(
        self,
        result_dir: Path,
        events_df: pd.DataFrame,
        candidate_forms_df: pd.DataFrame,
        scored_df: pd.DataFrame,
        refined_df: pd.DataFrame,
        validated_df: pd.DataFrame,
        signals_df,
        report: str,
    ):
        """保存所有结果"""
        # 确保所有数据都是DataFrame
        def ensure_df(data):
            if isinstance(data, dict):
                if "candidates_df" in data and isinstance(data.get("candidates_df"), pd.DataFrame):
                    return data.get("candidates_df")
                return pd.DataFrame([data])
            elif not isinstance(data, pd.DataFrame):
                return pd.DataFrame(data if data else [])
            return data
        
        events_df = ensure_df(events_df)
        event_quality_df = ensure_df(self.latest_event_quality_df)
        tech_chain_mapping_df = ensure_df(self.latest_tech_chain_mapping_df)
        temporal_validation_df = ensure_df(self.latest_temporal_validation_df)
        key_core_scored_df = ensure_df(self.latest_key_core_scored_df)
        key_core_candidates_df = ensure_df(self.latest_key_core_candidates_df)
        candidate_forms_df = ensure_df(candidate_forms_df)
        scored_df = ensure_df(scored_df)
        refined_df = ensure_df(refined_df)
        validated_df = ensure_df(validated_df)
        signals_df = ensure_df(signals_df)
        reverse_validation_df = ensure_df(self.latest_reverse_validation_df)
        family_metrics_df = ensure_df(self.latest_family_metrics_df)
        final_shortlist_df = ensure_df(self.latest_final_shortlist_df)
        shortlist_dedup_df = ensure_df(self.latest_shortlist_dedup_df)
        frequency_baseline_df = ensure_df(self.latest_frequency_baseline_df)
        baseline_comparison_df = ensure_df(self.latest_baseline_comparison_df)
        
        # 保存为JSON
        events_df.to_json(result_dir / "events.json", orient='records', force_ascii=False, indent=2)
        event_quality_df.to_json(result_dir / "event_quality.json", orient='records', force_ascii=False, indent=2)
        event_quality_df.to_csv(result_dir / "event_quality.csv", index=False, encoding='utf-8-sig')
        tech_chain_mapping_df.to_json(result_dir / "tech_chain_mapping.json", orient='records', force_ascii=False, indent=2)
        tech_chain_mapping_df.to_csv(result_dir / "tech_chain_mapping.csv", index=False, encoding='utf-8-sig')
        temporal_validation_df.to_json(result_dir / "temporal_validation.json", orient='records', force_ascii=False, indent=2)
        temporal_validation_df.to_csv(result_dir / "temporal_validation.csv", index=False, encoding='utf-8-sig')
        key_core_scored_df.to_json(result_dir / "key_core_scored.json", orient='records', force_ascii=False, indent=2)
        key_core_scored_df.to_csv(result_dir / "key_core_scored.csv", index=False, encoding='utf-8-sig')
        key_core_candidates_df.to_json(result_dir / "key_core_candidates.json", orient='records', force_ascii=False, indent=2)
        key_core_candidates_df.to_csv(result_dir / "key_core_candidates.csv", index=False, encoding='utf-8-sig')
        candidate_forms_df.to_json(result_dir / "candidate_forms.json", orient='records', force_ascii=False, indent=2)
        scored_df.to_json(result_dir / "scored.json", orient='records', force_ascii=False, indent=2)
        refined_df.to_json(result_dir / "refined.json", orient='records', force_ascii=False, indent=2)
        validated_df.to_json(result_dir / "validated.json", orient='records', force_ascii=False, indent=2)
        signals_df.to_json(result_dir / "signals.json", orient='records', force_ascii=False, indent=2)
        if not reverse_validation_df.empty:
            reverse_validation_df.to_json(result_dir / "reverse_validation.json", orient='records', force_ascii=False, indent=2)
            reverse_validation_df.to_csv(result_dir / "reverse_validation.csv", index=False, encoding='utf-8-sig')
        if not family_metrics_df.empty:
            family_metrics_df.to_json(result_dir / "family_evaluation.json", orient='records', force_ascii=False, indent=2)
            family_metrics_df.to_csv(result_dir / "family_evaluation.csv", index=False, encoding='utf-8-sig')
        final_shortlist_df.to_json(result_dir / "final_shortlist.json", orient='records', force_ascii=False, indent=2)
        final_shortlist_df.to_csv(result_dir / "final_shortlist.csv", index=False, encoding='utf-8-sig')
        shortlist_dedup_df.to_json(result_dir / "final_shortlist_dedup_map.json", orient='records', force_ascii=False, indent=2)
        shortlist_dedup_df.to_csv(result_dir / "final_shortlist_dedup_map.csv", index=False, encoding='utf-8-sig')
        frequency_baseline_df.to_json(result_dir / "frequency_baseline.json", orient='records', force_ascii=False, indent=2)
        frequency_baseline_df.to_csv(result_dir / "frequency_baseline.csv", index=False, encoding='utf-8-sig')
        baseline_comparison_df.to_json(result_dir / "baseline_comparison.json", orient='records', force_ascii=False, indent=2)
        baseline_comparison_df.to_csv(result_dir / "baseline_comparison.csv", index=False, encoding='utf-8-sig')

        with open(result_dir / "report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存CSV格式（便于查看）
        signals_df.to_csv(result_dir / "signals.csv", index=False, encoding='utf-8-sig')

        validation_lines = [
            "# 分阶段改造验证记录",
            "",
            "## 运行结论",
            "",
            "- 旧主线 CLI 已完成一次端到端运行。",
            "- `report.txt` 继续按旧主线报告样式生成。",
            "- 本轮已保存增强后的证据链、对象族评估、最终短名单和基线对照产物。",
            "",
            "## 产物计数",
            "",
            f"- 事件数: {len(events_df)}",
            f"- 事件质量记录数: {len(event_quality_df)}",
            f"- 技术链映射记录数: {len(tech_chain_mapping_df)}",
            f"- 时间验证记录数: {len(temporal_validation_df)}",
            f"- 关键核心评分记录数: {len(key_core_scored_df)}",
            f"- 关键核心候选记录数: {len(key_core_candidates_df)}",
            f"- 候选数: {len(candidate_forms_df)}",
            f"- 评分候选数: {len(scored_df)}",
            f"- 反向验证记录数: {len(reverse_validation_df)}",
            f"- 对象族评估记录数: {len(family_metrics_df)}",
            f"- 最终短名单记录数: {len(final_shortlist_df)}",
            f"- 频次基线记录数: {len(frequency_baseline_df)}",
            f"- 基线对照记录数: {len(baseline_comparison_df)}",
            "",
            "## feat 参考对照",
            "",
            "- `focus_sample`、`balanced600`、`supplement_v2` 冻结结果继续作为参考样本，不直接作为当前主线结果。",
            "- 当前主线已覆盖 feat 中最关键的中间产物类型：反向验证、对象族评估、最终短名单、频次基线和基线对照。",
            "- 后续若使用相同大样本数据运行，可直接用本目录 CSV 与 feat 冻结 CSV 做对象级对照。",
            "",
            "## 剩余风险",
            "",
            "- 离线或网络受限环境下，LLM 调用会回退到规则报告；这不影响主线运行，但会影响报告语言丰富度。",
            "- 小样本验证只能证明链路完整，不能代表大样本下的信号排序质量。",
        ]
        (result_dir / "migration_validation.md").write_text(
            "\n".join(validation_lines) + "\n",
            encoding="utf-8",
        )

        if self.report_artifacts:
            packets = self.report_artifacts.get("report_evidence_packets")
            if packets is not None:
                (result_dir / "report_evidence_packets.json").write_text(
                    json.dumps(packets, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            grounded_facts = self.report_artifacts.get("report_grounded_facts")
            if grounded_facts is not None:
                (result_dir / "report_grounded_facts.json").write_text(
                    json.dumps(grounded_facts, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            grounding_check = self.report_artifacts.get("report_grounding_check")
            if grounding_check is not None:
                (result_dir / "report_grounding_check.json").write_text(
                    json.dumps(grounding_check, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            metadata = {
                "report_generation_mode": self.report_artifacts.get("report_generation_mode", ""),
                "final_shortlist_count": int(len(final_shortlist_df)),
                "frequency_baseline_count": int(len(frequency_baseline_df)),
                "baseline_comparison_count": int(len(baseline_comparison_df)),
                "tech_chain_mapping_count": int(len(tech_chain_mapping_df)),
                "temporal_validation_count": int(len(temporal_validation_df)),
                "key_core_scored_count": int(len(key_core_scored_df)),
                "key_core_candidates_count": int(len(key_core_candidates_df)),
            }
            (result_dir / "report_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            family_report = self.report_artifacts.get("family_evaluation_report")
            if family_report:
                (result_dir / "family_evaluation_report.md").write_text(
                    str(family_report),
                    encoding="utf-8",
                )
            raw_response = self.report_artifacts.get("report_llm_raw_response")
            if raw_response:
                (result_dir / "report_llm_raw_response.txt").write_text(
                    str(raw_response),
                    encoding="utf-8",
                )
