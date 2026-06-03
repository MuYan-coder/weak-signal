import re
import logging
from typing import Any, Dict, List, Set, Tuple
from .models import DocumentRecord

logger = logging.getLogger("weak_signal.data_access.normalizer")

class DocumentNormalizer:
    """文档标准化处理器，负责去重、排序与采样"""

    @staticmethod
    def clean_title(title: str) -> str:
        """标题规范化：转小写，去除所有空格及标点符号"""
        if not title:
            return ""
        # 移除非字母数字及中文字符
        text = title.lower().strip()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
        return text

    @classmethod
    def deduplicate(cls, records: List[DocumentRecord]) -> List[DocumentRecord]:
        """
        进行两层去重：
        1. 同源去重：同 source_type + source_id
        2. 跨源近重复去重：标题清洗后完全相同，或 url 相同。
           保留字段更完整、日期更新、相关性更高的一条。
        """
        # 1. 同源去重
        seen_sources: Set[Tuple[str, str]] = set()
        first_stage: List[DocumentRecord] = []

        for r in records:
            key = (r.source_type, r.source_id)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            first_stage.append(r)

        # 2. 跨源近重复去重
        # 判定新记录是否应替换旧记录
        def should_replace(old_rec: DocumentRecord, new_rec: DocumentRecord) -> bool:
            # 2.1 字段完整度 (有内容的非空字段数)
            def get_completeness(rec: DocumentRecord) -> int:
                score = 0
                for f in ["title", "text", "org", "authors", "keywords", "url", "source_name", "classification", "industry"]:
                    val = getattr(rec, f)
                    if val and str(val).strip().lower() not in ("", "nan", "null", "none"):
                        score += 1
                # text 的长度也计入完整度比对
                score += len(rec.text) // 100
                return score

            comp_old = get_completeness(old_rec)
            comp_new = get_completeness(new_rec)
            if comp_new != comp_old:
                return comp_new > comp_old

            # 2.2 日期新旧比较
            date_old = old_rec.date or ""
            date_new = new_rec.date or ""
            if date_new != date_old:
                return date_new > date_old

            # 2.3 相关性得分
            score_old = old_rec.relevance_score or 0.0
            score_new = new_rec.relevance_score or 0.0
            return score_new > score_old

        cleaned_title_map: Dict[str, DocumentRecord] = {}
        url_map: Dict[str, DocumentRecord] = {}
        final_records: List[DocumentRecord] = []

        for r in first_stage:
            title_key = cls.clean_title(r.title)
            url_key = r.url.strip() if r.url else ""

            existing = None
            if title_key and title_key in cleaned_title_map:
                existing = cleaned_title_map[title_key]
            elif url_key and url_key in url_map:
                existing = url_map[url_key]

            if existing is not None:
                # 冲突，判断是否替换
                if should_replace(existing, r):
                    # 从结果列表中移除旧的
                    if existing in final_records:
                        final_records.remove(existing)
                    # 放入新的
                    final_records.append(r)
                    if title_key:
                        cleaned_title_map[title_key] = r
                    if url_key:
                        url_map[url_key] = r
                else:
                    # 放弃新的，保留旧的
                    continue
            else:
                final_records.append(r)
                if title_key:
                    cleaned_title_map[title_key] = r
                if url_key:
                    url_map[url_key] = r

        logger.info(f"去重完成：原始数据 {len(records)} 条，去重后 {len(final_records)} 条")
        return final_records

    @staticmethod
    def sample_and_sort(
        records: List[DocumentRecord],
        counts: Dict[str, int],
        sort_mode: str = "relevance"
    ) -> List[DocumentRecord]:
        """
        分来源采样并排序：
        1. 按照 counts 限额对每类来源进行截断
        2. 应用不同的混序排序逻辑：
           - relevance: relevance_score desc, date desc 整体排序
           - date: date desc, relevance_score desc 整体排序
           - balanced: 各来源分类内部排序后轮转合并
        """
        # 按 source_type 分组
        grouped: Dict[str, List[DocumentRecord]] = {}
        for r in records:
            grouped.setdefault(r.source_type, []).append(r)

        # source_type 中英文映射，确保 counts 的 key 能正确匹配
        _source_type_aliases = {
            '文献': 'paper', '论文': 'paper', 'paper': 'paper', 'literature': 'paper',
            '资讯': 'news', 'news': 'news', 'consulting': 'news',
            '政策': 'policy', 'policy': 'policy',
            '研报': 'report', 'report': 'report',
            '专利': 'patent', 'patent': 'patent',
        }

        def _resolve_count(s_type: str, counts: Dict[str, int], default: int = 100) -> int:
            """尝试用原始 key、中英文别名查找 counts 中的限额"""
            if s_type in counts:
                return counts[s_type]
            alias = _source_type_aliases.get(s_type)
            if alias and alias in counts:
                return counts[alias]
            # 反向查找：如果 counts 用中文 key，而 s_type 是英文
            for cn, en in _source_type_aliases.items():
                if en == s_type and cn in counts:
                    return counts[cn]
            return default

        # 对各组进行内部排序，并截断采样
        sampled_groups: Dict[str, List[DocumentRecord]] = {}
        for s_type, group in grouped.items():
            # 排序策略 (内部都先按匹配度/时间排好)
            group.sort(
                key=lambda x: (
                    x.relevance_score or 0.0,
                    -(1 if not x.date else 0),
                    x.date or ""
                ),
                reverse=True
            )

            # 截断采样
            limit = _resolve_count(s_type, counts)
            sampled_groups[s_type] = group[:limit]

        # 混合合并
        merged: List[DocumentRecord] = []
        sort_mode = sort_mode.lower().strip()

        if sort_mode == "balanced":
            # 平衡轮转模式：每类数据中轮流取一条
            lists_to_merge = [list(lst) for lst in sampled_groups.values()]
            while any(lists_to_merge):
                for lst in lists_to_merge:
                    if lst:
                        merged.append(lst.pop(0))
        elif sort_mode == "date":
            # 时间优先：整体排序
            all_sampled = []
            for g in sampled_groups.values():
                all_sampled.extend(g)
            # 日期倒序，相同日期按相关性得分倒序
            all_sampled.sort(
                key=lambda x: (
                    -(1 if not x.date else 0),
                    x.date or "",
                    x.relevance_score or 0.0
                ),
                reverse=True
            )
            merged = all_sampled
        else:
            # 默认相关性优先
            all_sampled = []
            for g in sampled_groups.values():
                all_sampled.extend(g)
            all_sampled.sort(
                key=lambda x: (
                    x.relevance_score or 0.0,
                    -(1 if not x.date else 0),
                    x.date or ""
                ),
                reverse=True
            )
            merged = all_sampled

        return merged
