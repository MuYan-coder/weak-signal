import os
import logging
import re
from typing import Dict, List, Any, Optional
import pandas as pd

from .models import SourceQuery, DocumentRecord, DataSourceStats
from .mysql_client import MySQLClient
from .es_client import ESClient
from .query_builder import QueryBuilder
from .field_mapping import (
    normalize_record,
    normalize_mysql_literature,
    normalize_es_consulting,
    normalize_es_policy,
    normalize_es_report,
    normalize_es_patent
)
from .normalizer import DocumentNormalizer

logger = logging.getLogger("weak_signal.data_access.repository")

class DataRepository:
    """仓储类，数据访问层的统一入口，管理数据库调用及本地 Excel 降级"""

    SOURCE_COUNT_ALIASES = {
        "paper": ("paper", "literature", "文献", "论文"),
        "literature": ("literature", "paper", "文献", "论文"),
        "news": ("news", "consulting", "资讯", "咨询"),
        "consulting": ("consulting", "news", "资讯", "咨询"),
        "policy": ("policy", "政策"),
        "report": ("report", "研报"),
        "patent": ("patent", "专利"),
    }
    ES_DEFAULT_COUNT_SCAN_LIMIT = 5000
    ES_PAGE_SIZE_CAP = 1000

    def __init__(self, mysql_client: MySQLClient, es_client: ESClient, backend: str = "mock"):
        self.mysql_client = mysql_client
        self.es_client = es_client
        self.backend = backend

        # 探测连接是否可用，并设置实际工作的 backend
        self.mysql_active = False
        self.es_active = False

        if backend == "db":
            logger.info("测试 MySQL 与 Elasticsearch 实时数据库连接...")
            self.mysql_active = self.mysql_client.test_connection()
            self.es_active = self.es_client.test_connection()

            if not self.mysql_active:
                logger.warning("MySQL 连接不可用，将针对文献来源启用离线降级轨道")
            if not self.es_active:
                logger.warning("Elasticsearch 连接不可用，将针对资讯/政策/研报/专利启用离线降级轨道")
        else:
            logger.info("WEAK_SIGNAL_DATA_BACKEND 被设置为 mock，将直接使用本地 Excel 数据")

    @classmethod
    def from_env(cls) -> "DataRepository":
        """工厂方法，从环境变量自动创建配置并初始化"""
        backend = os.getenv("WEAK_SIGNAL_DATA_BACKEND", "mock").lower().strip()
        mysql_client = MySQLClient()
        es_client = ESClient()
        return cls(mysql_client=mysql_client, es_client=es_client, backend=backend)

    @classmethod
    def _resolve_source_limit(cls, counts: Dict[str, int], source_type: str, default: int = 100) -> int:
        """Resolve per-source limits while accepting both public and raw source keys."""
        if not counts:
            return default

        aliases = cls.SOURCE_COUNT_ALIASES.get(source_type, (source_type,))
        for key in aliases:
            if key in counts:
                try:
                    return int(counts.get(key) or 0)
                except (TypeError, ValueError):
                    return 0
        return default

    @staticmethod
    def _raw_type_for_source(source_type: str) -> str:
        if source_type == "policy":
            return "policy"
        if source_type == "report":
            return "report"
        if source_type == "patent":
            return "patent"
        return "consulting"

    @staticmethod
    def _date_field_for_source(source_type: str) -> str:
        if source_type == "patent":
            return "public_date"
        return "publish_date"

    @staticmethod
    def _safe_positive_int_env(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, default))
        except (TypeError, ValueError):
            return default
        return max(value, 1)

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        text = str(value or "").lower()
        text = re.sub(r"[-_/]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _cjk_length(value: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", value or ""))

    @classmethod
    def _is_strong_domain_term(cls, term: str) -> bool:
        normalized = cls._normalize_match_text(term)
        if not normalized:
            return False
        if " " in normalized and len(normalized) >= 4:
            return True
        cjk_len = cls._cjk_length(normalized)
        if cjk_len >= 3:
            return True
        if cjk_len >= 2 and re.search(r"[a-z0-9]", normalized):
            return True
        return False

    @classmethod
    def _record_matches_domain_terms(cls, record: DocumentRecord, search_terms: List[str]) -> bool:
        """Apply one shared domain relevance filter for count and load paths."""
        normalized_terms = []
        seen = set()
        for term in search_terms:
            normalized = cls._normalize_match_text(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_terms.append(normalized)
        if not normalized_terms:
            return True

        haystack = cls._normalize_match_text(
            " ".join(
                str(value or "")
                for value in [
                    record.title,
                    record.text,
                    record.keywords,
                    record.classification,
                    record.industry,
                    record.source_name,
                    record.org,
                ]
            )
        )
        if not haystack:
            return False

        if len(normalized_terms) == 1:
            return normalized_terms[0] in haystack

        soft_hits = set()
        for term in normalized_terms:
            if term not in haystack:
                continue
            if re.fullmatch(r"[a-z0-9]{2,3}", term) and term not in QueryBuilder.ENGLISH_FRAGMENT_STOPWORDS:
                return True
            if cls._is_strong_domain_term(term):
                return True
            soft_hits.add(term)
        return len(soft_hits) >= 2

    @classmethod
    def _filter_records_by_domain_terms(
        cls,
        records: List[DocumentRecord],
        search_terms: List[str],
    ) -> List[DocumentRecord]:
        if not search_terms:
            return records
        return [record for record in records if cls._record_matches_domain_terms(record, search_terms)]

    @staticmethod
    def _normalize_es_hit(hit: Dict[str, Any], source_type: str) -> DocumentRecord:
        source_data = dict(hit.get("_source", {}))
        source_data["relevance_score"] = hit.get("_score", 0.0)
        source_data["id"] = hit.get("_id", source_data.get("id", ""))

        if source_type == "news":
            return normalize_es_consulting(source_data)
        if source_type == "policy":
            return normalize_es_policy(source_data)
        if source_type == "report":
            return normalize_es_report(source_data)
        if source_type == "patent":
            return normalize_es_patent(source_data)
        return normalize_es_consulting(source_data)

    def _load_es_documents_paginated(
        self,
        query: SourceQuery,
        *,
        source_type: str,
        index_name: str,
        date_field: str,
        limit: int,
        search_terms: List[str],
        existing_records: List[DocumentRecord],
    ) -> List[DocumentRecord]:
        page_size = min(max(limit * 2, 1), self.ES_PAGE_SIZE_CAP)
        raw_fetch_cap = max(limit * 8, page_size)
        search_after = None
        raw_seen = 0
        source_records: List[DocumentRecord] = []
        other_records = [record for record in existing_records if record.source_type != source_type]

        while raw_seen < raw_fetch_cap and len(source_records) < limit:
            current_size = min(page_size, raw_fetch_cap - raw_seen)
            dsl = QueryBuilder.build_es_dsl(
                query,
                date_field=date_field,
                size=current_size,
                search_after=search_after,
            )
            hits = self.es_client.search_documents(index_name, dsl)
            if not hits:
                break

            raw_seen += len(hits)
            current_batch = [self._normalize_es_hit(hit, source_type) for hit in hits]
            current_batch = self._filter_records_by_domain_terms(current_batch, search_terms)

            deduped_temp = DocumentNormalizer.deduplicate(other_records + source_records + current_batch)
            source_records = [record for record in deduped_temp if record.source_type == source_type][:limit]

            if len(source_records) >= limit or len(hits) < current_size:
                break

            next_search_after = hits[-1].get("sort")
            if not next_search_after:
                break
            search_after = next_search_after

        logger.info(
            f"数据源 {source_type} (需求: {limit}): ES 翻页扫描 {raw_seen} 条，"
            f"过滤去重后保留 {len(source_records)} 条。"
        )
        return source_records

    def _count_es_documents_after_filter(
        self,
        query: SourceQuery,
        *,
        source_type: str,
        index_name: str,
        date_field: str,
        search_terms: List[str],
    ) -> int:
        scan_limit = self._safe_positive_int_env(
            "WEAK_SIGNAL_COUNT_SCAN_LIMIT",
            self.ES_DEFAULT_COUNT_SCAN_LIMIT,
        )
        page_size = min(scan_limit, self.ES_PAGE_SIZE_CAP)
        search_after = None
        raw_seen = 0
        matched_count = 0

        while raw_seen < scan_limit:
            current_size = min(page_size, scan_limit - raw_seen)
            dsl = QueryBuilder.build_es_dsl(
                query,
                date_field=date_field,
                size=current_size,
                search_after=search_after,
            )
            hits = self.es_client.search_documents(index_name, dsl)
            if not hits:
                break

            raw_seen += len(hits)
            records = [self._normalize_es_hit(hit, source_type) for hit in hits]
            matched_count += len(self._filter_records_by_domain_terms(records, search_terms))

            if len(hits) < current_size:
                break

            next_search_after = hits[-1].get("sort")
            if not next_search_after:
                break
            search_after = next_search_after

        return matched_count

    def get_source_counts(self, query: SourceQuery) -> Dict[str, int]:
        """获取各来源在当前查询条件下的总可用数量"""
        counts = {}
        source_types = query.source_types if query.source_types else ["paper", "news", "policy", "report", "patent"]
        search_terms = QueryBuilder.expand_search_terms(query)

        for s_type in source_types:
            counts[s_type] = 0
            try:
                if s_type == "paper":
                    # MySQL
                    if self.backend == "db" and self.mysql_active:
                        sql, params = QueryBuilder.build_mysql_count_query(query, table_name=self.mysql_client.table_name)
                        counts[s_type] = self.mysql_client.query_count(sql, params)
                    else:
                        # 降级
                        mock_data = self.mysql_client.mock_query_documents(
                            keywords=search_terms,
                            synonyms=[],
                            exclude_terms=query.exclude_terms,
                            start_date=query.start_date,
                            end_date=query.end_date,
                            limit=99999
                        )
                        counts[s_type] = len(mock_data)
                else:
                    # ES
                    raw_type = self._raw_type_for_source(s_type)
                    index_name = self.es_client.indices.get(raw_type, s_type)
                    date_field = self._date_field_for_source(s_type)

                    if self.backend == "db" and self.es_active:
                        counts[s_type] = self._count_es_documents_after_filter(
                            query,
                            source_type=s_type,
                            index_name=index_name,
                            date_field=date_field,
                            search_terms=search_terms,
                        )
                    else:
                        # 降级
                        mock_data = self.es_client.mock_query_documents(
                            raw_source_type=raw_type,
                            keywords=search_terms,
                            synonyms=[],
                            exclude_terms=query.exclude_terms,
                            start_date=query.start_date,
                            end_date=query.end_date,
                            limit=99999
                        )
                        counts[s_type] = len(mock_data)
            except Exception as e:
                logger.error(f"获取 {s_type} 来源可用数据量失败: {e}")
                counts[s_type] = 0

        return counts

    def load_documents(self, query: SourceQuery) -> pd.DataFrame:
        """检索各渠道的文档数据，通过归一化、同源去重、近标题/URL去重及多源采样混序后，返回标准的 DataFrame"""
        all_records: List[DocumentRecord] = []
        source_types = query.source_types if query.source_types else ["paper", "news", "policy", "report", "patent"]
        search_terms = QueryBuilder.expand_search_terms(query)

        for s_type in source_types:
            limit = self._resolve_source_limit(query.counts, s_type)
            if limit <= 0:
                continue

            try:
                # 动态倍数加载补齐机制，最大到 limit * 8
                multiplier = 2
                max_multiplier = 8
                source_records = []

                while True:
                    current_limit = limit * multiplier
                    current_batch = []

                    if s_type == "paper":
                        # MySQL
                        if self.backend == "db" and self.mysql_active:
                            sql, params = QueryBuilder.build_mysql_query(query, table_name=self.mysql_client.table_name, limit=current_limit)
                            rows = self.mysql_client.query_documents(sql, params)
                            for row in rows:
                                current_batch.append(normalize_mysql_literature(row))
                        else:
                            # 降级 mock
                            rows = self.mysql_client.mock_query_documents(
                                keywords=search_terms,
                                synonyms=[],
                                exclude_terms=query.exclude_terms,
                                start_date=query.start_date,
                                end_date=query.end_date,
                                limit=current_limit
                            )
                            for row in rows:
                                current_batch.append(normalize_mysql_literature(row))
                    else:
                        # ES
                        raw_type = self._raw_type_for_source(s_type)
                        index_name = self.es_client.indices.get(raw_type, s_type)
                        date_field = self._date_field_for_source(s_type)

                        if self.backend == "db" and self.es_active:
                            source_records = self._load_es_documents_paginated(
                                query,
                                source_type=s_type,
                                index_name=index_name,
                                date_field=date_field,
                                limit=limit,
                                search_terms=search_terms,
                                existing_records=all_records,
                            )
                            all_records = [r for r in all_records if r.source_type != s_type] + source_records
                            break
                        else:
                            # 降级 mock
                            hits = self.es_client.mock_query_documents(
                                raw_source_type=raw_type,
                                keywords=search_terms,
                                synonyms=[],
                                exclude_terms=query.exclude_terms,
                                start_date=query.start_date,
                                end_date=query.end_date,
                                limit=current_limit
                            )
                            for hit in hits:
                                current_batch.append(self._normalize_es_hit(hit, s_type))

                    # A2: domain relevance pre-filtering on retrieved documents
                    raw_batch_size = len(current_batch)
                    if search_terms:
                        current_batch = self._filter_records_by_domain_terms(current_batch, search_terms)

                    # 针对当前单源与已加载的所有数据整体进行清洗去重
                    combined_temp = all_records + current_batch
                    deduped_temp = DocumentNormalizer.deduplicate(combined_temp)

                    # 统计在整体去重后，当前 source_type 拥有的唯一记录数
                    s_type_unique = [r for r in deduped_temp if r.source_type == s_type]

                    # 终止条件：
                    # 1. 整体去重后，属于该源的唯一数据量达到了限制数量 limit
                    # 2. 底层检索返回的数据量少于索求的当前限制（说明底层数据库已查空）
                    # 3. 达到最大倍数限制
                    if len(s_type_unique) >= limit or raw_batch_size < current_limit or multiplier >= max_multiplier:
                        source_records = s_type_unique[:limit]
                        # 更新已加载数据，保留新加载的唯一数据以及其他类型的数据
                        all_records = [r for r in deduped_temp if r.source_type != s_type] + source_records
                        logger.info(f"数据源 {s_type} (需求: {limit}): 最终通过倍数 {multiplier} 加载，获取 {raw_batch_size} 条，后过滤保留 {len(current_batch)} 条，去重合并后该源有 {len(s_type_unique)} 条，保留并合并 {len(source_records)} 条。")
                        break
                    else:
                        multiplier = min(multiplier * 2, max_multiplier)

            except Exception as e:
                logger.error(f"检索 {s_type} 渠道数据失败，将跳过: {e}")

        # 数据清洗过滤三层控制
        # 1. 归一化去重
        deduped = DocumentNormalizer.deduplicate(all_records)

        # 2. 分来源截断及合并排序
        final_list = DocumentNormalizer.sample_and_sort(deduped, query.counts, query.sort)

        # 3. 将对象转换为 DataFrame
        if not final_list:
            return pd.DataFrame(columns=[
                "id", "source_id", "source_type", "raw_source_type", "title", "text", "date",
                "publish_time", "org", "authors", "keywords", "url", "source_name",
                "classification", "industry", "relevance_score", "data_backend"
            ])

        data_dicts = []
        for r in final_list:
            data_dicts.append({
                "id": r.id,
                "source_id": r.source_id,
                "source_type": r.source_type,
                "raw_source_type": r.raw_source_type,
                "title": r.title,
                "text": r.text,
                "date": r.date,
                "publish_time": r.publish_time,
                "org": r.org,
                "authors": r.authors,
                "keywords": r.keywords,
                "url": r.url,
                "source_name": r.source_name,
                "classification": r.classification,
                "industry": r.industry,
                "relevance_score": r.relevance_score,
                "data_backend": r.data_backend
            })

        return pd.DataFrame(data_dicts)
