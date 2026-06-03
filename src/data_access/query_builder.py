from typing import Any, Dict, List, Tuple, Optional
from .models import SourceQuery
from ..extraction.tech_lexicon import TECH_ALIASES

class QueryBuilder:
    """查询构造器，将 SourceQuery 翻译为 SQL 和 ES DSL"""

    @staticmethod
    def expand_search_terms(query: SourceQuery) -> List[str]:
        """Combine user terms and optional local topic aliases into a deduped term list."""
        terms = []
        for term in list(query.keywords) + list(query.synonyms):
            text = str(term or "").strip()
            if text:
                terms.append(text)

        if getattr(query, "use_topic_index", True):
            lower_terms = {term.lower() for term in terms}
            for canonical, aliases in TECH_ALIASES.items():
                alias_terms = [canonical] + list(aliases)
                alias_lowers = {str(alias or "").strip().lower() for alias in alias_terms if str(alias or "").strip()}
                if lower_terms & alias_lowers:
                    terms.extend(alias_terms)

        deduped = []
        seen = set()
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(term)
        return deduped

    @staticmethod
    def build_mysql_query(
        query: SourceQuery,
        table_name: str = "dw_science_literature",
        limit: int = 100
    ) -> Tuple[str, Dict[str, Any]]:
        """
        构造 MySQL 参数化查询 SQL 和参数字典
        支持关键词匹配、同义词匹配、排除词以及时间范围过滤
        """
        params = {}
        where_clauses = ["1=1"]

        # 时间过滤
        if query.start_date:
            where_clauses.append("event_time >= %(start_date)s")
            params["start_date"] = query.start_date
        if query.end_date:
            # 包含当天：小于结束日期加1天
            where_clauses.append("event_time < DATE_ADD(%(end_date)s, INTERVAL 1 DAY)")
            params["end_date"] = query.end_date

        # 关键词与同义词
        search_terms = QueryBuilder.expand_search_terms(query)
        if search_terms:
            or_clauses = []
            for i, term in enumerate(search_terms):
                key = f"term_{i}"
                params[key] = f"%{term}%"
                or_clauses.append(
                    f"(title LIKE %({key})s OR abstract LIKE %({key})s OR keyword LIKE %({key})s OR features LIKE %({key})s)"
                )
            where_clauses.append(f"({' OR '.join(or_clauses)})")

        # 排除词
        if query.exclude_terms:
            for i, term in enumerate(query.exclude_terms):
                key = f"ex_term_{i}"
                params[key] = f"%{term}%"
                where_clauses.append(
                    f"NOT (title LIKE %({key})s OR abstract LIKE %({key})s OR features LIKE %({key})s)"
                )

        where_str = " AND ".join(where_clauses)

        # 排序
        if query.sort == "date":
            order_by = "event_time DESC"
        else:
            order_by = "event_time DESC"

        select_sql = f"""
            SELECT
                id,
                title,
                abstract,
                features,
                event_time,
                collect_source_name,
                authors,
                keyword,
                url,
                industry
            FROM {table_name}
            WHERE {where_str}
            ORDER BY {order_by}
            LIMIT %(limit)s
        """
        params["limit"] = limit

        return select_sql.strip(), params

    @staticmethod
    def build_mysql_count_query(
        query: SourceQuery,
        table_name: str = "dw_science_literature"
    ) -> Tuple[str, Dict[str, Any]]:
        """构造 MySQL COUNT 查询 SQL"""
        params = {}
        where_clauses = ["1=1"]

        if query.start_date:
            where_clauses.append("event_time >= %(start_date)s")
            params["start_date"] = query.start_date
        if query.end_date:
            where_clauses.append("event_time < DATE_ADD(%(end_date)s, INTERVAL 1 DAY)")
            params["end_date"] = query.end_date

        search_terms = QueryBuilder.expand_search_terms(query)
        if search_terms:
            or_clauses = []
            for i, term in enumerate(search_terms):
                key = f"term_{i}"
                params[key] = f"%{term}%"
                or_clauses.append(
                    f"(title LIKE %({key})s OR abstract LIKE %({key})s OR keyword LIKE %({key})s OR features LIKE %({key})s)"
                )
            where_clauses.append(f"({' OR '.join(or_clauses)})")

        if query.exclude_terms:
            for i, term in enumerate(query.exclude_terms):
                key = f"ex_term_{i}"
                params[key] = f"%{term}%"
                where_clauses.append(
                    f"NOT (title LIKE %({key})s OR abstract LIKE %({key})s OR features LIKE %({key})s)"
                )

        where_str = " AND ".join(where_clauses)
        count_sql = f"SELECT COUNT(*) AS total FROM {table_name} WHERE {where_str}"
        return count_sql.strip(), params

    @staticmethod
    def build_es_dsl(
        query: SourceQuery,
        date_field: str = "publish_date",
        size: int = 100,
        search_after: Optional[List[Any]] = None,
        tie_breaker_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        构造 ES DSL 查询体
        """
        dsl = {
            "size": size,
            "_source": [
                "id", "doc_id", "title", "abstract", "summary", "content", "main_content", "html",
                "viewpoints", "claims", "first_claim", "keywords", "tags", "entities", "org",
                "organization", "publisher", "applicant", "author", "authors", "inventors",
                "publish_time", "publish_date", "public_date", "apply_date", "priority_date",
                "created_at", "url", "link", "url_source", "classification", "ipc", "cpc",
                "industry", "stock_name", "stock_code", "direction", "lz_industry", "node_classify",
                "domain", "channel"
            ]
        }

        bool_query: Dict[str, Any] = {"filter": [], "must": [], "must_not": []}

        # 1. 时间过滤
        if query.start_date or query.end_date:
            range_filter = {}
            if query.start_date:
                range_filter["gte"] = query.start_date
            if query.end_date:
                range_filter["lte"] = query.end_date
            bool_query["filter"].append({"range": {date_field: range_filter}})

        # 2. 关键词与同义词全文本检索
        search_terms = QueryBuilder.expand_search_terms(query)
        if search_terms:
            search_str = " ".join(search_terms)
            bool_query["must"].append({
                "multi_match": {
                    "query": search_str,
                    "fields": [
                        "title^4", "title_cn^4",
                        "keywords^3", "tags^3",
                        "abstract^2", "abstract_cn^2", "summary^2",
                        "viewpoints^2", "claims^2", "first_claim^2",
                        "content", "main_content", "html"
                    ],
                    "type": "best_fields",
                    "operator": "or"
                }
            })
        else:
            bool_query["must"].append({"match_all": {}})

        # 3. 排除词
        if query.exclude_terms:
            exclude_str = " ".join(query.exclude_terms)
            bool_query["must_not"].append({
                "multi_match": {
                    "query": exclude_str,
                    "fields": ["title", "title_cn", "abstract", "abstract_cn", "content", "main_content", "html"]
                }
            })

        # 移除空列表
        dsl["query"] = {"bool": {k: v for k, v in bool_query.items() if v}}

        # 4. 排序
        # 默认使用 ES 的相关性评分排序，评分一致时按日期降序
        if query.sort == "date":
            dsl["sort"] = [
                {date_field: {"order": "desc"}},
                {"_score": {"order": "desc"}}
            ]
        else:
            # 默认 relevance: 相关性第一，日期第二
            dsl["sort"] = [
                {"_score": {"order": "desc"}},
                {date_field: {"order": "desc"}}
            ]

        # 5. 分页支持
        if search_after:
            if tie_breaker_field:
                dsl["sort"].append({tie_breaker_field: {"order": "asc"}})
            dsl["search_after"] = search_after

        return dsl
