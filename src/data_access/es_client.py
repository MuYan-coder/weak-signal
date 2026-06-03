import os
import logging
import hashlib
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

from ..utils.config import Config

# 配置日志
logger = logging.getLogger("weak_signal.data_access.es")

class ESClient:
    """Elasticsearch 客户端，负责连接、查询，并在不可达时提供 Excel 降级查询"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # 支持从 env 或 config 读取 Hosts
        hosts_str = self.config.get("hosts") or os.getenv("ES_HOSTS", "http://127.0.0.1:9200")
        self.hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]
        self.username = self.config.get("username") or os.getenv("ES_USERNAME")
        self.password = self.config.get("password") or os.getenv("ES_PASSWORD")
        self.verify_certs = str(self.config.get("verify_certs") or os.getenv("ES_VERIFY_CERTS", "false")).lower() == "true"

        # 索引配置
        self.indices = {
            "consulting": self.config.get("index_consulting") or os.getenv("ES_INDEX_CONSULTING", "icir_news"),
            "policy": self.config.get("index_policy") or os.getenv("ES_INDEX_POLICY", "icir_policy"),
            "report": self.config.get("index_report") or os.getenv("ES_INDEX_REPORT", "icir_report"),
            "topic": self.config.get("index_topic") or os.getenv("ES_INDEX_TOPIC", "icir_topic"),
            "patent": self.config.get("index_patent") or os.getenv("ES_INDEX_PATENT", "icir_patent")
        }

    def _get_client(self):
        """建立真实 Elasticsearch 客户端连接"""
        from elasticsearch import Elasticsearch

        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)

        return Elasticsearch(
            self.hosts,
            basic_auth=auth,
            verify_certs=self.verify_certs,
            request_timeout=30  # 延长超时时间以支持更复杂的查询
        )

    def test_connection(self) -> bool:
        """测试连接是否畅通"""
        try:
            es = self._get_client()
            # ping 接口测试
            connected = es.ping()
            es.close()
            return bool(connected)
        except Exception as e:
            logger.warning(f"Elasticsearch 连接失败 ({self.hosts}): {e}")
            return False

    def search_documents(self, index_name: str, dsl: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行 ES 检索查询，返回 hits list"""
        logger.info(f"执行 ES 检索，索引: {index_name}")
        es = self._get_client()
        try:
            response = es.search(index=index_name, body=dsl)
            hits = response.get("hits", {}).get("hits", [])
            return hits
        finally:
            es.close()

    def search_count(self, index_name: str, dsl: Dict[str, Any]) -> int:
        """执行 ES count 查询，返回匹配的总文档数"""
        es = self._get_client()
        try:
            # 统计查询忽略 size, sort, search_after 等分页字段
            count_dsl = {k: v for k, v in dsl.items() if k in ("query", "aggs")}
            response = es.count(index=index_name, body=count_dsl)
            return response.get("count", 0)
        finally:
            es.close()

    def mock_query_documents(
        self,
        raw_source_type: str,
        keywords: List[str],
        synonyms: List[str],
        exclude_terms: List[str],
        start_date: str,
        end_date: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        [降级轨道] 从本地 Excel 离线匹配并按 ES hit 结构输出。
        """
        logger.info(f"ES 降级运行：从本地 [{raw_source_type}] Excel 中检索数据")

        # 1. 查找合适的数据源文件
        data_dir = Path(os.getenv("WEAK_SIGNAL_MOCK_DATA_DIR", str(Config.DATA_DIR))).expanduser()
        file_path = None

        if raw_source_type == "consulting":
            file_path = next(data_dir.glob("*资讯*.xlsx"), None)
        elif raw_source_type == "report":
            file_path = next(data_dir.glob("*研报*.xlsx"), None)
        elif raw_source_type == "patent":
            file_path = next(data_dir.glob("*专利*.xlsx"), None)
        elif raw_source_type == "policy":
            file_path = next(data_dir.glob("*资讯*.xlsx"), None) # 政策使用资讯代替
        elif raw_source_type == "topic":
            # mock 话题使用文献的 keywords 进行话题字段生成
            file_path = next(data_dir.glob("*文献*.xlsx"), None)

        if not file_path or not file_path.exists():
            logger.warning(f"未找到本地 [{raw_source_type}] 测试文件")
            return []

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"加载本地测试数据失败: {e}")
            return []

        if df.empty:
            return []

        # 确定日期列
        date_col = next((c for c in ["publish_date", "public_date", "apply_date", "year"] if c in df.columns), None)

        # 2. 模拟时间过滤
        if date_col and (start_date or end_date):
            df['standard_date'] = pd.to_datetime(df[date_col].astype(str) + ('-01-01' if date_col == 'year' else ''), errors='coerce')
            if start_date:
                sd = pd.to_datetime(start_date, errors='coerce')
                if not pd.isna(sd):
                    df = df[df['standard_date'] >= sd]
            if end_date:
                ed = pd.to_datetime(end_date, errors='coerce')
                if not pd.isna(ed):
                    df = df[df['standard_date'] <= ed]

        # 3. 模拟全匹配关键字 (TF-IDF 相似度打分模拟)
        search_terms = [t.lower().strip() for t in (list(keywords) + list(synonyms)) if t.strip()]

        # 收集检索列
        search_cols = [c for c in ["title", "title_cn", "abstract", "abstract_cn", "first_claim", "html", "content", "venue", "keywords"] if c in df.columns]

        scores = []
        for idx, row in df.iterrows():
            row_score = 0.0
            row_text_lower = " ".join([str(row[c]).lower() for c in search_cols])

            # 计算包含词数量作为相关性打分
            for term in search_terms:
                if term in row_text_lower:
                    # 标题中匹配权重更高
                    title_col = next((c for c in ["title", "title_cn"] if c in df.columns), None)
                    if title_col and term in str(row[title_col]).lower():
                        row_score += 4.0
                    else:
                        row_score += 1.0
            scores.append(row_score)

        df['mock_score'] = scores

        # 如果提供了检索词，则必须匹配至少一个 (mock_score > 0)
        if search_terms:
            df = df[df['mock_score'] > 0]
        else:
            df['mock_score'] = 1.0 # 默认分

        # 4. 模拟排除词
        ex_terms = [t.lower().strip() for t in exclude_terms if t.strip()]
        if ex_terms:
            for term in ex_terms:
                ex_mask = pd.Series([False] * len(df), index=df.index)
                for col in search_cols:
                    ex_mask = ex_mask | df[col].astype(str).str.lower().str.contains(term, na=False, regex=False)
                df = df[~ex_mask]

        # 5. 排序：模拟相关度排序，相同分数按时间降序
        sort_by = ['mock_score']
        ascending = [False]
        if date_col:
            sort_by.append(date_col)
            ascending.append(False)

        df = df.sort_values(by=sort_by, ascending=ascending)

        # 6. 截断限制
        df = df.head(limit)

        # 7. 转成类似 ES 的 Hit 结构
        hits = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            # 清理 float NaN 为 None
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items() if k not in ('mock_score', 'standard_date')}

            # 补齐 ID 确保在 ES 去重时通过
            doc_id = str(row_dict.get("id", row_dict.get("doc_id", "")))
            if not doc_id or doc_id == 'None':
                title_val = str(row_dict.get("title_cn", row_dict.get("title", "")))
                doc_id = hashlib.md5(title_val.encode('utf-8')).hexdigest()[:16]
                row_dict["id"] = doc_id

            hits.append({
                "_id": doc_id,
                "_score": float(row.get("mock_score", 1.0)),
                "_source": row_dict
            })

        return hits
