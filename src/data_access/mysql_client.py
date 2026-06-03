import os
import logging
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

from ..utils.config import Config

# 配置日志
logger = logging.getLogger("weak_signal.data_access.mysql")

class MySQLClient:
    """MySQL 客户端，负责连接、查询，并在不可达或 MOCK 模式时提供 Excel 降级查询"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.host = self.config.get("host") or os.getenv("MYSQL_HOST", "127.0.0.1")
        self.port = int(self.config.get("port") or os.getenv("MYSQL_PORT", 3306))
        self.database = self.config.get("database") or os.getenv("MYSQL_DATABASE", "weak_signal")
        self.user = self.config.get("user") or os.getenv("MYSQL_USER", "weak_signal_reader")
        self.password = self.config.get("password") or os.getenv("MYSQL_PASSWORD", "change_me")
        self.charset = self.config.get("charset") or os.getenv("MYSQL_CHARSET", "utf8mb4")
        self.table_name = self.config.get("table_literature") or os.getenv("MYSQL_TABLE_LITERATURE", "dw_science_literature")

    def _get_connection(self):
        """建立真实 MySQL 连接，失败时抛出异常"""
        import pymysql
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=3  # 快速超时，防止阻塞 Web UI
        )

    def test_connection(self) -> bool:
        """测试连接是否畅通"""
        try:
            conn = self._get_connection()
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"MySQL 连接失败 ({self.host}:{self.port}): {e}")
            return False

    def query_documents(self, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行 SQL 参数化查询，并返回 dict 列表"""
        logger.info(f"执行 MySQL 检索，表名: {self.table_name}")
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 转换 SQL 中的参数格式（若有必要，有些驱动可能使用 %s，PyMySQL 支持 %(name)s 字典占位符）
                cursor.execute(sql, params)
                return cursor.fetchall()
        finally:
            conn.close()

    def query_count(self, sql: str, params: Dict[str, Any]) -> int:
        """执行统计 SQL 并返回总数"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                res = cursor.fetchone()
                if res:
                    return res.get("total", 0)
                return 0
        finally:
            conn.close()

    def mock_query_documents(
        self,
        keywords: List[str],
        synonyms: List[str],
        exclude_terms: List[str],
        start_date: str,
        end_date: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        [降级轨道] 从本地 Excel (文献测试数据.xlsx) 中过滤数据，
        模拟相同的时间过滤、关键词匹配和排除词逻辑。
        """
        logger.info("MySQL 降级运行：从本地文献 Excel 中检索数据")

        # 查找本地文献测试文件
        data_dir = Path(os.getenv("WEAK_SIGNAL_MOCK_DATA_DIR", str(Config.DATA_DIR))).expanduser()
        file_path = next(data_dir.glob("*文献*.xlsx"), None)
        if not file_path or not file_path.exists():
            logger.warning("未找到本地文献测试文件，返回空数据集")
            return []

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"加载本地文献测试数据失败: {e}")
            return []

        if df.empty:
            return []

        # 1. 模拟时间过滤 (year 字段标准化)
        # year 在文献数据中为整数，如 2021
        if start_date or end_date:
            df['standard_date'] = pd.to_datetime(df['year'].astype(str) + '-01-01', errors='coerce')
            if start_date:
                sd = pd.to_datetime(start_date, errors='coerce')
                if not pd.isna(sd):
                    df = df[df['standard_date'] >= sd]
            if end_date:
                ed = pd.to_datetime(end_date, errors='coerce')
                if not pd.isna(ed):
                    df = df[df['standard_date'] <= ed]

        # 2. 模拟匹配关键词和同义词
        search_terms = [t.lower().strip() for t in (list(keywords) + list(synonyms)) if t.strip()]
        if search_terms:
            mask = pd.Series([False] * len(df), index=df.index)
            # 全文本匹配字段: title, abstract, keywords, authors
            for term in search_terms:
                term_mask = (
                    df['title'].astype(str).str.lower().str.contains(term, na=False, regex=False) |
                    df['abstract'].astype(str).str.lower().str.contains(term, na=False, regex=False) |
                    df['keywords'].astype(str).str.lower().str.contains(term, na=False, regex=False) |
                    df['authors'].astype(str).str.lower().str.contains(term, na=False, regex=False)
                )
                mask = mask | term_mask
            df = df[mask]

        # 3. 模拟排除词
        ex_terms = [t.lower().strip() for t in exclude_terms if t.strip()]
        if ex_terms:
            for term in ex_terms:
                ex_mask = (
                    df['title'].astype(str).str.lower().str.contains(term, na=False, regex=False) |
                    df['abstract'].astype(str).str.lower().str.contains(term, na=False, regex=False)
                )
                df = df[~ex_mask]

        # 4. 排序 (year 降视)
        if 'year' in df.columns:
            df = df.sort_values(by='year', ascending=False)

        # 5. 限制数量
        df = df.head(limit)

        # 6. 转成与 MySQL 查询结果对应的 dict 字段格式
        results = []
        for idx, row in df.iterrows():
            results.append({
                "id": row.get("doi/arxiv_id") or f"paper_mock_{idx}",
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "content": row.get("abstract", ""),  # mock 中以 abstract 兼正文
                "publish_date": f"{row.get('year', 2021)}-01-01",
                "author_org": row.get("affiliations", ""),
                "authors": row.get("authors", ""),
                "keywords": row.get("keywords", ""),
                "doi": row.get("doi/arxiv_id", ""),
                "journal": row.get("venue", ""),
                "classification": row.get("label", "")
            })

        return results
