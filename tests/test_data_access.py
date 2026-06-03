import sys
from pathlib import Path
import pandas as pd
import unittest

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_access.models import SourceQuery, DocumentRecord
from src.data_access.query_builder import QueryBuilder
from src.data_access.normalizer import DocumentNormalizer
from src.data_access.repository import DataRepository
from src.data_access.mysql_client import MySQLClient
from src.data_access.es_client import ESClient

class DataAccessTests(unittest.TestCase):
    """数据访问层单元测试"""

    def test_source_query_creation(self):
        """测试 SourceQuery 能否正确被实例化"""
        query = SourceQuery(
            tech_field_id="humanoid_robot",
            tech_field_name="人形机器人",
            keywords=["人形机器人", "灵巧手"],
            synonyms=["双足机器人"],
            exclude_terms=["招聘", "广告"],
            start_date="2023-01-01",
            end_date="2025-12-31",
            source_types=["paper", "news", "patent"],
            counts={"paper": 10, "news": 20, "patent": 15}
        )
        self.assertEqual(query.tech_field_id, "humanoid_robot")
        self.assertIn("灵巧手", query.keywords)
        self.assertEqual(query.counts["paper"], 10)

    def test_query_builder_mysql(self):
        """测试 QueryBuilder 构建 MySQL SQL"""
        query = SourceQuery(
            tech_field_id="humanoid_robot",
            tech_field_name="人形机器人",
            keywords=["机器人"],
            exclude_terms=["招聘"],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        sql, params = QueryBuilder.build_mysql_query(query, table_name="test_table", limit=50)

        self.assertIn("FROM test_table", sql)
        self.assertIn("event_time >= %(start_date)s", sql)
        self.assertIn("LIMIT %(limit)s", sql)
        self.assertEqual(params["start_date"], "2023-01-01")
        self.assertEqual(params["limit"], 50)
        self.assertIn("%机器人%", params.values())
        self.assertIn("%招聘%", params.values())

    def test_query_builder_es(self):
        """测试 QueryBuilder 构建 ES DSL"""
        query = SourceQuery(
            tech_field_id="humanoid_robot",
            tech_field_name="人形机器人",
            keywords=["机器人"],
            exclude_terms=["招聘"],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        dsl = QueryBuilder.build_es_dsl(query, date_field="public_date", size=20)

        self.assertEqual(dsl["size"], 20)
        self.assertIn("public_date", dsl["query"]["bool"]["filter"][0]["range"])
        self.assertEqual(dsl["query"]["bool"]["filter"][0]["range"]["public_date"]["gte"], "2023-01-01")

        # 验证 sort
        sort_fields = [list(item.keys())[0] for item in dsl["sort"]]
        self.assertIn("public_date", sort_fields)
        self.assertIn("_score", sort_fields)
        self.assertNotIn("_id", sort_fields)

        dsl_with_tie_breaker = QueryBuilder.build_es_dsl(
            query,
            date_field="public_date",
            size=20,
            search_after=[1.0, "2023-12-31"],
            tie_breaker_field="id.keyword",
        )
        sort_fields_with_tie_breaker = [list(item.keys())[0] for item in dsl_with_tie_breaker["sort"]]
        self.assertIn("id.keyword", sort_fields_with_tie_breaker)

    def test_query_builder_expands_topic_aliases(self):
        """测试启用话题词库时会扩展本地技术别名"""
        query = SourceQuery(
            tech_field_id="humanoid_robot",
            tech_field_name="人形机器人",
            keywords=["人形机器人"],
            use_topic_index=True,
        )

        terms = QueryBuilder.expand_search_terms(query)
        self.assertIn("humanoid robot", terms)

        disabled_query = SourceQuery(
            tech_field_id="humanoid_robot",
            tech_field_name="人形机器人",
            keywords=["人形机器人"],
            use_topic_index=False,
        )
        disabled_terms = QueryBuilder.expand_search_terms(disabled_query)
        self.assertEqual(disabled_terms, ["人形机器人"])

    def test_normalizer_deduplicate(self):
        """测试 DocumentNormalizer 去重逻辑"""
        records = [
            # 同源重复
            DocumentRecord(
                id="news:123", source_id="123", source_type="news", raw_source_type="consulting",
                title="机器人大脑升级", text="内容 A", date="2025-01-01", url="http://test.com/1",
                relevance_score=1.5, data_backend="es"
            ),
            DocumentRecord(
                id="news:123", source_id="123", source_type="news", raw_source_type="consulting",
                title="机器人大脑升级", text="内容 A", date="2025-01-01", url="http://test.com/1",
                relevance_score=1.5, data_backend="es"
            ),
            # 跨源标题去重且保留完整的一条
            DocumentRecord(
                id="news:456", source_id="456", source_type="news", raw_source_type="consulting",
                title="双足机器人发布", text="内容 B", date="2025-01-02", url="http://test.com/2",
                org="A公司", relevance_score=1.0, data_backend="es"
            ),
            DocumentRecord(
                id="patent:789", source_id="789", source_type="patent", raw_source_type="patent",
                title=" 双足 机器人 发布！", text="更丰富的内容更完整的信息描述更长更饱满", date="2025-01-03", url="http://test.com/2",
                org="A公司", authors="发明人", relevance_score=2.0, data_backend="es"
            )
        ]

        # 执行去重
        deduped = DocumentNormalizer.deduplicate(records)

        # 期待结果：同源去重后剩3个，跨源去重后剩2个 (并且双足机器人发布应该保留后者，因为字段更多更完整、日期更新、得分更高)
        self.assertEqual(len(deduped), 2)

        titles = [r.title for r in deduped]
        self.assertIn("机器人大脑升级", titles)
        # 应该保留的是包含感叹号或修饰词的 patent 记录
        self.assertTrue(any("双足 机器人 发布！" in t for t in titles))

    def test_normalizer_sorting(self):
        """测试 DocumentNormalizer 各种排序与采样"""
        records = [
            DocumentRecord(id="n1", source_id="n1", source_type="news", raw_source_type="c", title="T1", text="X", date="2025-01-01", relevance_score=1.0),
            DocumentRecord(id="n2", source_id="n2", source_type="news", raw_source_type="c", title="T2", text="X", date="2025-01-02", relevance_score=2.0),
            DocumentRecord(id="p1", source_id="p1", source_type="patent", raw_source_type="p", title="T3", text="X", date="2025-01-03", relevance_score=1.5),
            DocumentRecord(id="p2", source_id="p2", source_type="patent", raw_source_type="p", title="T4", text="X", date="2025-01-04", relevance_score=0.5)
        ]

        # 采样截断：每种最多 1 个
        counts = {"news": 1, "patent": 1}

        # 1. 相关性排序
        merged_rel = DocumentNormalizer.sample_and_sort(records, counts, sort_mode="relevance")
        self.assertEqual(len(merged_rel), 2)
        # news 应该选得分更高的 n2 (score 2.0)，patent 应该选得分更高的 p1 (score 1.5)
        # 整体排序：n2 (2.0) -> p1 (1.5)
        self.assertEqual(merged_rel[0].id, "n2")
        self.assertEqual(merged_rel[1].id, "p1")

        # 2. 时间排序
        merged_date = DocumentNormalizer.sample_and_sort(records, counts, sort_mode="date")
        self.assertEqual(len(merged_date), 2)
        # 整体时间排序：p1 (2025-01-03) -> n2 (2025-01-02)
        self.assertEqual(merged_date[0].id, "p1")
        self.assertEqual(merged_date[1].id, "n2")

        # 3. 平衡轮转
        merged_bal = DocumentNormalizer.sample_and_sort(records, counts, sort_mode="balanced")
        self.assertEqual(len(merged_bal), 2)
        # 轮流从 news(n2) 和 patent(p1) 各取一个
        self.assertEqual({merged_bal[0].id, merged_bal[1].id}, {"n2", "p1"})

    def test_repository_mock_load(self):
        """测试 DataRepository 模拟本地数据加载"""
        repo = DataRepository(MySQLClient(), ESClient(), backend="mock")

        query = SourceQuery(
            tech_field_id="embodied_ai",
            tech_field_name="具身智能",
            keywords=["具身智能", "机器人"],
            start_date="2020-01-01",
            end_date="2026-12-31",
            source_types=["paper", "news", "patent", "report"],
            counts={"paper": 2, "news": 2, "patent": 2, "report": 2},
            sort="relevance"
        )

        # 检查数量获取
        source_counts = repo.get_source_counts(query)
        self.assertIsInstance(source_counts, dict)
        self.assertIn("paper", source_counts)

        # 执行加载
        df = repo.load_documents(query)

        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            # 验证核心字段契约
            for col in ["id", "source_id", "source_type", "raw_source_type", "title", "text", "date"]:
                self.assertIn(col, df.columns)

            # 验证采样上限约束
            type_counts = df["source_type"].value_counts().to_dict()
            for s_type, max_cnt in query.counts.items():
                self.assertLessEqual(type_counts.get(s_type, 0), max_cnt)

    def test_repository_load_backfilling(self):
        """测试 DataRepository 的动态倍数补齐机制是否正常工作"""
        repo = DataRepository(MySQLClient(), ESClient(), backend="mock")

        query = SourceQuery(
            tech_field_id="embodied_ai",
            tech_field_name="具身智能",
            keywords=["机器人"],
            start_date="2020-01-01",
            end_date="2026-12-31",
            source_types=["paper"],
            counts={"paper": 5},
            sort="relevance"
        )

        df = repo.load_documents(query)
        self.assertIsInstance(df, pd.DataFrame)
    def test_repository_load_backfilling_with_duplicates(self):
        """测试 DataRepository 的动态倍数补齐机制在存在重复时的补齐表现"""
        repo = DataRepository(MySQLClient(), ESClient(), backend="mock")

        class MockClient:
            def __init__(self, s_type):
                self.s_type = s_type
            def mock_query_documents(self, **kwargs):
                limit = kwargs.get("limit", 100)
                rows = []
                for i in range(limit):
                    idx = i if i < 5 else 4
                    if self.s_type == "paper":
                        rows.append({
                            "id": f"paper_{idx}",
                            "title": f"重复文献标题_{idx}",
                            "abstract": f"摘要内容 {i}",
                            "event_time": "2025-01-01",
                            "collect_source_name": "MySQL",
                            "authors": "作者",
                            "keyword": "关键词",
                            "url": f"http://test.com/paper_{idx}"
                        })
                    else:
                        rows.append({
                            "_id": f"doc_{idx}",
                            "_score": 1.0,
                            "_source": {
                                "doc_id": f"doc_{idx}",
                                "title": f"重复ES标题_{idx}",
                                "abstract": f"摘要内容 {i}",
                                "publish_date": "2025-01-01",
                                "url": f"http://test.com/doc_{idx}"
                            }
                        })
                return rows

        repo.mysql_client = MockClient("paper")

        query = SourceQuery(
            tech_field_id="ai",
            tech_field_name="AI",
            keywords=["AI"],
            start_date="2020-01-01",
            end_date="2026-12-31",
            source_types=["paper"],
            counts={"literature": 8},
            sort="relevance"
        )

        df = repo.load_documents(query)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)

    def test_repository_respects_public_paper_count_key(self):
        """测试 Web 端传入 paper 数量时不会退回默认 100 条"""
        class MockMySQLClient:
            def mock_query_documents(self, **kwargs):
                limit = kwargs.get("limit", 100)
                return [
                    {
                        "id": f"paper_{idx}",
                        "title": f"文献标题_{idx}",
                        "abstract": "机器人摘要",
                        "event_time": "2025-01-01",
                        "collect_source_name": "mock",
                        "authors": "作者",
                        "keyword": "机器人",
                        "url": f"http://example.com/paper_{idx}",
                    }
                    for idx in range(limit)
                ]

        class MockESClient:
            indices = {}

        repo = DataRepository(MockMySQLClient(), MockESClient(), backend="mock")
        query = SourceQuery(
            tech_field_id="robot",
            tech_field_name="机器人",
            keywords=["机器人"],
            source_types=["paper"],
            counts={"paper": 150},
        )

        df = repo.load_documents(query)
        self.assertEqual(len(df), 150)
        self.assertEqual(df["source_type"].value_counts().to_dict(), {"paper": 150})

    def test_db_cache_path_handles_empty_source_query_id(self):
        """测试自定义数据库模板留空 ID 时不会把 SourceQuery 当 dict 调用"""
        from src.core.pipeline import AnalysisPipeline

        query = SourceQuery(
            tech_field_id="",
            tech_field_name="",
            counts={"paper": 1},
        )
        cache_path = AnalysisPipeline()._event_cache_path(
            pd.DataFrame({"id": ["doc1"]}),
            Path("memory/cache"),
            source_config={"backend": "db", "query": query},
        )

        self.assertIn("db_db__", cache_path.name)


if __name__ == "__main__":
    unittest.main()
