# 产业技术预见智能体

面向产业技术预见场景的弱信号识别系统。项目从专利、文献、研报、资讯等多源文本中抽取事件并同步完成事件质量评分，形成候选技术对象，完成弱信号评分、主题细化、反向验证、时间验证、持续观测分层和报告生成。

当前主线是自研 Pipeline 架构，不再保留早期 LangChain/RAG 原型流程。

## 当前版本

- 当前发布版本：`0.04`
- 目标发布标签：`v0.04`
- v0.04 主线聚焦 Domain Pack 通用化、生成复核、领域隔离、候选命名质量防护和信号生成诊断，继续保留弱信号识别、时间验证与持续观测主流程。
- 本版新增功能说明：
  - Domain Pack 生成流程升级为分阶段缓存、结构化修复、基础校验与运行包复用，生成结果保存到 `memory/domain_packs/`。
  - Web 入口新增领域运行包生成、preset 选择、历史运行包载入、规则预览、数据源推荐数量和人工确认闸口。
  - 事件抽取、事件质量、候选成形和信号生成统一读取 Domain Pack 领域锚点、排除词、壳词和候选范式，减少默认人形机器人规则泄漏。
  - 候选命名增加壳词/泛词防护、离域过滤、技术名回退与诊断字段，避免将应用场景或通用方法误提升为弱信号名称。
  - 分析结果保存 Domain Pack 快照和 `signal_generation_diagnostics.json`，历史续跑与报告重生成可恢复运行包上下文。
  - 新增 Phase 8/9 泛化回归、legacy 清理审计、数据源推荐数量和候选质量防护相关测试。
- v2.6 验收参考目录：`result/20260430_100821_v26_final_validation`

## 项目架构

```text
tf_agent/
├── main.py                    # CLI 入口
├── web_app.py                 # Streamlit Web 入口
├── requirements.txt
├── data/                      # 输入数据样例与本地数据
├── memory/                    # 运行缓存、上传文件、事件缓存
├── result/                    # 分析输出
└── src/
    ├── core/
    │   ├── agent.py           # 对外智能体封装
    │   └── pipeline.py        # 主分析流水线
    ├── domain/                # Domain Pack schema、生成、校验、复核与保存
    ├── extraction/
    │   ├── event_schema.py       # 事件 schema 契约、回填与读写工具
    │   ├── event_extractor.py
    │   ├── candidate_former.py
    │   └── tech_lexicon.py
    ├── scoring/
    │   ├── scorer.py
    │   ├── signal_generator.py
    │   └── topic_refiner.py
    ├── validation/
    │   ├── reverse_validator.py
    │   ├── event_quality.py
    │   ├── temporal_validator.py
    │   ├── object_family_canonicalizer.py
    │   └── family_evaluator.py
    ├── utils/
    │   ├── api_stats.py
    │   ├── config.py
    │   ├── env_config.py
    │   ├── env_utils.py
    │   ├── llm_client.py
    │   └── semantic_utils.py
    ├── config/
    │   ├── domain_packs/      # preset Domain Pack，例如 neutral 与 humanoid_robot
    │   ├── object_family_canonicalization.yaml
    │   └── object_family_registry.yaml
    └── data/
        └── api_stats.json
```

## 主流程

1. 数据加载与标准化
2. 事件抽取、结构化与质量评分
3. 候选技术对象成形与候选证据质量聚合
4. 弱信号评分
5. 主题细化
6. 反向验证
7. 时间验证与持续观测分层
8. 信号生成
9. 报告生成

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

命令行运行：

```bash
python main.py
python main.py --sample 100
python main.py --data data/your_file.xlsx
python main.py --from-result result/20260424_092011
python main.py --backfill-events result/old/events.json
```

Web 页面运行：

```bash
streamlit run web_app.py
```

## Domain Pack 工作流

系统采用“通用主流程 + 领域运行包 Domain Pack”的运行方式。主流程不默认任何具体技术领域；缺失 Domain Pack 时只使用 `neutral` pack。人形机器人经验已经迁移为 `src/config/domain_packs/humanoid_robot.yaml` preset，只有用户显式选择该 preset 时才启用对应旧规则。

Domain Pack 的推荐使用流程：

1. 生成：在 Web 入口输入任意技术领域、关键词、排除词和数据源后，点击生成 Domain Pack。运行期生成结果保存到 `memory/domain_packs/`，不写入 `src/config/`。
2. 校验：生成结果必须通过 schema validator；字段缺失、候选壳层词比例过高、dry-run 报告缺失或 hash 不一致时不得进入主流程。
3. 复核：Web 页面展示 Domain Pack 摘要、推荐数据源数量、dry-run 质量结论和人工确认状态。用户编辑后必须重新 dry-run，或显式标记 `manual_override`。
4. 复用：可以从 `memory/domain_packs/` 复用已有运行包，或显式选择源码 preset。进入 Pipeline 的是同一份 `DomainContext`，后续抽取、候选、评分、主题细化、对象族归一和报告都读取它。
5. 版本化：Domain Pack 内容变化后必须重新计算 `domain_pack_hash`，并递增或重新生成 `domain_pack_version`。正式分析结果会在 `result/<run_id>/domain_pack.yaml` 保存快照，并在事件、候选、评分和报告元数据中记录 `domain_pack_id`、`domain_pack_version` 与 `domain_pack_hash`，历史续跑和报告重生成优先恢复该快照。

## 环境变量

在 `.env` 中配置模型访问参数：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

EXTRACTION_MODEL=Qwen/Qwen2.5-14B-Instruct
AGENT_MODEL=Qwen/Qwen2.5-32B-Instruct
REPORT_MODEL=deepseek-ai/DeepSeek-V3

# 可选：控制单篇文档最多保留的抽取事件数与最低置信度
EVENT_EXTRACTION_MAX_EVENTS_PER_DOC=6
EVENT_EXTRACTION_MIN_CONFIDENCE=0
EVENT_EXTRACTION_RETRY_EMPTY_BATCH=1
EVENT_EXTRACTION_BATCH_TIMEOUT=60
EVENT_EXTRACTION_SINGLE_TIMEOUT=45
EVENT_EXTRACTION_BATCH_RETRIES=0
EVENT_EXTRACTION_TIMEOUT_FALLBACK_TO_LOCAL=1
```

## 事件抽取 schema

事件抽取主契约集中在 `src/extraction/event_schema.py`，当前版本为 `weak_signal_event_v2`。新版事件在兼容 `subject/action/technology/scene/time` 的基础上，新增 `technical_object`、`mechanism`、`task`、`data_modality`、`method`、`evidence_span`、`confidence`、`weak_signal_reason` 等面向候选成形和证据复核的字段。

- 抽取结果支持 0-N 个事件，批量 API 返回需携带 `doc_index` 以映射原文。
- 若批量 API 对整批返回空数组 `[]`，默认会逐条重试一次；可通过 `EVENT_EXTRACTION_RETRY_EMPTY_BATCH=0` 关闭。
- 批量 API 超时会按 `EVENT_EXTRACTION_BATCH_RETRIES` 重试，仍失败时默认退回本地规则兜底，避免单批长时间卡住；如需继续逐条 API 兜底，可设置 `EVENT_EXTRACTION_TIMEOUT_FALLBACK_TO_LOCAL=0`。
- 历史 `events.json/csv` 会通过 schema 回填补齐缺失字段，并标记 `schema_migration_mode=legacy_backfill`。
- `--backfill-events` 可单独生成补齐后的事件文件及摘要 JSON。
- 事件质量评分已纳入 `evidence_span_score`、`confidence_score`、`uncertainty_risk_score` 和 `foresight_relevance_score`，这些字段会继续进入候选证据聚合、Web 复核视图和报告证据排序。

## 目录边界

- `src/core/` 只负责流程编排和智能体封装。
- `src/extraction/` 负责从原始文本到候选对象的结构化加工。
- `src/scoring/` 负责评分、主题细化和最终信号表达。
- `src/validation/` 负责反向验证、对象族归一化、事件质量和时间持续观测验证。
- `src/utils/` 只放跨模块基础设施，例如配置、LLM 客户端、语义相似度和调用统计。
- `data/`、`memory/`、`result/` 是运行时数据目录，不放核心业务代码。

## 已清理内容

本次结构整理移除了早期 LangChain/RAG 原型文件、根目录重复模块、旧演示副本和 Python 字节码缓存。项目现在只保留一条主线：`main.py` / `web_app.py` 调用 `src/core`，再由 Pipeline 串联 extraction、scoring、validation 与 utils。
