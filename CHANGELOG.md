# Changelog

## Local Maintenance - 2026-06-02

### 新增

- 新增根目录本地维护文档：`PROJECT_OVERVIEW.md`、`ARCHITECTURE.md`、`TASKS.md`、`DECISIONS.md`、`KNOWN_ISSUES.md`。
- 在 `.gitignore` 中追加本地维护文档忽略规则，避免新建维护文档进入后续提交。

### 说明

- `CHANGELOG.md` 当前已经是 Git 跟踪文件，`.gitignore` 不会自动忽略它的后续改动；如需彻底转为本地维护文档，需要单独执行索引移除操作。

## v0.02 - 2026-05-25

### 新增

- 新增独立事件 schema 契约 `src/extraction/event_schema.py`，统一维护 `weak_signal_event_v2` 字段、缓存列、标准化、历史回填和 schema 摘要。
- 事件抽取支持单篇文档 0-N 个事件，并通过 `EVENT_EXTRACTION_MAX_EVENTS_PER_DOC` 与 `EVENT_EXTRACTION_MIN_CONFIDENCE` 控制多事件膨胀。
- 批量事件抽取遇到整批空数组 `[]` 时默认逐条重试一次，可用 `EVENT_EXTRACTION_RETRY_EMPTY_BATCH=0` 关闭。
- 批量事件抽取支持 `EVENT_EXTRACTION_BATCH_RETRIES`、`EVENT_EXTRACTION_BATCH_TIMEOUT`、`EVENT_EXTRACTION_SINGLE_TIMEOUT` 和 `EVENT_EXTRACTION_TIMEOUT_FALLBACK_TO_LOCAL`；默认批量超时后快速退回本地规则兜底，避免单批长时间卡住。
- CLI 新增 `--backfill-events` 与 `--backfill-output`，可将历史 `events.json/csv` 补齐为新版 schema 并输出摘要。
- 事件质量评分新增 `evidence_span_score`、`confidence_score`、`uncertainty_risk_score` 和 `foresight_relevance_score`。
- 候选证据聚合新增 `candidate_evidence_foresight_relevance` 与 `high_foresight_evidence_count`。
- 时间验证新增持续观测字段，包括 `monitoring_priority`、`monitoring_action` 和下一观测窗口，用于弱信号后续跟踪。

### 改进

- API 与本地回退抽取均输出新版字段，包括 `technical_object`、`mechanism`、`task`、`data_modality`、`method`、`evidence_span`、`confidence` 和 `weak_signal_reason`。
- 候选成形优先利用新版 schema 字段生成对象-机制-任务候选单元，同时保留旧规则候选作为兼容补充。
- `from-events` 路径自动回填历史事件 schema，避免旧事件文件阻断后续评分、候选成形和报告生成。
- 报告证据排序优先考虑预见相关度、事件质量、证据片段质量和置信度，降低空泛或低质量证据进入核心报告的概率。
- Web 页面补充展示新版事件字段、事件质量细分分数和候选证据预见相关度，便于人工复核。
- 主流程移除当前项目暂不需要的技术链映射、关键核心潜力评分、关键核心候选短名单、频次基线和基线对照产物。
- 报告结构改为弱信号持续观测导向，不再输出关键核心技术候选章节。

## v2.7.0 - 2026-05-07

本次发布在 v2.6 质量控制和技术链映射基础上，新增时间验证与关键核心潜力评分，用于从弱信号候选中筛出更适合人工复核的关键核心技术潜力对象。

### 新增

- 新增时间验证模块 `src/validation/temporal_validator.py`，按候选汇总首末出现时间、观察窗口、增长率、来源扩散、主体扩散和时间验证层级。
- 新增关键核心潜力评分模块 `src/scoring/key_core_scorer.py`，综合弱信号强度、时间增长、技术链卡点、战略重要性、证据质量、资产支撑和门槛项。
- 新增输出产物 `temporal_validation.json/csv`、`key_core_scored.json/csv` 和 `key_core_candidates.json/csv`。
- 报告证据包和规则报告新增关键核心候选摘要，避免把弱信号分直接等同于关键核心结论。
- Web 新增时间验证与关键核心候选阶段、Tab、历史结果加载和报告产物查看入口。

### 改进

- Pipeline 在技术链映射后串联时间验证和关键核心潜力评分，再生成弱信号、短名单和 grounded 报告。
- `from-events` 与 `from-result` 路径可补算或读取 v2.7 新增产物。
- 信号生成保留时间验证字段和关键核心评分字段，便于后续导出、报告和人工复核。
- 版本号升级至 `2.7.0`，README 主流程同步更新。

### 验收

- `python -m compileall src main.py web_app.py` 通过。
- 已执行历史结果烟测，验证时间验证、关键核心评分和候选短名单构建链路可运行。

### 剩余风险

- 关键核心潜力评分仍是候选优先级排序，不替代专家确认。
- 时间验证受输入数据日期覆盖影响，日期缺失样本会进入保守层级。
- 资产支撑目前仍依赖轻量 CSV 先验，后续可继续扩充人工校准表。

## v2.6.0 - 2026-05-07

本次发布将系统从 v2.5 弱信号识别主线升级为具备证据质量控制和产业技术链对齐能力的 v2.6 主线。

### 新增

- 新增事件质量评分模块 `src/validation/event_quality.py`，输出事件质量分、质量层级和可解释原因。
- 新增候选证据质量聚合字段，包括 `candidate_evidence_quality`、`candidate_core_evidence_quality`、`low_quality_evidence_ratio`、`high_quality_evidence_count`、`quality_risk_flag` 和 `quality_adjusted_rank_score`。
- 新增轻量技术链映射模块 `src/validation/tech_chain_mapper.py`，支持 exact、alias、term、semantic 和 no-match 映射路径。
- 新增 `data/tech_chain/` 技术链先验数据底座，包括节点、术语、资产和维护说明。
- 新增 `event_quality.json/csv` 与 `tech_chain_mapping.json/csv` 输出产物。
- Web 增加事件质量评分、技术链映射阶段和结果 Tab，历史结果加载同步支持新增产物。

### 改进

- Pipeline 串联事件质量评分、候选质量聚合、技术链映射、最终短名单和报告证据增强。
- `from-events` 和 `from-result` 路径兼容 v2.6 新增产物。
- 信号生成、最终短名单和报告证据包保留证据质量与技术链映射字段。
- 报告生成优先过滤低质量核心证据，并避免将 no-match 或低置信 broader match 写成确定性关键核心结论。

### 验收

参考验收目录：`result/20260430_100821_v26_final_validation`。

- 事件数：83，事件质量记录数：83。
- 候选数：195，评分候选数：195。
- Top 20 核心证据平均质量分：8.99，高于 6.5 门槛。
- Top 50 技术链映射覆盖率：100%，高于 70% 门槛。
- 报告核心证据中未发现质量分低于 4 的事件。
- `python -m compileall src main.py web_app.py` 通过。

### 剩余风险

- 首批技术链先验仍是轻量数据底座，后续需要继续扩充节点和术语覆盖范围。
- v2.6 不直接输出关键核心技术综合评分，该能力保留到 v2.7 及后续版本。
- 大样本排序质量仍需要持续通过人工复核和新增数据集回归验证。
