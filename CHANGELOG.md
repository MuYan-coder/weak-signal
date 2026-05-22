# Changelog

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
