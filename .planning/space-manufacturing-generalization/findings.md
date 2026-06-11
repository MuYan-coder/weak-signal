# 太空制造领域候选成形泛化发现

## 已确认事实

- `candidate_former.py` 仍包含多组人形机器人/具身智能预设常量，例如 `SCOPE_LABELS`、`GENERIC_METHOD_ONLY_DISPLAY_NAMES`、`TECH_OBJECT_GENERIC_LABELS`、`GENERIC_TECH_OBJECT_SLOTS`、`DISPLAY_OBJECT_SURFACE_PATTERNS`、`DISPLAY_TASK_SURFACE_PATTERNS`。
- 这些常量不只是抽取词，也会影响展示名、聚类签名、候选具体性、scope-shell 判断和候选强弱阶段。
- `candidate_eligibility.py` 还有独立的 `TECHNICAL_ANCHOR_TERMS`，会影响非机器人领域是否被认为有技术锚点。
- `scorer.py` 还有独立的 `SCOPE_SHELL_*` 常量和 `_scope_shell_profile()` 复判逻辑，可能在候选成形后再次把太空制造候选降级。
- 现有测试已经包含 `test_no_default_robot_leakage.py`、`test_candidate_eligibility.py`、`test_scoring_score_suppression.py` 和太空制造相关片段，可作为改造基础。
- `memory/domain_packs/` 中存在多个太空制造运行包，但它们是运行产物，不应作为源码测试的唯一依赖。

## 设计结论

- 需要新增 `DomainCandidatePolicy`，把领域相关的候选形成规则从硬编码常量中抽象出来。
- 人形机器人常量应变成 legacy profile，而不是默认 profile。
- `candidate_former.py`、`candidate_eligibility.py`、`scorer.py` 必须使用同一领域策略，否则候选成形修好后仍会被资格判断或评分复判污染。

## 风险

- 该文件私有函数较多，直接改签名容易影响内部测试。每个新增参数都应默认 `None` 并保留旧常量 fallback。
- 如果一次性重写聚类逻辑，可能影响现有候选数量和报告结构。计划采用门控和策略适配，不重写聚类算法。
- `DomainPack` schema 当前没有专门的 surface pattern 字段，因此非人形领域第一阶段应禁用人形 surface pattern，而不是立即扩 schema。
