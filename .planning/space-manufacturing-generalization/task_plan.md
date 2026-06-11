# 太空制造领域候选成形泛化改造计划

## 目标

消除太空制造运行时被人形机器人/具身智能预设词污染的问题，让候选成形、候选资格判断和弱信号评分都以当前 Domain Pack 为准。

## 当前阶段

阶段 2：新增 `DomainCandidatePolicy` 策略适配层。

## 阶段清单

- [x] 阶段 1：新增太空制造污染复现测试
- [ ] 阶段 2：新增 `DomainCandidatePolicy` 策略适配层
- [ ] 阶段 3：在 `candidate_former.py` 中按策略门控人形机器人展示词、标题锚点和桥接逻辑
- [ ] 阶段 4：将候选具体性、泛词/壳词和聚类风险判断改为 Domain Pack 驱动
- [ ] 阶段 5：将 `candidate_eligibility.py` 改为 Domain Pack 技术锚点驱动
- [ ] 阶段 6：将 `scorer.py` 的 scope-shell 复判改为领域感知
- [ ] 阶段 7：补充太空制造 Domain Pack 质量契约
- [ ] 阶段 8：运行聚焦测试、Domain Pack 回归和 diff 检查

## 关键文件

- `docs/superpowers/plans/2026-06-11-space-manufacturing-domain-policy.md`
- `src/extraction/domain_candidate_policy.py`
- `src/extraction/candidate_former.py`
- `src/scoring/candidate_eligibility.py`
- `src/scoring/scorer.py`
- `src/core/pipeline.py`
- `tests/test_space_manufacturing_domain_policy.py`

## 决策

- 保留人形机器人预设，但只允许在 `humanoid_robot` 或 legacy robot mode 下启用。
- 非人形领域默认不使用 `双臂/灵巧手/夹爪/机械臂/具身智能/world model` 等候选展示与聚类启发式。
- 太空制造等非人形领域的具体性判断来自 Domain Pack 的 `candidate_formation`。
- 不依赖 `memory/domain_packs/` 中某个具体哈希文件作为源码级测试输入。

## 验收标准

- 太空制造 formed candidates 不出现人形机器人专属展示名污染。
- 太空制造有效技术项可进入 `fine_grained_topic`。
- 太空制造 scope-only 或 policy-only 项不会进入弱信号评分通道。
- eligibility 和 scorer 都能接收并使用 `domain_context`。
- 人形机器人、Battery、新型电子材料相关回归保持通过。

## 遇到的错误

| 错误 | 尝试次数 | 处理方式 |
|------|---------|---------|
| 无 | 0 | 尚未开始实现 |
