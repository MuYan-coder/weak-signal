# 阶段 0 基线冻结与风险盘点

来源：`通用化改造实施方案.md` 的“阶段 0：基线冻结与风险盘点”。
记录日期：2026-06-02。

本文件用于在 Domain Pack 通用化改造前固定当前行为。后续阶段如果迁移抽取、候选成形、评分或前端入口，应先对照这里的人形机器人基线、硬编码点和串域防护测试。

## 1. 人形机器人输出基线

### 长期全量基线

- 运行目录：`result/20260510_192455(v2.8全量抽取版)`
- 原始数据数：5490
- 事件数：5490
- 候选数：614
- signal 总数：424
- 弱信号数：81
- 被确认弱信号数：20
- 时间验证数：458
- 反向验证数：15
- 候选阶段分布：`formed_candidate` 369，`formed_candidate_strong` 242，`scope_overview` 3
- 主题粒度分布：`fine_grained_topic` 505，`scope_internal_candidate` 106，`generic_or_failed` 3
- 机器人串域过滤效果：`reverse_validation.json` 中 15 个反向验证对象均为 `natural_topic_in_source`，候选表中仅 3 个 `scope_overview` 未进入 scope 内部候选。该表现属于人形机器人 preset 需要保留的参照，不应作为新领域默认规则。

### 快速回归基线

- 运行目录：`result/20260525_193629`
- 事件数：333
- 候选数：483
- signal 总数：322
- 弱信号数：28
- 被确认弱信号数：9
- 时间验证数：381
- 反向验证数：6
- 候选阶段分布：`formed_candidate` 365，`formed_candidate_strong` 115，`scope_overview` 3
- 主题粒度分布：`fine_grained_topic` 397，`scope_internal_candidate` 83，`generic_or_failed` 3
- 机器人串域过滤效果：`reverse_validation.json` 中 6 个反向验证对象均为 `natural_topic_in_source`，可作为小样本人形机器人流程的快速迁移参照。

### 最新弱信号输出参照

- 运行目录：`result/20260601_190208`
- 事件数：152
- 候选数：183
- signal 总数：156
- 弱信号数：12
- `weak_signals.json` 数：12
- 时间验证数：22

## 2. 领域硬编码点盘点

| 文件 | 当前耦合点 | 阶段 0 结论 |
| --- | --- | --- |
| `src/extraction/tech_lexicon.py` | `TECH_ALIASES`、`OBSERVATION_SCOPE_TERMS`、机器人领域锚点、`PATENT_PROXY_SCOPE_TERMS`、world model 上下文支持规则。 | `humanoid preset 保留`：人形机器人、具身智能、世界模型别名和专利 proxy 规则。`通用范式迁移`：后续改为 Domain Pack 的 scope aliases、off-domain anchors、generic terms 和 proxy scope 规则。 |
| `src/extraction/candidate_former.py` | `SCOPE_LABELS`、generic token、scope shell token、anchor priority、机器人 display surface、机械臂/灵巧手/夹爪表面词。 | `humanoid preset 保留`：机器人对象表面词、操控/装配/抓取显示规则。`通用范式迁移`：generic terms、shell terms、valid/invalid candidate patterns 和候选命名槽位。 |
| `src/scoring/signal_generator.py` | `_OBJECT_FAMILY_SCOPE_HINTS`、机器人/AI off-domain token、材料领域临时相关词、`of_006` 拆分为 embodied simulation、robot training、robot planning。 | `humanoid preset 保留`：对象族归一和 robot training/planning/simulation 特例。`通用范式迁移`：对象族开关、alias groups、off-domain leakage terms 和领域相关词。 |
| `src/scoring/topic_refiner.py` | world model + robot 的小主题后处理会输出机器人规划、机器人训练、机器人控制等中文主题名。 | `humanoid preset 保留`：world model/robot 组合命名规则。`通用范式迁移`：小主题类型、机制、任务、数据、场景模板由 Domain Pack 提供。 |
| `web_app.py` | 领域模板分支中默认项为“人形机器人 (默认)”，并硬编码人形机器人、具身智能、世界模型等关键词和同义词。 | `humanoid preset 保留`：模板可作为 preset 表单填充值。`通用范式迁移`：默认入口改为自定义领域，模板不再作为业务默认知识。 |

额外观察点：`src/extraction/event_extractor.py`、`src/scoring/scorer.py`、`src/validation/reverse_validator.py` 和 `src/validation/event_quality.py` 也含机器人、具身智能、world model 或通用机制词。阶段 0 不迁移这些代码，但后续阶段应在接入 Domain Pack 时同步复核。

## 3. 串域防护测试归类

以下现有测试归为“串域防护测试”，后续通用化阶段必须保持或迁移为 Domain Pack 场景测试：

- `tests/test_event_extraction_refactor.py::test_candidate_forms_use_selected_domain_as_generic_scope`：非机器人材料领域应使用用户选择领域作为 scope，不注入机器人显示名。
- `tests/test_event_extraction_refactor.py::test_signal_generation_filters_to_selected_analysis_domain`：材料领域信号生成应过滤机器人规划和泛化规划壳词。
- `tests/test_event_extraction_refactor.py::test_non_robot_domain_disables_robot_family_normalization`：非机器人领域默认禁用机器人对象族归一。
- `tests/test_event_extraction_refactor.py::test_configured_local_loading_does_not_fallback_to_default_samples`：显式数据源配置为空时不得回退默认样本。
- `tests/test_data_access.py::test_db_cache_path_handles_empty_source_query_id`：自定义数据库模板留空 ID 时缓存路径仍应稳定，不把空领域误处理成默认领域。

以下测试归为“humanoid preset 保留测试”，后续迁移到 preset 后仍要通过：

- `tests/test_event_extraction_refactor.py::test_schema_fields_generate_candidate_units`
- `tests/test_event_extraction_refactor.py::test_local_process_events_outputs_v2_schema`
- `tests/test_event_extraction_refactor.py::test_subject_strict_cleaning_and_metadata_fallback`
- `tests/test_event_extraction_refactor.py::test_action_cleaning`
- `tests/test_data_access.py::test_query_builder_expands_topic_aliases`

## 4. 阶段 0 验收快照

- 当前主流程测试入口：`python -m unittest tests.test_event_extraction_refactor tests.test_data_access tests.test_phase0_generalization_baseline -v`
- 当前人形机器人结果已固定为全量基线、小样本基线和最新弱信号输出参照。
- 已明确哪些行为要保留为 `humanoid_robot` preset，哪些行为要迁移到 Domain Pack 通用范式。
- 已将非机器人领域回归测试归类为“串域防护测试”，后续阶段改造不得让这些测试失效。
