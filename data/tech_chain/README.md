# 技术链先验数据说明

本目录是 `tf_agent v2.6` 的轻量技术链数据底座，用于把弱信号候选映射到产业技术链节点。当前阶段只使用 CSV 文件，不引入图数据库或外部知识库。

## 文件说明

- `technology_nodes.csv`：技术链节点表。每行代表一个可映射的技术节点。
- `technology_terms.csv`：术语和别名表。每行代表一个可命中的术语或中英文变体，并通过 `belongs_to_tech_node_id` 回挂到节点。
- `technology_assets.csv`：技术资产和证据线索表。当前仅收录已有结果中可追溯的专利、论文、模型、软件和产品平台线索。

## 首批数据来源

首批数据不是由 LLM 直接生成，而是从以下来源整理：

1. 项目已有方案和 schema 设计。

   - `v2.6实施方案.md`
   - `关键核心技术识别项目推进方案.md`
   - `IncCore_技术域_带注释说明版.schema` 中的 Technology、TechnologyTerm、TechnologyAsset 思路

2. 当前仓库已有对象族配置。

   - `src/config/object_family_registry.yaml`
   - `src/config/object_family_canonicalization.yaml`

3. 当前离线分析结果。

   - `result/20260427_094827/signals.csv`
   - `result/20260427_094827/report_evidence_packets.json`

4. 当前离线数据集可支撑的技术对象。

   - `data/专利测试数据.xlsx`
   - `data/文献测试数据.xlsx`
   - `data/研报测试数据.xlsx`
   - `data/资讯测试数据.xlsx`
   - `data/其他类型/*.csv`

## 字段维护原则

### technology_nodes.csv

核心字段：

- `tech_node_id`：稳定节点 ID，当前使用 `tc_001` 这类编号。
- `name`：中文标准名。
- `name_en`：英文标准名。
- `alias`：别名，使用 `|` 分隔。
- `classification_code` 和 `classification_name`：轻量技术链分类。
- `parent_technology`、`depends_on`、`improves`、`cooperates_with`：轻量关系字段，保存节点 ID 列表，多个值用 `|` 分隔。
- `strategic_importance_level`：战略重要性，取值建议为 `high`、`medium`、`low`、`unknown`。
- `bottleneck_level`：卡点等级，取值建议为 `high`、`medium`、`low`、`unknown`。
- `source_reference`：数据来源，必须保留。

### technology_terms.csv

核心字段：

- `term_id`：稳定术语 ID。
- `term_name`：可用于匹配的原始术语。
- `belongs_to_tech_node_id`：回挂节点 ID，必须存在于 `technology_nodes.csv`。
- `mapping_confidence`：术语回挂置信度，取值 0 到 1。
- `semantic_tag`：术语类型，如 `model`、`method`、`component`、`task`、`capability`。
- `source_reference`：术语来源。

### technology_assets.csv

核心字段：

- `asset_id`：稳定资产 ID。
- `asset_name`：资产名称或证据标题。
- `asset_category`：资产类别，如 `patent`、`research_paper`、`model_artifact`、`software_framework`、`product_platform`。
- `belongs_to_tech_node_id`：回挂节点 ID。
- `source_reference`：资产来源。

## 使用约束

- `technology_nodes.csv` 是 v2.6 技术链映射的主表。
- `technology_terms.csv` 用于 exact、alias、term_belongs_to 三类匹配。
- `technology_assets.csv` 在 v2.6 中只作为证据补充，不直接影响关键核心结论。
- `broader_match` 只能说明候选属于某个上位技术方向，不能直接作为关键核心技术高置信结论。
- LLM 可以辅助扩写描述，但不能替代人工维护 `bottleneck_level` 和 `strategic_importance_level`。

## 更新流程

1. 从新的 `signals.csv`、`final_shortlist.csv` 和人工复核结果中收集 `no_match` 候选。
2. 判断候选是否应新建技术节点，还是补充到现有节点的术语别名。
3. 若新建节点，补充 `technology_nodes.csv`。
4. 若只是术语变体，补充 `technology_terms.csv`。
5. 若有专利、论文、标准、模型、产品平台等证据，补充 `technology_assets.csv`。
6. 每次更新都必须写明 `source_reference`、`maintainer` 和 `updated_at`。

## v2.6 首版覆盖范围

首版覆盖 42 个技术链节点，重点覆盖：

- 具身智能和人形机器人主干。
- 世界模型、生成式世界模型、视频世界模型、V-JEPA、GAIA-1 等模型节点。
- 强化学习、机器人训练、具身仿真、机器人规划和控制。
- 机器人感知、多模态感知、SLAM、三维表示。
- 大语言模型、机器人操作系统、Python 等基础支撑。
- 物理 AI 数据飞轮、机器人 AI 计算平台和多模态具身交互等从当前结果中沉淀出的早期节点。
