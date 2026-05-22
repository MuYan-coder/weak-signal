# 产业技术预见智能体

面向产业技术预见场景的弱信号识别系统。项目从专利、文献、研报、资讯等多源文本中抽取事件，形成候选技术对象，完成事件质量评分、弱信号评分、主题细化、反向验证、技术链映射、时间验证、关键核心潜力评分和报告生成。

当前主线是自研 Pipeline 架构，不再保留早期 LangChain/RAG 原型流程。

## 当前版本

- 当前发布版本：`v2.7.0`
- 目标发布标签：`v2.7`
- v2.7 主线新增时间验证、关键核心潜力评分、关键核心候选短名单、报告关键核心候选章节和 Web 展示增强。
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
    ├── extraction/
    │   ├── event_extractor.py
    │   ├── candidate_former.py
    │   └── tech_lexicon.py
    ├── scoring/
    │   ├── scorer.py
    │   ├── signal_generator.py
    │   ├── topic_refiner.py
    │   └── key_core_scorer.py
    ├── validation/
    │   ├── reverse_validator.py
    │   ├── event_quality.py
    │   ├── tech_chain_mapper.py
    │   ├── temporal_validator.py
    │   ├── object_family_canonicalizer.py
    │   ├── family_evaluator.py
    │   ├── final_shortlist.py
    │   └── baseline_compare.py
    ├── utils/
    │   ├── api_stats.py
    │   ├── config.py
    │   ├── env_config.py
    │   ├── env_utils.py
    │   ├── llm_client.py
    │   └── semantic_utils.py
    ├── config/
    │   ├── object_family_canonicalization.yaml
    │   └── object_family_registry.yaml
    └── data/
        └── api_stats.json
```

## 主流程

1. 数据加载与标准化
2. 事件抽取
3. 事件质量评分
4. 候选技术对象成形与候选证据质量聚合
5. 弱信号评分
6. 主题细化
7. 反向验证
8. 技术链映射
9. 时间验证
10. 关键核心潜力评分
11. 信号生成
12. 报告生成

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
```

Web 页面运行：

```bash
streamlit run web_app.py
```

## 环境变量

在 `.env` 中配置模型访问参数：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

EXTRACTION_MODEL=Qwen/Qwen2.5-14B-Instruct
AGENT_MODEL=Qwen/Qwen2.5-32B-Instruct
REPORT_MODEL=deepseek-ai/DeepSeek-V3
```

## 目录边界

- `src/core/` 只负责流程编排和智能体封装。
- `src/extraction/` 负责从原始文本到候选对象的结构化加工。
- `src/scoring/` 负责评分、主题细化和最终信号表达。
- `src/validation/` 负责反向验证、对象族归一化和最终名单整理。
- `src/utils/` 只放跨模块基础设施，例如配置、LLM 客户端、语义相似度和调用统计。
- `data/`、`memory/`、`result/` 是运行时数据目录，不放核心业务代码。

## 已清理内容

本次结构整理移除了早期 LangChain/RAG 原型文件、根目录重复模块、旧演示副本和 Python 字节码缓存。项目现在只保留一条主线：`main.py` / `web_app.py` 调用 `src/core`，再由 Pipeline 串联 extraction、scoring、validation 与 utils。
