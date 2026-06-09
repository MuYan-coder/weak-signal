import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from .models import DocumentRecord

# 来源类型映射契约
SOURCE_TYPE_MAPPING = {
    "literature": "paper",
    "consulting": "news",
    "policy": "policy",  # 可由配置决定是否映射为 news
    "report": "report",
    "patent": "patent",
    "topic": "topic"
}

# 逆向来源类型映射
REVERSE_SOURCE_TYPE_MAPPING = {
    "paper": "literature",
    "news": "consulting",
    "policy": "policy",
    "report": "report",
    "patent": "patent"
}

def parse_name_list(val: Any) -> str:
    """安全解析包含列表、字典或JSON格式的姓名/机构名，返回逗号分隔的字符串"""
    if val is None:
        return ""
    if isinstance(val, list):
        names = []
        for item in val:
            if isinstance(item, dict):
                names.append(str(item.get("name", "")).strip())
            else:
                names.append(str(item).strip())
        return ", ".join(n for n in names if n)

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "nat", "null", "none"):
        return ""

    # 如果是 JSON 数组格式
    if val_str.startswith("[") and val_str.endswith("]"):
        import json
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return parse_name_list(parsed)
        except Exception:
            pass

    # 正则提取字典结构中的 name
    names = re.findall(r"'name':\s*'([^']*)'", val_str)
    if not names:
        names = re.findall(r'"name":\s*"([^"]*)"', val_str)

    if names:
        return ", ".join(n.strip() for n in names if n.strip())

    # 处理类似 ['name1', 'name2']
    items = re.findall(r"'([^']*)'", val_str)
    if not items:
        items = re.findall(r'"([^"]*)"', val_str)
    if items:
        return ", ".join(i.strip() for i in items if i.strip())

    return val_str

def clean_date_str(date_val: Any) -> str:
    """标准化日期为 YYYY-MM-DD 或 YYYY-MM-DD HH:mm:ss"""
    if date_val is None:
        return ""

    # 如果已经是 datetime
    if isinstance(date_val, datetime):
        return date_val.strftime("%Y-%m-%d %H:%M:%S")

    date_str = str(date_val).strip()
    if not date_str or date_str.lower() in ("nan", "nat", "null", "none"):
        return ""

    # 处理四位纯年份 (如 2021)
    if re.match(r"^\d{4}$", date_str):
        return f"{date_str}-01-01"

    # 处理类似 ISO 8601 或带 T 的格式
    if "T" in date_str:
        try:
            # 尝试解析 2023-05-15T00:00:00
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # 正则提取 YYYY-MM-DD 和可选的时间
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2}):(\d{1,2}))?", date_str)
    if match:
        year, month, day, hour, minute, second = match.groups()
        if hour and minute and second:
            return f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return date_str

def get_first_valid(data: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    """按优先级顺序获取第一个有效的字段值"""
    for key in keys:
        val = data.get(key)
        if val is not None and str(val).strip().lower() not in ("", "nan", "nat", "null", "none"):
            return val
    return default

def normalize_mysql_literature(row: Dict[str, Any]) -> DocumentRecord:
    """映射并标准化 MySQL 中的文献 (Literature)"""
    source_id = str(get_first_valid(row, ["doi/arxiv_id", "doi", "id", "literature_id"]))
    if not source_id:
        source_id = str(hash(row.get("title", "")) & 0xffffffffffff)

    title = str(get_first_valid(row, ["title", "paper_title"], ""))
    abstract = str(get_first_valid(row, ["abstract"], ""))
    content = str(get_first_valid(row, ["content", "features", "full_text", "introduction", "conclusion"], ""))

    # 拼接 text
    text = f"{title}\n{abstract}\n{content}".strip()

    # 日期优先级: event_time -> publish_date -> online_date -> year -> created_at
    raw_date = get_first_valid(row, ["event_time", "publish_date", "online_date", "year", "create_time", "created_at"], "")
    date_str = clean_date_str(raw_date)

    org = str(get_first_valid(row, ["author_org", "affiliations", "organization", "institution"], ""))
    authors = parse_name_list(get_first_valid(row, ["authors", "author"], ""))
    keywords = str(get_first_valid(row, ["keyword", "keywords", "label"], ""))
    url = str(get_first_valid(row, ["url", "doi/arxiv_id", "doi"], ""))
    source_name = str(get_first_valid(row, ["collect_source_name", "venue", "journal", "source_name"], ""))
    classification = str(get_first_valid(row, ["classification", "label"], ""))

    return DocumentRecord(
        id=f"paper:{source_id}",
        source_id=source_id,
        source_type="paper",
        raw_source_type="literature",
        title=title,
        text=text,
        date=date_str,
        publish_time=str(raw_date),
        org=org,
        authors=authors,
        keywords=keywords,
        url=url,
        source_name=source_name,
        classification=classification,
        industry=str(row.get("industry", "")),
        relevance_score=float(row.get("relevance", row.get("relevance_score", 0.0))),
        data_backend="mysql"
    )

def normalize_es_consulting(hit: Dict[str, Any], policy_as_news: bool = True) -> DocumentRecord:
    """映射并标准化 ES 中的咨询 (Consulting)"""
    source_id = str(hit.get("id", hit.get("_id", "")))

    title = str(get_first_valid(hit, ["title", "title_cn", "title_zh", "title_en", "name", "标题"], ""))
    abstract = str(get_first_valid(hit, ["abstract", "abstract_cn", "abstract_zh", "abstract_en", "summary", "description"], ""))
    content = str(get_first_valid(hit, ["content", "main_content", "body", "html", "content_html"], ""))
    viewpoints = str(hit.get("viewpoints", ""))

    # 拼接 text
    text_parts = [title, abstract]
    if viewpoints:
        text_parts.append(f"观点：{viewpoints}")
    if content:
        text_parts.append(content)
    text = "\n".join(part for part in text_parts if part).strip()

    # 日期优先级: publish_date -> publish_time -> crawl_time -> created_at -> update_time
    raw_date = get_first_valid(hit, ["publish_date", "publish_time", "crawl_time", "created_at", "create_time", "update_time"], "")
    date_str = clean_date_str(raw_date)

    org = str(get_first_valid(hit, ["source", "org", "organization", "publisher"], ""))
    authors = parse_name_list(get_first_valid(hit, ["author", "authors"], ""))
    keywords = str(get_first_valid(hit, ["tags", "entities", "keywords"], ""))
    url = str(get_first_valid(hit, ["url", "link", "source_url", "url_source", "pdf_link"], ""))
    source_name = str(get_first_valid(hit, ["source", "source_name", "publisher"], ""))
    classification = str(get_first_valid(hit, ["node_classify", "domain", "channel", "classification"], ""))
    industry = str(get_first_valid(hit, ["lz_industry", "industry"], ""))

    return DocumentRecord(
        id=f"news:{source_id}",
        source_id=source_id,
        source_type="news",
        raw_source_type="consulting",
        title=title,
        text=text,
        date=date_str,
        publish_time=str(raw_date),
        org=org,
        authors=authors,
        keywords=keywords,
        url=url,
        source_name=source_name,
        classification=classification,
        industry=industry,
        relevance_score=float(hit.get("relevance_score", hit.get("_score", 0.0) or 0.0)),
        data_backend="es"
    )

def normalize_es_policy(hit: Dict[str, Any], policy_as_news: bool = False) -> DocumentRecord:
    """映射并标准化 ES 中的政策 (Policy)"""
    source_id = str(hit.get("id", hit.get("_id", "")))

    title = str(get_first_valid(hit, ["title", "title_cn", "title_zh", "title_en", "name", "标题"], ""))
    abstract = str(get_first_valid(hit, ["abstract", "abstract_cn", "abstract_zh", "abstract_en", "summary", "policy_summary", "description"], ""))
    content = str(get_first_valid(hit, ["content", "main_content", "body", "html", "content_html"], ""))

    # 拼接 text
    text = f"{title}\n{abstract}\n{content}".strip()

    # 日期优先级: publish_date -> publish_time -> release_date -> created_at
    raw_date = get_first_valid(hit, ["publish_date", "publish_time", "release_date", "created_at", "create_time"], "")
    date_str = clean_date_str(raw_date)

    org = str(get_first_valid(hit, ["publisher", "org", "organization"], ""))
    authors = parse_name_list(get_first_valid(hit, ["authors", "author"], ""))
    keywords = str(get_first_valid(hit, ["keywords", "tags"], ""))
    url = str(get_first_valid(hit, ["url", "link", "source_url", "url_source", "pdf_link"], ""))
    source_name = str(get_first_valid(hit, ["publisher", "org", "source_name"], ""))
    classification = str(get_first_valid(hit, ["classification", "category"], ""))

    source_type = "news" if policy_as_news else "policy"

    return DocumentRecord(
        id=f"{source_type}:{source_id}",
        source_id=source_id,
        source_type=source_type,
        raw_source_type="policy",
        title=title,
        text=text,
        date=date_str,
        publish_time=str(raw_date),
        org=org,
        authors=authors,
        keywords=keywords,
        url=url,
        source_name=source_name,
        classification=classification,
        industry=str(hit.get("industry", "")),
        relevance_score=float(hit.get("relevance_score", hit.get("_score", 0.0) or 0.0)),
        data_backend="es"
    )

def normalize_es_report(hit: Dict[str, Any]) -> DocumentRecord:
    """映射并标准化 ES 中的研报 (Report)"""
    source_id = str(hit.get("id", hit.get("doc_id", hit.get("_id", ""))))

    title = str(get_first_valid(hit, ["title", "title_cn", "title_zh", "title_en", "name"], ""))
    abstract = str(get_first_valid(hit, ["abstract", "abstract_cn", "abstract_zh", "abstract_en", "summary", "viewpoints", "description"], ""))
    content = str(get_first_valid(hit, ["html", "content", "main_content", "body", "content_html"], ""))

    # 拼接 text
    text = f"{title}\n{abstract}\n{content}".strip()

    # 日期优先级: publish_date -> report_date -> publish_time -> created_at
    raw_date = get_first_valid(hit, ["publish_date", "report_date", "publish_time", "crawl_time", "created_at", "create_time"], "")
    date_str = clean_date_str(raw_date)

    org = str(get_first_valid(hit, ["institution", "source", "org"], ""))
    authors = parse_name_list(get_first_valid(hit, ["authors", "author"], ""))
    keywords = str(hit.get("keywords", ""))
    url = str(get_first_valid(hit, ["url", "url_source", "link", "source_url", "pdf_link"], ""))
    source_name = str(get_first_valid(hit, ["source", "institution"], ""))
    classification = str(get_first_valid(hit, ["type", "classification"], ""))
    industry = str(get_first_valid(hit, ["industry", "stock_name"], ""))

    return DocumentRecord(
        id=f"report:{source_id}",
        source_id=source_id,
        source_type="report",
        raw_source_type="report",
        title=title,
        text=text,
        date=date_str,
        publish_time=str(raw_date),
        org=org,
        authors=authors,
        keywords=keywords,
        url=url,
        source_name=source_name,
        classification=classification,
        industry=industry,
        relevance_score=float(hit.get("relevance_score", hit.get("_score", 0.0) or 0.0)),
        data_backend="es"
    )

def normalize_es_patent(hit: Dict[str, Any]) -> DocumentRecord:
    """映射并标准化 ES 中的专利 (Patent)"""
    # 专利可能没有自然主键，从日期与标题计算 hash
    title = str(get_first_valid(hit, ["title_cn", "title_zh", "title", "title_en", "name", "专利名称", "发明名称"], ""))
    raw_date = get_first_valid(hit, ["public_date", "apply_date", "priority_date", "created_at"], "")
    date_str = clean_date_str(raw_date)

    source_id = str(get_first_valid(hit, ["id", "_id", "doc_id", "公开号", "申请号", "专利号"], ""))
    if not source_id:
        import hashlib
        seed = f"{title}:{date_str}"
        source_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]

    abstract = str(get_first_valid(hit, ["abstract_cn", "abstract_zh", "abstract", "abstract_en", "summary", "description"], ""))
    claims = str(get_first_valid(hit, ["first_claim", "claims", "claim"], ""))

    # 拼接 text
    text_parts = [title]
    if abstract:
        text_parts.append(abstract)
    if claims:
        text_parts.append(claims)
    ipc = str(get_first_valid(hit, ["ipc"], ""))
    if ipc:
        text_parts.append(f"IPC分类: {ipc}")
    text = "\n".join(text_parts).strip()

    org_val = get_first_valid(hit, ["applicants_norm", "applicants", "applicant", "申请人", "org"], "")
    org = parse_name_list(org_val)

    authors_val = get_first_valid(hit, ["inventors", "inventor", "发明人", "authors"], "")
    authors = parse_name_list(authors_val)

    keywords = str(get_first_valid(hit, ["ipc", "cpc"], ""))
    url = str(get_first_valid(hit, ["pdf_url", "url", "link", "source_url", "url_source", "pdf_link"], ""))
    classification = str(get_first_valid(hit, ["ipc", "cpc", "classification"], ""))
    industry = str(get_first_valid(hit, ["方向", "industry"], ""))

    return DocumentRecord(
        id=f"patent:{source_id}",
        source_id=source_id,
        source_type="patent",
        raw_source_type="patent",
        title=title,
        text=text,
        date=date_str,
        publish_time=str(raw_date),
        org=org,
        authors=authors,
        keywords=keywords,
        url=url,
        source_name="专利局",
        classification=classification,
        industry=industry,
        relevance_score=float(hit.get("relevance_score", hit.get("_score", 0.0) or 0.0)),
        data_backend="es"
    )

def normalize_es_topic(hit: Dict[str, Any]) -> DocumentRecord:
    """映射并标准化 ES 中的话题 (Topic)"""
    source_id = str(hit.get("id", hit.get("_id", "")))
    title = str(get_first_valid(hit, ["topic_name", "name", "title"], ""))
    keywords = str(get_first_valid(hit, ["keywords", "tags", "synonyms"], ""))
    raw_date = get_first_valid(hit, ["updated_at", "created_at", "update_time"], "")
    date_str = clean_date_str(raw_date)

    return DocumentRecord(
        id=f"topic:{source_id}",
        source_id=source_id,
        source_type="topic",
        raw_source_type="topic",
        title=title,
        text=f"话题：{title}\n关键词：{keywords}",
        date=date_str,
        publish_time=str(raw_date),
        keywords=keywords,
        data_backend="es"
    )

def normalize_record(raw_data: Dict[str, Any], raw_source_type: str, policy_as_news: bool = True) -> DocumentRecord:
    """分发并归一化一条记录"""
    raw_source_type = raw_source_type.lower().strip()
    if raw_source_type in ("literature", "paper"):
        return normalize_mysql_literature(raw_data)
    elif raw_source_type == "consulting":
        return normalize_es_consulting(raw_data, policy_as_news=policy_as_news)
    elif raw_source_type == "policy":
        return normalize_es_policy(raw_data, policy_as_news=policy_as_news)
    elif raw_source_type == "report":
        return normalize_es_report(raw_data)
    elif raw_source_type == "patent":
        return normalize_es_patent(raw_data)
    elif raw_source_type == "topic":
        return normalize_es_topic(raw_data)
    else:
        # 默认回退
        source_id = str(raw_data.get("id", raw_data.get("_id", "unknown")))
        title = str(raw_data.get("title", ""))
        text = str(raw_data.get("text", title))
        date = clean_date_str(raw_data.get("date", ""))
        return DocumentRecord(
            id=f"unknown:{source_id}",
            source_id=source_id,
            source_type="unknown",
            raw_source_type=raw_source_type,
            title=title,
            text=text,
            date=date,
            data_backend="mock"
        )
