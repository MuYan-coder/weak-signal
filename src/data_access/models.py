from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class SourceQuery:
    tech_field_id: str
    tech_field_name: str
    keywords: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    source_types: List[str] = field(default_factory=list)
    raw_source_types: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    use_topic_index: bool = True
    sort: str = "relevance"
    include_full_text: bool = True

@dataclass
class DataSourceStats:
    source_type: str
    available_count: int
    loaded_count: int

@dataclass
class DocumentRecord:
    id: str
    source_id: str
    source_type: str
    raw_source_type: str
    title: str
    text: str
    date: str
    publish_time: Optional[str] = None
    org: Optional[str] = None
    authors: Optional[str] = None
    keywords: Optional[str] = None
    url: Optional[str] = None
    source_name: Optional[str] = None
    classification: Optional[str] = None
    industry: Optional[str] = None
    relevance_score: Optional[float] = 0.0
    data_backend: Optional[str] = None
