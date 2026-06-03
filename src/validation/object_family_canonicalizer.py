"""
对象族归一化模块

用于处理细粒度跨源技术术语的归一化，支持：
1. 英文变体归一（单复数、词形变体）
2. 中英文对齐
3. 上下位关系识别
4. 跨源模式分类
5. 分层归一（surface_term, canonical_term, parent_term）
6. 对象族匹配
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# 默认配置路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "object_family_canonicalization.yaml"
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "config" / "object_family_registry.yaml"


@dataclass
class CanonicalizationResult:
    """分层归一化结果"""
    surface_term: str           # 原始短语中最自然的术语
    canonical_term: str         # 对象族内部统一术语
    parent_term: str            # 上位概念
    term_type: str              # model_name | task | capability | component | scenario | method
    family_id: str              # 命中的对象族 ID
    normalization_type: str     # english_variant | chinese_english | hierarchy | none
    cross_source_pattern: str   # 跨源模式
    patent_role: str            # 专利侧角色
    priority: str               # 优先级


def _extract_domain_pack(value: Optional[Any]) -> Optional[Any]:
    if value is None:
        return None
    if hasattr(value, "domain_pack"):
        return getattr(value, "domain_pack")
    if hasattr(value, "canonicalization"):
        return value
    return None


def _object_family_enabled(pack: Optional[Any]) -> bool:
    if pack is None:
        return True
    canonicalization = getattr(pack, "canonicalization", {}) or {}
    if not isinstance(canonicalization, dict):
        return False
    return bool(canonicalization.get("object_family_enabled", False))


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _clean_terms(values: Any) -> List[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = []
    cleaned = []
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _family_priority(family: Dict[str, Any]) -> str:
    raw_priority = str(family.get("priority", "")).strip().lower()
    if raw_priority in {"high", "medium", "low"}:
        return raw_priority
    try:
        confidence = float(family.get("merge_confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _domain_pack_config(pack: Any) -> Dict[str, Any]:
    canonicalization = getattr(pack, "canonicalization", {}) or {}
    hierarchy = {}
    for relation in canonicalization.get("parent_child_terms", []) or []:
        if not isinstance(relation, dict):
            continue
        parent = str(relation.get("parent", "")).strip()
        children = _clean_terms(relation.get("children", []))
        if parent and children:
            hierarchy[parent] = {"children": children}
    return {
        "english_variants": {},
        "chinese_english_alignment": {},
        "hierarchy": hierarchy,
        "high_value_families": [],
        "cross_source_patterns": {},
    }


def _domain_pack_registry(pack: Any) -> Dict[str, List[Dict[str, Any]]]:
    canonicalization = getattr(pack, "canonicalization", {}) or {}
    families_by_canonical: Dict[str, Dict[str, Any]] = {}
    parent_by_child = {}
    for relation in canonicalization.get("parent_child_terms", []) or []:
        if not isinstance(relation, dict):
            continue
        parent = str(relation.get("parent", "")).strip()
        for child in _clean_terms(relation.get("children", [])):
            if parent:
                parent_by_child[child.lower()] = parent

    for family in canonicalization.get("object_families", []) or []:
        if not isinstance(family, dict):
            continue
        canonical = str(family.get("canonical_term", "")).strip()
        if not canonical:
            continue
        aliases = _clean_terms(family.get("aliases", []))
        zh_aliases = [term for term in aliases if _contains_cjk(term)]
        en_aliases = [term for term in aliases if not _contains_cjk(term)]
        entry = {
            "family_id": str(family.get("family_id", "")).strip() or f"dp_{len(families_by_canonical) + 1}",
            "canonical_term": canonical,
            "alias_terms": aliases,
            "zh_en_pairs": {
                "zh": zh_aliases,
                "en": en_aliases,
            },
            "parent_term": str(family.get("parent_term", "")).strip() or parent_by_child.get(canonical.lower(), ""),
            "term_type": str(family.get("term_type", "unknown")).strip() or "unknown",
            "cross_source_pattern": str(family.get("cross_source_pattern", "single_source")).strip() or "single_source",
            "patent_role": str(family.get("patent_role", "optional")).strip() or "optional",
            "priority": _family_priority(family),
            "merge_confidence": family.get("merge_confidence", 0.0),
            "evidence_required": bool(family.get("evidence_required", True)),
        }
        families_by_canonical[canonical.lower()] = entry

    for group in canonicalization.get("alias_groups", []) or []:
        if not isinstance(group, dict):
            continue
        canonical = str(group.get("canonical", "")).strip()
        if not canonical:
            continue
        aliases = _clean_terms(group.get("aliases", []))
        key = canonical.lower()
        entry = families_by_canonical.get(key)
        if entry is None:
            zh_aliases = [term for term in aliases if _contains_cjk(term)]
            en_aliases = [term for term in aliases if not _contains_cjk(term)]
            entry = {
                "family_id": f"alias_{len(families_by_canonical) + 1}",
                "canonical_term": canonical,
                "alias_terms": aliases,
                "zh_en_pairs": {"zh": zh_aliases, "en": en_aliases},
                "parent_term": parent_by_child.get(canonical.lower(), ""),
                "term_type": "unknown",
                "cross_source_pattern": "single_source",
                "patent_role": "optional",
                "priority": "medium",
                "do_not_merge_with": _clean_terms(group.get("do_not_merge_with", [])),
            }
            families_by_canonical[key] = entry
        else:
            merged_aliases = _clean_terms([*entry.get("alias_terms", []), *aliases])
            entry["alias_terms"] = merged_aliases
            entry["zh_en_pairs"] = {
                "zh": [term for term in merged_aliases if _contains_cjk(term)],
                "en": [term for term in merged_aliases if not _contains_cjk(term)],
            }
            entry["do_not_merge_with"] = _clean_terms(group.get("do_not_merge_with", []))

    return {"families": list(families_by_canonical.values())}


class ObjectFamilyCanonicalizer:
    """对象族归一化器"""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        domain_pack: Optional[Any] = None,
    ):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        self.domain_pack = _extract_domain_pack(domain_pack)
        self.object_family_enabled = _object_family_enabled(self.domain_pack)
        if self.domain_pack is None:
            self.config = self._load_config()
            self.registry = self._load_registry()
            self.object_family_enabled = True
        elif self.object_family_enabled:
            self.config = _domain_pack_config(self.domain_pack)
            self.registry = _domain_pack_registry(self.domain_pack)
        else:
            self.config = {}
            self.registry = {"families": []}
        self._build_lookup_tables()

    def _load_config(self) -> dict:
        """加载配置文件"""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _load_registry(self) -> dict:
        """加载对象族注册表"""
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _build_lookup_tables(self):
        """构建查找表"""
        # 英文变体 -> canonical
        self.variant_to_canonical: Dict[str, str] = {}
        for canonical, info in self.config.get('english_variants', {}).items():
            self.variant_to_canonical[canonical.lower()] = canonical
            for variant in info.get('variants', []):
                self.variant_to_canonical[variant.lower()] = canonical

        # 中文 -> canonical (英文)
        self.chinese_to_canonical: Dict[str, str] = {}
        for canonical, info in self.config.get('chinese_english_alignment', {}).items():
            for chinese in info.get('chinese_terms', []):
                self.chinese_to_canonical[chinese.lower()] = canonical

        # 子术语 -> 父术语
        self.child_to_parent: Dict[str, str] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)
        for parent, info in self.config.get('hierarchy', {}).items():
            for child in info.get('children', []):
                self.child_to_parent[child.lower()] = parent
                self.parent_to_children[parent.lower()].append(child)

        # 高价值对象族（旧格式兼容）
        self.high_value_families: Dict[str, dict] = {}
        for family in self.config.get('high_value_families', []):
            term = family.get('term', '').lower()
            self.high_value_families[term] = family

        # 对象族注册表（新格式）
        self.families_by_id: Dict[str, dict] = {}
        self.families_by_term: Dict[str, dict] = {}
        for family in self.registry.get('families', []):
            family_id = family.get('family_id', '')
            canonical_term = family.get('canonical_term', '').lower()

            if family_id:
                self.families_by_id[family_id] = family

            # 按 canonical_term 索引
            if canonical_term:
                self.families_by_term[canonical_term] = family

            # 按 alias_terms 索引
            for alias in family.get('alias_terms', []):
                self.families_by_term[alias.lower()] = family

            # 按 zh_en_pairs 索引
            for zh in family.get('zh_en_pairs', {}).get('zh', []):
                self.families_by_term[zh.lower()] = family
            for en in family.get('zh_en_pairs', {}).get('en', []):
                self.families_by_term[en.lower()] = family

    def canonicalize(self, term: str) -> Tuple[str, str, Optional[str]]:
        """
        归一化术语

        Returns:
            (canonical_term, normalization_type, parent_term)
            - canonical_term: 归一化后的术语
            - normalization_type: 归一化类型 (english_variant, chinese_english, hierarchy, none)
            - parent_term: 父术语（如果存在）
        """
        term_lower = term.lower().strip()

        # 1. 检查英文变体
        if term_lower in self.variant_to_canonical:
            canonical = self.variant_to_canonical[term_lower]
            parent = self.child_to_parent.get(canonical.lower())
            return canonical, 'english_variant', parent

        # 2. 检查中英文对齐
        if term_lower in self.chinese_to_canonical:
            canonical = self.chinese_to_canonical[term_lower]
            parent = self.child_to_parent.get(canonical.lower())
            return canonical, 'chinese_english', parent

        # 3. 检查层级关系
        if term_lower in self.child_to_parent:
            parent = self.child_to_parent[term_lower]
            return term, 'hierarchy', parent

        # 4. 无归一化
        return term, 'none', None

    def canonicalize_with_layers(self, term: str) -> CanonicalizationResult:
        """
        分层归一化术语

        Returns:
            CanonicalizationResult 包含所有分层信息
        """
        term_lower = term.lower().strip()

        # 默认值
        result = CanonicalizationResult(
            surface_term=term,
            canonical_term=term,
            parent_term="",
            term_type="unknown",
            family_id="",
            normalization_type="none",
            cross_source_pattern="single_source",
            patent_role="optional",
            priority="low"
        )

        # 1. 尝试从对象族注册表匹配
        family = self.match_family(term)
        if family:
            result.family_id = family.get('family_id', '')
            result.canonical_term = family.get('canonical_term', term)
            result.parent_term = family.get('parent_term', '') or ""
            result.term_type = family.get('term_type', 'unknown')
            result.cross_source_pattern = family.get('cross_source_pattern', 'single_source')
            result.patent_role = family.get('patent_role', 'optional')
            result.priority = family.get('priority', 'low')

            # 确定归一化类型
            if term_lower != result.canonical_term.lower():
                if any(zh in term_lower for zh in family.get('zh_en_pairs', {}).get('zh', [])):
                    result.normalization_type = 'chinese_english'
                else:
                    result.normalization_type = 'english_variant'
            else:
                result.normalization_type = 'none'

            return result

        # 2. 使用旧的归一化逻辑
        canonical, norm_type, parent = self.canonicalize(term)
        result.canonical_term = canonical
        result.normalization_type = norm_type
        result.parent_term = parent or ""

        # 检测术语类型
        result.term_type = self.detect_term_type(term)

        return result

    def detect_term_type(self, term: str) -> str:
        """
        检测术语类型

        Returns:
            term_type: model_name | task | capability | component | scenario | method | unknown
        """
        term_lower = term.lower().strip()

        # 模型名称关键词
        model_keywords = ['model', 'net', 'gpt', 'bert', 'transformer', 'jepa', 'dreamer', 'gaia', 'v-jepa']
        if any(kw in term_lower for kw in model_keywords):
            return 'model_name'

        # 任务关键词
        task_keywords = ['planning', 'navigation', 'manipulation', 'grasping', 'control', 'training', 'learning']
        if any(kw in term_lower for kw in task_keywords):
            return 'task'

        # 能力关键词
        capability_keywords = ['perception', 'reasoning', 'understanding', 'intelligence', 'vision']
        if any(kw in term_lower for kw in capability_keywords):
            return 'capability'

        # 组件关键词
        component_keywords = ['robot', 'arm', 'sensor', 'camera', 'lidar', 'imu', 'hardware', 'system']
        if any(kw in term_lower for kw in component_keywords):
            return 'component'

        # 场景关键词
        scenario_keywords = ['indoor', 'outdoor', 'driving', 'navigation', 'manipulation', 'scene']
        if any(kw in term_lower for kw in scenario_keywords):
            return 'scenario'

        # 方法关键词
        method_keywords = ['learning', 'reinforcement', 'simulation', 'training', 'algorithm']
        if any(kw in term_lower for kw in method_keywords):
            return 'method'

        return 'unknown'

    def match_family(self, phrase: str) -> Optional[dict]:
        """
        从短语匹配对象族

        Args:
            phrase: 待匹配的短语

        Returns:
            匹配的对象族信息，如果无匹配则返回 None
        """
        import re
        if not phrase:
            return None

        phrase_lower = phrase.lower().strip()

        # 1. 精确匹配 canonical_term
        if phrase_lower in self.families_by_term:
            return self.families_by_term[phrase_lower]

        # 2. 检查是否包含对象族的 canonical_term 或 alias（使用词边界匹配）
        for family in self.registry.get('families', []):
            canonical = family.get('canonical_term', '').lower()
            if canonical and len(canonical) >= 3:
                # 使用词边界匹配，避免 "nn" 匹配到 "planning"
                pattern = r'\b' + re.escape(canonical) + r'\b'
                if re.search(pattern, phrase_lower):
                    return family

            for alias in family.get('alias_terms', []):
                alias_lower = alias.lower()
                # 对于短别名（< 3 字符），要求精确匹配
                if len(alias_lower) < 3:
                    if alias_lower == phrase_lower:
                        return family
                else:
                    pattern = r'\b' + re.escape(alias_lower) + r'\b'
                    if re.search(pattern, phrase_lower):
                        return family

            for zh in family.get('zh_en_pairs', {}).get('zh', []):
                # 中文不需要词边界
                if zh.lower() in phrase_lower:
                    return family

            for en in family.get('zh_en_pairs', {}).get('en', []):
                en_lower = en.lower()
                if len(en_lower) >= 3:
                    pattern = r'\b' + re.escape(en_lower) + r'\b'
                    if re.search(pattern, phrase_lower):
                        return family

        # 3. 使用旧的归一化逻辑检查
        canonical, norm_type, _ = self.canonicalize(phrase)
        if norm_type != 'none':
            canonical_lower = canonical.lower()
            if canonical_lower in self.families_by_term:
                return self.families_by_term[canonical_lower]

        return None

    def match_candidate_family(self, concept: Dict[str, Any]) -> Optional[dict]:
        """Match a candidate row against the active object-family registry."""
        if not self.object_family_enabled:
            return None
        raw_phrase = str((concept or {}).get("raw_phrase", "")).strip()
        display_name = str((concept or {}).get("display_candidate_name", "")).strip()
        canonical_name_en = str((concept or {}).get("canonical_candidate_name_en", "")).strip()
        mechanism_core = str((concept or {}).get("mechanism_core", "")).strip().lower()

        family = None
        for term in [canonical_name_en, raw_phrase, display_name]:
            if term:
                family = self.match_family(term)
                if family:
                    break
        if not family:
            return None

        if self.domain_pack is None and family.get("family_id") == "of_006":
            legacy_split = {
                "simulation": {
                    "family_id": "of_027",
                    "canonical_term": "embodied simulation",
                    "term_type": "method",
                    "cross_source_pattern": "news_paper",
                    "patent_role": "optional",
                    "priority": "high",
                },
                "training": {
                    "family_id": "of_028",
                    "canonical_term": "robot training",
                    "term_type": "method",
                    "cross_source_pattern": "news_paper",
                    "patent_role": "optional",
                    "priority": "medium",
                },
                "planning": {
                    "family_id": "of_029",
                    "canonical_term": "robot planning",
                    "term_type": "task",
                    "cross_source_pattern": "news_paper",
                    "patent_role": "optional",
                    "priority": "medium",
                },
            }
            return legacy_split.get(mechanism_core, family)
        return family

    def extract_canonical_from_phrase(self, phrase: str) -> Tuple[str, str, Optional[str]]:
        """
        从短语中提取并归一化配置中的术语。

        检查短语是否包含配置中的术语，如果包含则返回归一化后的术语。

        Args:
            phrase: 待检查的短语

        Returns:
            (canonical_term, normalization_type, parent_term)
        """
        phrase_lower = phrase.lower().strip()

        # 1. 检查英文变体是否在短语中
        for variant, canonical in self.variant_to_canonical.items():
            if variant in phrase_lower:
                parent = self.child_to_parent.get(canonical.lower())
                return canonical, 'english_variant', parent

        # 2. 检查中英文对齐是否在短语中
        for chinese, canonical in self.chinese_to_canonical.items():
            if chinese in phrase_lower:
                parent = self.child_to_parent.get(canonical.lower())
                return canonical, 'chinese_english', parent

        # 3. 检查层级关系是否在短语中
        for child, parent in self.child_to_parent.items():
            if child in phrase_lower:
                return child, 'hierarchy', parent

        # 4. 无匹配
        return phrase, 'none', None

    def get_family_by_id(self, family_id: str) -> Optional[dict]:
        """按 ID 获取对象族"""
        return self.families_by_id.get(family_id)

    def get_family_by_term(self, term: str) -> Optional[dict]:
        """按术语获取对象族"""
        term_lower = term.lower().strip()
        return self.families_by_term.get(term_lower)

    def get_all_families(self) -> List[dict]:
        """获取所有对象族"""
        return list(self.families_by_id.values())

    def get_cross_source_pattern(self, source_types: List[str]) -> Tuple[str, dict]:
        """
        获取跨源模式

        Args:
            source_types: 来源类型列表，如 ['news', 'paper']

        Returns:
            (pattern_name, pattern_info)
        """
        source_set = set(s.lower() for s in source_types)

        patterns = self.config.get('cross_source_patterns', {})

        # 检查全源
        if source_set == {'news', 'paper', 'patent'}:
            return 'full_cross_source', patterns.get('full_cross_source', {})

        # 检查双源
        if source_set == {'news', 'paper'}:
            return 'news_paper', patterns.get('news_paper', {})
        if source_set == {'paper', 'patent'}:
            return 'paper_patent', patterns.get('paper_patent', {})
        if source_set == {'news', 'patent'}:
            return 'news_patent', patterns.get('news_patent', {})

        # 单源或其他
        return 'single_source', {}

    def get_patent_role(self, term: str, cross_source_pattern: str) -> str:
        """
        获取专利侧角色

        Args:
            term: 技术术语
            cross_source_pattern: 跨源模式

        Returns:
            patent_role: 专利侧角色
        """
        term_lower = term.lower().strip()

        # 1. 检查对象族注册表
        family = self.get_family_by_term(term)
        if family and 'patent_role' in family:
            return family['patent_role']

        # 2. 检查高价值对象族配置
        if term_lower in self.high_value_families:
            family = self.high_value_families[term_lower]
            if 'patent_role' in family:
                return family['patent_role']

        # 3. 检查跨源模式默认角色
        patterns = self.config.get('cross_source_patterns', {})
        pattern_info = patterns.get(cross_source_pattern, {})
        return pattern_info.get('patent_role', 'optional')

    def is_high_value_family(self, term: str) -> bool:
        """判断是否为高价值对象族"""
        canonical, _, _ = self.canonicalize(term)
        return canonical.lower() in self.families_by_term or term.lower() in self.families_by_term

    def get_family_priority(self, term: str) -> str:
        """获取对象族优先级"""
        family = self.get_family_by_term(term)
        if family:
            return family.get('priority', 'medium')
        return 'low'

    def get_family_info(self, term: str) -> Optional[dict]:
        """获取对象族完整信息"""
        return self.get_family_by_term(term)

    def get_all_canonical_terms(self) -> Set[str]:
        """获取所有归一化后的术语"""
        terms = set()
        # 英文 canonical
        for canonical in self.config.get('english_variants', {}).keys():
            terms.add(canonical)
        # 中英文对齐 canonical
        for canonical in self.config.get('chinese_english_alignment', {}).keys():
            terms.add(canonical)
        # 对象族注册表
        for family in self.registry.get('families', []):
            if family.get('canonical_term'):
                terms.add(family['canonical_term'])
        return terms

    def get_families_by_pattern(self, pattern: str) -> List[dict]:
        """按跨源模式获取对象族列表"""
        families = []
        for family in self.registry.get('families', []):
            if family.get('cross_source_pattern') == pattern:
                families.append(family)
        return families


# 全局实例
_canonicalizer: Optional[ObjectFamilyCanonicalizer] = None


def get_canonicalizer(
    config_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    domain_pack: Optional[Any] = None,
    domain_context: Optional[Any] = None,
) -> ObjectFamilyCanonicalizer:
    """获取全局归一化器实例"""
    global _canonicalizer
    active_pack = _extract_domain_pack(domain_context) or _extract_domain_pack(domain_pack)
    if active_pack is not None:
        return ObjectFamilyCanonicalizer(config_path, registry_path, domain_pack=active_pack)
    if _canonicalizer is None or config_path is not None or registry_path is not None:
        _canonicalizer = ObjectFamilyCanonicalizer(config_path, registry_path)
    return _canonicalizer


def canonicalize_term(term: str) -> Tuple[str, str, Optional[str]]:
    """归一化术语（便捷函数）"""
    return get_canonicalizer().canonicalize(term)


def canonicalize_with_layers(term: str) -> CanonicalizationResult:
    """分层归一化术语（便捷函数）"""
    return get_canonicalizer().canonicalize_with_layers(term)


def get_cross_source_pattern(source_types: List[str]) -> Tuple[str, dict]:
    """获取跨源模式（便捷函数）"""
    return get_canonicalizer().get_cross_source_pattern(source_types)


def get_patent_role(term: str, cross_source_pattern: str) -> str:
    """获取专利侧角色（便捷函数）"""
    return get_canonicalizer().get_patent_role(term, cross_source_pattern)


def match_family(phrase: str) -> Optional[dict]:
    """从短语匹配对象族（便捷函数）"""
    return get_canonicalizer().match_family(phrase)
