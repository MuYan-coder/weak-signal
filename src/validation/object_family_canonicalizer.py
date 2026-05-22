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
from typing import Dict, List, Optional, Set, Tuple
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


class ObjectFamilyCanonicalizer:
    """对象族归一化器"""

    def __init__(self, config_path: Optional[Path] = None, registry_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        self.config = self._load_config()
        self.registry = self._load_registry()
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


def get_canonicalizer(config_path: Optional[Path] = None, registry_path: Optional[Path] = None) -> ObjectFamilyCanonicalizer:
    """获取全局归一化器实例"""
    global _canonicalizer
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
