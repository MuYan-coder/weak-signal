from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Sequence, Tuple

from .tech_lexicon import normalize_signal_phrase


SurfacePattern = Tuple[str, Tuple[str, ...]]

LEGACY_HUMANOID_PACK_IDS = {
    "humanoid_robot",
    "humanoid_robot_preset",
    "legacy_humanoid_robot",
}


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _list_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    get_value = getattr(source, "get", None)
    if callable(get_value):
        return get_value(name, default)
    return getattr(source, name, default)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return dict(items())
        except (TypeError, ValueError):
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if mapped is not value:
            return _as_mapping(mapped)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    try:
        attrs = vars(value)
    except TypeError:
        return {}
    return {
        key: item
        for key, item in attrs.items()
        if isinstance(key, str) and not key.startswith("_") and not callable(item)
    }


def _mapping_value(source: Any, name: str) -> dict[str, Any]:
    return _as_mapping(_value(source, name, None))


def _has_pack_sections(source: Any) -> bool:
    return any(
        _mapping_value(source, key)
        for key in (
            "domain_identity",
            "observation_scopes",
            "candidate_formation",
            "reporting",
        )
    )


def _attached_pack(domain_lexicon: Any) -> Any:
    for source in (
        domain_lexicon,
        _value(domain_lexicon, "domain_context"),
        _value(domain_lexicon, "context"),
        _value(domain_lexicon, "domain_pack"),
        _value(domain_lexicon, "pack"),
    ):
        if source is None:
            continue
        pack = _value(source, "domain_pack")
        if pack is not None:
            return pack
        if isinstance(source, dict):
            pack = source.get("domain_pack") or source.get("pack")
            if pack is not None:
                return pack
            if _has_pack_sections(source):
                return source
        elif _has_pack_sections(source):
            return source
    return None


def _section(domain_lexicon: Any, name: str) -> dict[str, Any]:
    pack = _attached_pack(domain_lexicon)
    section = _mapping_value(pack, name)
    if section:
        return section
    return _mapping_value(domain_lexicon, name)


def _section_terms(domain_lexicon: Any, section_name: str, key: str) -> list[str]:
    return _dedupe(_list_values(_section(domain_lexicon, section_name).get(key)))


@dataclass(frozen=True)
class DomainCandidatePolicy:
    domain_lexicon: Any
    legacy_scope_labels: dict[str, str] = field(default_factory=dict)
    legacy_generic_method_display_names: set[str] = field(default_factory=set)
    legacy_generic_object_labels: set[str] = field(default_factory=set)
    legacy_generic_tech_object_slots: set[str] = field(default_factory=set)
    legacy_surface_object_patterns: Sequence[SurfacePattern] = field(default_factory=tuple)
    legacy_surface_task_patterns: Sequence[SurfacePattern] = field(default_factory=tuple)
    legacy_title_topic_anchors: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "legacy_scope_labels", dict(self.legacy_scope_labels or {}))
        object.__setattr__(
            self,
            "legacy_generic_method_display_names",
            set(self.legacy_generic_method_display_names or set()),
        )
        object.__setattr__(
            self,
            "legacy_generic_object_labels",
            set(self.legacy_generic_object_labels or set()),
        )
        object.__setattr__(
            self,
            "legacy_generic_tech_object_slots",
            set(self.legacy_generic_tech_object_slots or set()),
        )
        object.__setattr__(
            self,
            "legacy_surface_object_patterns",
            tuple(self.legacy_surface_object_patterns or ()),
        )
        object.__setattr__(
            self,
            "legacy_surface_task_patterns",
            tuple(self.legacy_surface_task_patterns or ()),
        )
        object.__setattr__(
            self,
            "legacy_title_topic_anchors",
            tuple(self.legacy_title_topic_anchors or ()),
        )

    @property
    def pack_id(self) -> str:
        pack = _attached_pack(self.domain_lexicon)
        for source in (pack, self.domain_lexicon, _section(self.domain_lexicon, "domain_identity")):
            for key in ("pack_id", "domain_pack_id", "field_id"):
                value = str(_value(source, key, "") or "").strip()
                if value:
                    return value
        return "neutral"

    @property
    def is_legacy_humanoid(self) -> bool:
        return bool(_value(self.domain_lexicon, "use_legacy_robot_rules", False)) or (
            self.pack_id in LEGACY_HUMANOID_PACK_IDS
        )

    def scope_label(self, scope: str) -> str:
        raw_scope = str(scope or "").strip()
        reporting = _section(self.domain_lexicon, "reporting")
        display_labels = reporting.get("display_labels", {})
        if isinstance(display_labels, dict):
            label = str(display_labels.get(raw_scope, "") or "").strip()
            if label:
                return label
        if self.is_legacy_humanoid and self.legacy_scope_labels:
            return self.legacy_scope_labels.get(raw_scope, raw_scope)

        observation_scopes = _section(self.domain_lexicon, "observation_scopes")
        domain_identity = _section(self.domain_lexicon, "domain_identity")
        main_scope = str(observation_scopes.get("main_scope") or "").strip()
        field_name = str(domain_identity.get("field_name") or "").strip()
        if raw_scope and raw_scope == main_scope and field_name:
            return field_name
        return raw_scope

    def generic_terms(self) -> tuple[str, ...]:
        terms = _section_terms(self.domain_lexicon, "candidate_formation", "generic_terms")
        if not terms:
            terms = _dedupe(_list_values(_value(self.domain_lexicon, "generic_terms", [])))
        return tuple(terms)

    def shell_terms(self) -> tuple[str, ...]:
        terms = _section_terms(self.domain_lexicon, "candidate_formation", "shell_terms")
        if not terms:
            terms = _dedupe(_list_values(_value(self.domain_lexicon, "shell_terms", [])))
        return tuple(terms)

    def technical_anchor_terms(self) -> tuple[str, ...]:
        candidate_formation = _section(self.domain_lexicon, "candidate_formation")
        observation_scopes = _section(self.domain_lexicon, "observation_scopes")
        domain_identity = _section(self.domain_lexicon, "domain_identity")
        values: list[Any] = []
        for key in (
            "technical_object_types",
            "mechanism_types",
            "task_or_performance_types",
            "data_or_method_types",
            "scene_or_application_types",
        ):
            values.extend(_list_values(candidate_formation.get(key)))
        values.extend(_list_values(observation_scopes.get("technical_object")))
        values.extend(_list_values(domain_identity.get("core_keywords")))
        values.extend(_list_values(domain_identity.get("english_terms")))
        if not values:
            values.extend(_list_values(_value(self.domain_lexicon, "domain_anchor_terms", [])))
        return tuple(_dedupe(values))

    def generic_method_display_names(self) -> set[str]:
        if self.is_legacy_humanoid and self.legacy_generic_method_display_names:
            return set(self.legacy_generic_method_display_names)
        return {
            normalized
            for normalized in (
                normalize_signal_phrase(value)
                for value in (*self.generic_terms(), *self.shell_terms())
            )
            if normalized
        }

    def generic_object_labels(self) -> set[str]:
        if self.is_legacy_humanoid and self.legacy_generic_object_labels:
            return set(self.legacy_generic_object_labels)
        return set(_dedupe([*self.generic_terms(), *self.shell_terms()]))

    def generic_tech_object_slots(self) -> set[str]:
        if self.is_legacy_humanoid and self.legacy_generic_tech_object_slots:
            return set(self.legacy_generic_tech_object_slots)
        return set(_dedupe([*self.generic_terms(), *self.shell_terms()]))

    def surface_object_patterns(self) -> Sequence[SurfacePattern]:
        if self.is_legacy_humanoid:
            return tuple(self.legacy_surface_object_patterns)
        return ()

    def surface_task_patterns(self) -> Sequence[SurfacePattern]:
        if self.is_legacy_humanoid:
            return tuple(self.legacy_surface_task_patterns)
        return ()

    def title_topic_anchors(self) -> tuple[str, ...]:
        if self.is_legacy_humanoid and self.legacy_title_topic_anchors:
            return tuple(_dedupe(self.legacy_title_topic_anchors))
        candidate_formation = _section(self.domain_lexicon, "candidate_formation")
        values: list[Any] = []
        for key in (
            "technical_object_types",
            "mechanism_types",
            "task_or_performance_types",
            "scene_or_application_types",
        ):
            values.extend(_list_values(candidate_formation.get(key)))
        if not values:
            values.extend(_list_values(_value(self.domain_lexicon, "domain_anchor_terms", [])))
            for alias_map_name in (
                "object_aliases",
                "mechanism_aliases",
                "task_aliases",
                "scene_aliases",
            ):
                alias_map = _value(self.domain_lexicon, alias_map_name, {})
                if isinstance(alias_map, Mapping):
                    values.extend(alias_map.keys())
        return tuple(_dedupe(values))

    def is_generic_or_shell(self, value: str) -> bool:
        checker = _value(self.domain_lexicon, "is_generic_or_shell")
        if callable(checker):
            return bool(checker(value))
        normalized = normalize_signal_phrase(value)
        if not normalized:
            return True
        generic_shell = {
            normalize_signal_phrase(term)
            for term in (*self.generic_terms(), *self.shell_terms())
            if normalize_signal_phrase(term)
        }
        return normalized in generic_shell


def build_domain_candidate_policy(
    domain_lexicon: Any,
    *,
    legacy_scope_labels: dict[str, str] | None = None,
    legacy_generic_method_display_names: set[str] | None = None,
    legacy_generic_object_labels: set[str] | None = None,
    legacy_generic_tech_object_slots: set[str] | None = None,
    legacy_surface_object_patterns: Sequence[SurfacePattern] = (),
    legacy_surface_task_patterns: Sequence[SurfacePattern] = (),
    legacy_title_topic_anchors: Sequence[str] = (),
) -> DomainCandidatePolicy:
    return DomainCandidatePolicy(
        domain_lexicon=domain_lexicon,
        legacy_scope_labels=dict(legacy_scope_labels or {}),
        legacy_generic_method_display_names=set(legacy_generic_method_display_names or set()),
        legacy_generic_object_labels=set(legacy_generic_object_labels or set()),
        legacy_generic_tech_object_slots=set(legacy_generic_tech_object_slots or set()),
        legacy_surface_object_patterns=tuple(legacy_surface_object_patterns or ()),
        legacy_surface_task_patterns=tuple(legacy_surface_task_patterns or ()),
        legacy_title_topic_anchors=tuple(legacy_title_topic_anchors or ()),
    )


__all__ = [
    "DomainCandidatePolicy",
    "LEGACY_HUMANOID_PACK_IDS",
    "SurfacePattern",
    "_dedupe",
    "build_domain_candidate_policy",
]
