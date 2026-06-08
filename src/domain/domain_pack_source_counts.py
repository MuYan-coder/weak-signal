from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from .models import DomainPack


DEFAULT_SOURCE_TYPES = ("paper", "news", "policy", "report", "patent")


@dataclass(frozen=True)
class SourceCountRecommendationRow:
    source_type: str
    weight: float
    available_count: int
    recommended_count: int
    final_count: int
    limit_reason: str = "weight"


@dataclass(frozen=True)
class SourceCountRecommendation:
    rows: tuple[SourceCountRecommendationRow, ...]

    @property
    def rows_by_source(self) -> Dict[str, SourceCountRecommendationRow]:
        return {row.source_type: row for row in self.rows}

    @property
    def recommended_counts(self) -> Dict[str, int]:
        return {row.source_type: row.recommended_count for row in self.rows}

    @property
    def final_counts(self) -> Dict[str, int]:
        return {row.source_type: row.final_count for row in self.rows}

    @property
    def total_count(self) -> int:
        return sum(self.final_counts.values())


def recommend_source_counts(
    domain_pack: DomainPack,
    *,
    available_counts: Mapping[str, Any] | None = None,
    total_sample_size: int | None = None,
    explicit_counts: Mapping[str, Any] | None = None,
    source_types: Iterable[str] | None = None,
) -> SourceCountRecommendation:
    """Recommend SourceQuery.counts from Domain Pack source weights.

    When explicit_counts is supplied, those counts become the final counts.
    The weighted recommendation is still returned so the UI can show the
    difference without reimplementing allocation rules.
    """
    available = _clean_int_map(available_counts or {})
    weights = _source_weights(domain_pack)
    policy = _count_policy(domain_pack)
    ordered_sources = _ordered_sources(source_types, available, weights, explicit_counts)

    # 如果所有数据源的可用数量都为0（例如初始化状态），我们临时假设一个足够大的可用量来计算理想推荐值
    if sum(available.values()) == 0:
        available = {source: 1000 for source in ordered_sources}

    default_total = _clean_int(policy.get("default_total_sample_size"), default=0)
    target_total = _clean_int(total_sample_size, default=0) or default_total
    if target_total <= 0:
        target_total = sum(available.get(source, 0) for source in ordered_sources)

    recommended = _allocate_weighted_counts(
        ordered_sources,
        weights=weights,
        available=available,
        total_sample_size=target_total,
        min_per_source=_clean_int_map(policy.get("min_per_source", {})),
        max_per_source=_clean_int_map(policy.get("max_per_source", {})),
    )

    final_counts = dict(recommended)
    explicit = _clean_int_map(explicit_counts or {})
    explicit_sources = set(explicit)
    if explicit_counts is not None:
        final_counts = {
            source: _cap_to_available(explicit.get(source, 0), available.get(source, 0))
            for source in ordered_sources
            if source in explicit_sources
        }

    rows = []
    for source in ordered_sources:
        if explicit_counts is not None and source not in explicit_sources:
            continue
        recommended_count = recommended.get(source, 0)
        final_count = final_counts.get(source, 0)
        rows.append(
            SourceCountRecommendationRow(
                source_type=source,
                weight=weights.get(source, 0.0),
                available_count=available.get(source, 0),
                recommended_count=recommended_count,
                final_count=final_count,
                limit_reason=_limit_reason(
                    source,
                    recommended_count,
                    final_count,
                    explicit_counts=explicit_counts,
                    available=available,
                    weights=weights,
                    min_per_source=_clean_int_map(policy.get("min_per_source", {})),
                    max_per_source=_clean_int_map(policy.get("max_per_source", {})),
                ),
            )
        )
    return SourceCountRecommendation(tuple(rows))


def _source_weights(domain_pack: DomainPack) -> Dict[str, float]:
    strategy = domain_pack.search_strategy if isinstance(domain_pack.search_strategy, dict) else {}
    raw_weights = strategy.get("source_type_weights", {})
    if not isinstance(raw_weights, dict):
        raw_weights = {}
    weights = {}
    for source, value in raw_weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 0.0
        weights[str(source)] = max(weight, 0.0)
    return weights


def _count_policy(domain_pack: DomainPack) -> Dict[str, Any]:
    strategy = domain_pack.search_strategy if isinstance(domain_pack.search_strategy, dict) else {}
    policy = strategy.get("count_policy", {})
    return policy if isinstance(policy, dict) else {}


def _ordered_sources(
    source_types: Iterable[str] | None,
    available: Mapping[str, int],
    weights: Mapping[str, float],
    explicit_counts: Mapping[str, Any] | None,
) -> list[str]:
    requested = list(source_types or [])
    if explicit_counts is not None:
        requested.extend(str(source) for source in explicit_counts)
    requested.extend(str(source) for source in available)
    requested.extend(str(source) for source in weights)
    requested.extend(DEFAULT_SOURCE_TYPES)

    seen = set()
    ordered = []
    for source in requested:
        cleaned = str(source or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _allocate_weighted_counts(
    sources: list[str],
    *,
    weights: Mapping[str, float],
    available: Mapping[str, int],
    total_sample_size: int,
    min_per_source: Mapping[str, int],
    max_per_source: Mapping[str, int],
) -> Dict[str, int]:
    positive_sources = [source for source in sources if weights.get(source, 0.0) > 0]
    if not positive_sources or total_sample_size <= 0:
        return {source: 0 for source in sources}

    target_total = min(max(total_sample_size, 0), sum(max(available.get(source, 0), 0) for source in positive_sources))
    ideal = {source: 0 for source in sources}
    for source, count in _largest_remainder_allocation(
        positive_sources,
        weights=weights,
        total=target_total,
    ).items():
        ideal[source] = count

    caps = {}
    for source in positive_sources:
        available_cap = max(available.get(source, 0), 0)
        policy_cap = max_per_source.get(source)
        caps[source] = min(available_cap, policy_cap) if policy_cap is not None else available_cap

    counts = dict(ideal)
    for source in positive_sources:
        minimum = min(max(min_per_source.get(source, 0), 0), caps[source])
        counts[source] = max(counts[source], minimum)
        counts[source] = min(counts[source], caps[source])

    if sum(counts.values()) > target_total:
        return _trim_to_total(counts, target_total, weights)

    remaining = target_total - sum(counts.values())
    while remaining > 0:
        eligible = [source for source in positive_sources if counts[source] < caps[source]]
        if not eligible:
            break
        catch_up_sources = [source for source in eligible if counts[source] <= ideal.get(source, 0)]
        allocation_sources = catch_up_sources or eligible
        additions = _largest_remainder_allocation(
            allocation_sources,
            weights=weights,
            total=remaining,
            capacities={source: caps[source] - counts[source] for source in allocation_sources},
        )
        added = sum(additions.values())

        if added <= 0:
            break
        for source, addition in additions.items():
            counts[source] += addition
        remaining -= added

    return counts


def _largest_remainder_allocation(
    sources: list[str],
    *,
    weights: Mapping[str, float],
    total: int,
    capacities: Mapping[str, int] | None = None,
) -> Dict[str, int]:
    if total <= 0 or not sources:
        return {source: 0 for source in sources}
    weight_total = sum(weights.get(source, 0.0) for source in sources)
    if weight_total <= 0:
        return {source: 0 for source in sources}

    capacities = capacities or {}
    result = {}
    remainders = []
    for source in sources:
        capacity = capacities.get(source, total)
        raw_share = total * (weights.get(source, 0.0) / weight_total)
        count = min(max(capacity, 0), int(raw_share))
        result[source] = count
        remainders.append((raw_share - int(raw_share), weights.get(source, 0.0), source))

    remaining = total - sum(result.values())
    for _remainder, _weight, source in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        capacity = capacities.get(source, total)
        if result[source] >= capacity:
            continue
        result[source] += 1
        remaining -= 1
    return result


def _trim_to_total(counts: Dict[str, int], total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    trimmed = dict(counts)
    while sum(trimmed.values()) > total:
        candidates = [source for source, count in trimmed.items() if count > 0]
        if not candidates:
            break
        source = min(candidates, key=lambda item: (weights.get(item, 0.0), item))
        trimmed[source] -= 1
    return trimmed


def _limit_reason(
    source: str,
    recommended_count: int,
    final_count: int,
    *,
    explicit_counts: Mapping[str, Any] | None,
    available: Mapping[str, int],
    weights: Mapping[str, float],
    min_per_source: Mapping[str, int],
    max_per_source: Mapping[str, int],
) -> str:
    if explicit_counts is not None:
        return "user_explicit"
    if weights.get(source, 0.0) <= 0:
        return "zero_weight"
    if recommended_count >= available.get(source, 0) and available.get(source, 0) >= 0:
        return "available_cap"
    if source in max_per_source and final_count >= max_per_source[source]:
        return "policy_max"
    if source in min_per_source and final_count >= min_per_source[source]:
        return "policy_min"
    return "weight"


def _clean_int_map(values: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(values, Mapping):
        return {}
    return {str(key): _clean_int(value, default=0) for key, value in values.items()}


def _clean_int(value: Any, *, default: int) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _cap_to_available(count: int, available_count: int) -> int:
    if available_count <= 0:
        return max(count, 0)
    return min(max(count, 0), available_count)
