from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import yaml

from .models import DomainContext, DomainPack


DOMAIN_PACK_DIR = Path(__file__).resolve().parents[1] / "config" / "domain_packs"
MEMORY_DOMAIN_PACK_DIR = Path(__file__).resolve().parents[2] / "memory" / "domain_packs"
NEUTRAL_PACK_ID = "neutral"


def _read_pack_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Domain Pack file must contain a mapping: {path}")
    return payload


def _candidate_preset_paths(pack_id: str):
    cleaned = str(pack_id or "").strip()
    if not cleaned:
        return []
    return [
        DOMAIN_PACK_DIR / f"{cleaned}.yaml",
        DOMAIN_PACK_DIR / f"{cleaned}.yml",
        DOMAIN_PACK_DIR / f"{cleaned}.json",
    ]


def _resolve_pack_path(pack_ref: Union[str, Path, None]) -> Path | None:
    if isinstance(pack_ref, Path):
        return pack_ref if pack_ref.exists() else None
    text = str(pack_ref or "").strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct
    for path in _candidate_preset_paths(text):
        if path.exists():
            return path
    # Search memory/domain_packs/ for a YAML file whose pack_id matches
    if MEMORY_DOMAIN_PACK_DIR.exists():
        for yaml_file in sorted(
            MEMORY_DOMAIN_PACK_DIR.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                payload = _read_pack_file(yaml_file)
                if payload.get("pack_id") == text:
                    return yaml_file
            except Exception:
                continue
    return None


def _neutral_payload() -> dict:
    return {
        "schema_version": "domain_pack_v1",
        "pack_id": NEUTRAL_PACK_ID,
        "pack_name": "Neutral Domain Pack",
        "pack_version": "preset.v1",
        "source": {
            "mode": "preset",
            "model": "",
            "prompt_version": "",
            "based_on_user_input": {
                "field_id": NEUTRAL_PACK_ID,
                "field_name": "Neutral",
                "keywords": [],
                "synonyms": [],
                "exclude_terms": [],
            },
        },
        "domain_identity": {
            "field_id": NEUTRAL_PACK_ID,
            "field_name": "Neutral",
            "domain_boundary": "No domain-specific assumptions.",
        },
    }


def load_domain_pack(pack_ref: Union[str, Path, None] = NEUTRAL_PACK_ID) -> DomainPack:
    path = _resolve_pack_path(pack_ref)
    if path is not None:
        return DomainPack.from_dict(_read_pack_file(path))
    neutral_path = _resolve_pack_path(NEUTRAL_PACK_ID)
    if neutral_path is not None:
        return DomainPack.from_dict(_read_pack_file(neutral_path))
    return DomainPack.from_dict(_neutral_payload())


def load_domain_context(pack_ref: Union[str, Path, None] = NEUTRAL_PACK_ID) -> DomainContext:
    pack = load_domain_pack(pack_ref)
    return DomainContext.from_pack(pack)
