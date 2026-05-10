from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from app.services.llm import anthropic_client
from app.utils.dispatcher import dispatcher
from app.utils.logger import logger

_RULES_PATH = os.getenv("BUSINESS_RULES_PATH", "").strip()
_LLM_TIMEOUT = float(os.getenv("VALIDATOR_LLM_TIMEOUT_SEC", "2"))
_MAX_SAMPLE_ROWS = int(os.getenv("VALIDATOR_MAX_SAMPLE_ROWS", "15"))

_RESERVED_IDS: Set[str] = {
    "and",
    "or",
    "not",
    "in",
    "true",
    "false",
    "today",
}

def _load_rules() -> List[Dict[str, Any]]:
    """Load declarative rules from JSON; on failure return [] and log (fail-open)."""
    path = Path(_RULES_PATH)
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            logger.warning("business rules: 'rules' is not a list in %s", path)
            return []
        return rules
    except FileNotFoundError:
        logger.warning("business rules file not found: %s", path)
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("business rules could not be read (%s): %s", path, e)
        return []


def _identifiers_in_expression(expr: str) -> Set[str]:
    """Extract candidate column identifiers from a boolean expression."""
    return {
        n
        for n in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expr)
        if n.lower() not in _RESERVED_IDS
    }


def _coerce_work_frame(df: pd.DataFrame, columns: Set[str]) -> pd.DataFrame:
    """
    Working copy with legacy-safe coercion so eval does not crash on stringly-typed data.
    Columns whose names look like dates get datetime coercion; others numeric.
    """
    work = df.copy()
    for col in columns:
        if col not in work.columns:
            continue
        lower = col.lower()
        if lower.endswith("date") or lower.endswith("_at"):
            work[col] = pd.to_datetime(work[col], errors="coerce")
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def _evaluate_declared_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[str]:
    """
    Evaluate each rule.condition with pandas eval on a coerced frame.
    Conditions use pandas eval syntax. For "as of today" comparisons, write @today
    in the JSON rule; it is bound to pd.Timestamp.today().normalize() via local_dict.
    """
    errors: List[str] = []
    if not rules or df.empty:
        return errors

    for rule in rules:
        name = rule.get("name", "unnamed_rule")
        desc = rule.get("description", "")
        raw_condition = rule.get("condition", "")
        if not raw_condition or not isinstance(raw_condition, str):
            errors.append(f"{name}: missing or invalid condition")
            continue

        expr = raw_condition.strip()
        idents = _identifiers_in_expression(expr)
        missing = [c for c in idents if c not in df.columns]
        if missing:
            logger.warning(
                "rule %s skipped — columns not in DataFrame: %s", name, missing
            )
            continue

        try:
            work = _coerce_work_frame(df, idents)
            today = pd.Timestamp.today().normalize()
            violations = work.eval(
                expr, local_dict={"today": today}, engine="python"
            )
            if violations.dtype != bool:
                violations = violations.astype(bool)
        except Exception as e:
            logger.warning("rule %s eval failed (non-fatal): %s", name, e)
            continue

        bad_mask = violations.fillna(False)
        bad_idx = work.index[bad_mask]
        if len(bad_idx) == 0:
            continue

        sample = bad_idx[:5].tolist()
        errors.append(
            f"{name}: {desc} — {len(bad_idx)} row(s), e.g. index {sample}"
        )

    return errors


def _build_llm_user_message(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> str:
    """Compact dataset summary plus rules for the optional LLM pass."""
    rules_text = json.dumps(rules, ensure_ascii=False, indent=2)
    sample = df.head(_MAX_SAMPLE_ROWS)
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    parts = [
        "Declarative business rules (JSON):",
        rules_text,
        "",
        "Column dtypes:",
        json.dumps(dtypes, indent=2),
        "",
        f"Sample (up to {_MAX_SAMPLE_ROWS} rows):",
        sample.to_json(orient="records", date_format="iso", default_handler=str),
        "",
        'Reply with ONLY valid JSON: {"violations": [{"rule": "id", "detail": "..."}]}',
        "Use violations [] if nothing beyond the declarative rules is wrong.",
    ]
    return "\n".join(parts)


def _parse_llm_violations(text: str) -> List[str]:
    """Parse model JSON; never raise."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM validation: could not parse JSON from model output")
        return []
    items = data.get("violations")
    if items is None:
        items = data.get("extra_violations", [])
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("rule", "llm")
        detail = item.get("detail", "")
        out.append(f"{rid}: {detail}".strip())
    return out


async def _llm_extra_violations(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[str]:
    """
    Optional LLM pass for domain issues not captured by JSON rules.
    Timeout / API errors → log and return [] (do not fail the main request).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        logger.info("LLM validation skipped — ANTHROPIC_API_KEY unset or placeholder")
        return []

    system = (
        "You validate tabular query results against business rules. "
        "Be conservative; only flag clear domain inconsistencies. "
        "Output only the JSON object requested by the user."
    )
    user_msg = _build_llm_user_message(df, rules)

    async def _call() -> str:
        return await anthropic_client.chat_completion(system, user_msg)

    try:
        text = await asyncio.wait_for(_call(), timeout=_LLM_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(
            "LLM validation timed out after %ss — continuing without LLM flags",
            _LLM_TIMEOUT,
        )
        return []
    except Exception as e:
        logger.warning("LLM validation error (non-fatal): %s", e)
        return []

    return _parse_llm_violations(text)


# Optional second LLM pass for fuzzy / cross-column rules: ``hard_business_rules_ai``.


@dispatcher.register_validator(name="business_rules_validator")
async def validate_business_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates the dataset against business rules (declarative JSON + optional LLM assist).
    Returns {"valid": bool, "errors": list[str]}.
    """

    if df.empty:
        return {"valid": True, "errors": []}
    rules = _load_rules()

    errors = _evaluate_declared_rules(df, rules)

    if errors:
        logger.info("Skipping LLM validation — declarative rules already violated")
    else:
        llm_errs = await _llm_extra_violations(df, rules)
        errors.extend(llm_errs)

    return {"valid": len(errors) == 0, "errors": errors}
