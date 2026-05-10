"""
Optional AI agent for business rules that are hard to express as JSON + pandas eval.

Examples: cross-column plausibility, tier vs discount sanity, informal "smell tests".
This module is separate from ``validator.py`` so declarative validation stays small.

Wire-in from ``app.agents.validator`` after declarative checks pass::

    from app.agents.hard_business_rules_ai import simulate_ai_hard_business_rules
    errors.extend(await simulate_ai_hard_business_rules(df, rules))

Keep ``asyncio.wait_for`` so latency stays bounded; handle API errors like the main LLM pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd

from app.services.llm import anthropic_client
from app.utils.logger import logger


def _max_sample_rows() -> int:
    return int(os.getenv("VALIDATOR_MAX_SAMPLE_ROWS", "15"))


def _llm_timeout_sec() -> float:
    return float(os.getenv("VALIDATOR_LLM_TIMEOUT_SEC", "2"))


def _parse_violations_json(text: str) -> List[str]:
    """Parse model JSON; strip markdown fences; never raise."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("hard-rules AI: could not parse JSON from model output")
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


async def simulate_ai_hard_business_rules(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
) -> List[str]:
    """
    AI reviews richer context than the light pass in ``validator._llm_extra_violations``.

    Returns a list of error strings (same shape as declarative validator). On missing
    key, timeout, or API error, returns [] and logs — does not raise.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        logger.info("hard-rules AI skipped — ANTHROPIC_API_KEY unset or placeholder")
        return []

    max_rows = _max_sample_rows()
    timeout_sec = _llm_timeout_sec()

    payload = {
        "declarative_rules_applied": [r.get("name") for r in rules],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "row_count": len(df),
        "sample_rows": df.head(max_rows).to_dict(orient="records"),
    }
    user_message = json.dumps(payload, ensure_ascii=False, default=str)
    system_prompt = (
        "You are a domain validator. Beyond simple numeric thresholds, flag "
        "only clear inconsistencies (e.g. incompatible combinations). "
        'Reply with JSON only: {"violations":[{"rule":"logical_id","detail":"..."}]}'
    )

    async def _call() -> str:
        return await anthropic_client.chat_completion(system_prompt, user_message)

    try:
        raw = await asyncio.wait_for(_call(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(
            "hard-rules AI timed out after %ss — continuing without extra flags",
            timeout_sec,
        )
        return []
    except Exception as e:
        logger.warning("hard-rules AI error (non-fatal): %s", e)
        return []

    return _parse_violations_json(raw)
