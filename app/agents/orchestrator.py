from typing import Any, Dict, List

import pandas as pd

from app.agents import validator as _validator_register  # noqa: F401 — registers dispatcher
from app.agents.query_agent import QueryAgent
from app.utils.dispatcher import dispatcher
from app.utils.logger import logger


class Orchestrator:
    def __init__(self):
        self.query_agent = QueryAgent()

    async def process_request(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Processing user query: {user_query}")
        
        # 1. Generate and execute SQL
        query_result = await self.query_agent.execute(user_query)
        
        if not query_result.get("success"):
            return {"error": "Failed to generate valid results"}

        dataset = query_result.get("data")

        # --- Flow: run registered validators on the result set before returning to callers ---
        validation_results: List[Dict[str, Any]] = []
        if isinstance(dataset, pd.DataFrame):
            validation_results = await dispatcher.run_validators(dataset)

        validation_ok = (
            all(r.get("passed") for r in validation_results)
            if validation_results
            else True
        )

        return {
            "status": "success",
            "data": dataset,
            "metadata": query_result.get("metadata"),
            "validation": validation_results,
            "validation_ok": validation_ok,
        }