import json
import logging
from copy import deepcopy
from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.llm import provider
from app.services.tools import availiable_tools

log = logging.getLogger("app")

router = APIRouter()



class QueryRequest(BaseModel):
    query: List[Dict]

@router.post("/rag")
async def rag_endpoint(request: QueryRequest):
    history = deepcopy(request.query)
    log.debug("Incoming history: %s", history)
    last_message_content = history[-1].get("content", "") if history else ""

    agent_answer = provider.chat(
        messages=history,
        temperature=0.7,
        functions=availiable_tools,
        function_call="auto"
    )

    log.debug(f"Generated answer: {agent_answer}")
    return {"answer": agent_answer}
