import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.core import config
from app.core.prompts import AGENT_SYSTEM_PROMPT

log = logging.getLogger("app")

client = OpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY
)

embed_client = OpenAI(
    base_url=config.EMBED_BASE_URL,
    api_key=config.EMBED_API_KEY
)

def generate(
    prompt: str,
    temperature: float = 0.0
) -> str:
    global log
    """
    Возвращает:
      "text": str
    """
    log.info(f"Generating LLM response with prompt: {prompt}")
    try:
        model = config.LLM_MODEL

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=config.MAX_TOKENS
        )
        log.info(f"LLM response: {resp}")
        choice = resp.choices[0]

        return choice.message.content
    except Exception as e:
        log = logging.getLogger("errors")
        log.error(f"Error generating LLM response: {e}")
        raise

def generate_with_history(
    history: list,
    temperature: float = 0.0
) -> str:
    global log
    """
    Возвращает:
      "text": str
    """
    log.info(f"Generating LLM response with history: {history}")
    model = config.LLM_MODEL
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=config.MAX_TOKENS
        )
        log.info(f"LLM response: {resp}")
        choice = resp.choices[0]

        return choice.message.content
    except Exception as e:
        log = logging.getLogger("errors")
        log.error(f"Error generating LLM response: {e}")
        raise

def get_embeddings(
    text: str
) -> str:
    global log
    """
    Возвращает:
      "embs": list[float]
    """
    log.info(f"Generating LLM embeddings with text: {text}")
    model = config.EMBED_MODEL

    try:
        resp = embed_client.embeddings.create(
            model=model,
            input=[text]
        )
        log.info(f"LLM embeddings: {resp}")
        embs = resp.data[0]

        return embs.embedding
    except Exception as e:
        log = logging.getLogger("errors")
        log.error(f"Error generating LLM embeddings: {e}")
        raise


def chat(
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Optional[str] = None,
    max_tokens: Optional[int] = None
):
    llm = ChatOpenAI(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        model=config.LLM_MODEL,
        temperature=temperature if temperature is not None else config.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS    
    )
    agent = create_agent(
        llm,
        tools=functions,
        system_prompt=AGENT_SYSTEM_PROMPT
    )
    result = agent.invoke(
        {"messages": messages},
    )
    result_answer = result['messages'][-1].content


    log.info("Invoking agent result: %s", result_answer)
    return result_answer