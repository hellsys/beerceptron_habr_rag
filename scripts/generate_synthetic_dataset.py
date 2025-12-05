import asyncio
import json
import os
import pickle
import random
import re
import sys
from typing import Any, Dict, List
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import AsyncOpenAI

from habr_rag.config import (
    CHUNKS_FILE,
    META_COLS_FILE,
    MODEL_NAME,
    MULTI_HOP_DATASET,
    SINGLE_HOP_DATASET,
    OPENAI_API_KEY,
)

load_dotenv()

NUM_SINGLE_HOP = 800
NUM_MULTI_HOP = 800

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def load_chunks(filepath: str) -> List[Document]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File {filepath} not found. Please ensure chunks are generated."
        )
    with open(filepath, "rb") as f:
        chunks = pickle.load(f)
    return chunks


def group_chunks_by_article(chunks: List[Document]) -> Dict[str, List[Document]]:
    grouped = {}
    for chunk in chunks:
        article_id = chunk.metadata.get("id")
        if not article_id:
            continue
        if article_id not in grouped:
            grouped[article_id] = []
        grouped[article_id].append(chunk)
    return grouped


async def generate_single_hop_question(
    article_id: str, article_chunks: List[Document]
) -> Dict[str, Any]:
    context_text = ""
    for chunk in article_chunks:
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        context_text += f"\n<chunk id='{chunk_id}'>\n{chunk.page_content}\n</chunk>\n"

    context_text = context_text[:100000]

    system_prompt = """
    Вы - эксперт в области формирования искуственных данных для RAG.

    Задача:
    1. Проанализируй предоставленные чанки одной статьи.
    2. Выбери один конкретный чанк (или несколько смежных), который содержит законченную мысль или факт.
    3. Сформулируй вопрос к этому факту. 
    
    КЛЮЧЕВЫЕ ТРЕБОВАНИЯ К ВОПРОСУ:
    - Вопрос должен звучать так, будто человек ищет информацию в интернете, НЕ ЗНАЯ о существовании этой статьи.
    - ИЗБЕГАЙ любых отсылок к "статье", "автору", "тексту" (например, НЕЛЬЗЯ: "Что автор говорит о...", "В тексте сказано...", "Согласно статье...").
    - Вопрос должен быть самостоятельным, понятным без контекста статьи.
    - Вопрос должен выглядеть как естественный поисковый запрос или вопрос на форуме.

    4. Сформулируй ответ, используя информацию ИЗ ЧАНКА.
    5. Укажи chunk_id, который содержит ответ.

    Выходной JSON формат:
    {
        "question": "Текст вопроса (без упоминания статьи/автора)",
        "answer": "Текст ответа",
        "chunk_id": "id_выбранного_чанка"
    }
    """

    user_prompt = f"""
    ID статьи: {article_id}
    
    Текст статьи (разбит на чанки):
    {context_text}
    """

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["article_id"] = article_id
        return data
    except Exception as e:
        print(f"Error generating single hop for article {article_id}: {e}")
        return None


async def generate_multi_hop_question(
    articles_group: List[tuple[str, List[Document]]],
) -> Dict[str, Any]:
    combined_context = ""
    source_article_ids = []

    for i, (art_id, chunks) in enumerate(articles_group):
        source_article_ids.append(art_id)
        combined_context += f"\n=== Статья {i + 1} (ID: {art_id}) ===\n"
        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", "unknown")
            combined_context += f"<chunk id='{chunk_id}'>\n{chunk.page_content[:2000]}...\n</chunk>\n"

    combined_context = combined_context[:100000]

    system_prompt = """
    Вы - эксперт в области формирования искуственных данных для RAG.

    Задача:
    1. Проанализируй чанки из НЕСКОЛЬКИХ разных статей.
    2. Придумай вопрос, для ответа на который НЕОБХОДИМО использовать информацию как минимум из двух разных источников.
    
    КЛЮЧЕВЫЕ ТРЕБОВАНИЯ К ВОПРОСУ:
    - Вопрос должен звучать как запрос человека, который хочет разобраться в теме, сравнить технологии или подходы.
    - Вопрос НЕ должен содержать упоминаний "статей", "документов", "авторов". 
    - Вопрос должен быть сформулирован так, будто пользователь не знает, из каких именно источников придет ответ.
    - ПРИМЕР ХОРОШЕГО ВОПРОСА: "В чем разница между подходом А и Б при решении задачи X?" или "Какие есть плюсы и минусы использования Y?" (если плюсы в одной статье, а минусы в другой).
    - ПРИМЕР ПЛОХОГО ВОПРОСА: "Что говорится в первой статье про Х по сравнению со второй?"

    3. Сформулируй синтезированный ответ.
    4. Укажи список chunk_id, которые были использованы для ответа.

    Выходной JSON формат:
    {
        "question": "Текст сложного вопроса (без мета-ссылок)",
        "answer": "Синтезированный ответ",
        "reasoning": "Почему этот вопрос требует нескольких источников",
        "source_chunk_ids": ["chunk_id_1", "chunk_id_2"]
    }
    """

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_context},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["source_article_ids"] = source_article_ids
        return data
    except Exception as e:
        print(f"Error generating multi hop: {e}")
        return None


def get_related_article_groups_from_chunks(
    grouped_chunks: Dict[str, List[Document]], df: pd.DataFrame, n_groups: int
) -> List[List[tuple[str, List[Document]]]]:
    tag_to_ids = {}
    for _, row in df.iterrows():
        index_type = type(list(grouped_chunks.keys())[0])
        art_id = index_type(row["id"])
        if art_id not in grouped_chunks:
            continue
        row_tags = row["tags"]
        lst = re.findall(r"'([^']*)'", row_tags)
        tags_array = list(lst)
        for tag in tags_array:
            if tag not in tag_to_ids:
                tag_to_ids[tag] = []
            tag_to_ids[tag].append(art_id)
    valid_tags = [t for t in tag_to_ids if len(tag_to_ids[t]) >= 2]
    print(f"Found {len(valid_tags)} tags with 2+ available articles.")

    groups = []
    attempts = 0
    max_attempts = n_groups * 20

    all_article_ids = list(grouped_chunks.keys())

    while len(groups) < n_groups and attempts < max_attempts:
        attempts += 1

        if valid_tags:
            tag = random.choice(valid_tags)
            ids = tag_to_ids[tag]
            if len(ids) >= 2:
                pair_ids = random.sample(ids, 2)
                group = [(pid, grouped_chunks[pid]) for pid in pair_ids]
                if group not in groups:
                    groups.append(group)
                    continue

        if attempts > max_attempts / 2:
            pair_ids = random.sample(all_article_ids, 2)
            group = [(pid, grouped_chunks[pid]) for pid in pair_ids]
            groups.append(group)

    while len(groups) < n_groups:
        pair_ids = random.sample(all_article_ids, 2)
        group = [(pid, grouped_chunks[pid]) for pid in pair_ids]
        groups.append(group)

    return groups


async def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    try:
        chunks = load_chunks(CHUNKS_FILE)
    except Exception as e:
        print(f"Failed to load chunks: {e}")
        return
    print(f"Loaded {len(chunks)} chunks.")

    articles_map = group_chunks_by_article(chunks)
    print(f"Grouped into {len(articles_map)} unique articles.")

    print(f"Loading metadata from {META_COLS_FILE}...")
    try:
        df = pd.read_csv(META_COLS_FILE)
        df["id"] = df["id"].astype(str)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    print(f"\nGenerating {NUM_SINGLE_HOP} single-hop questions...")

    article_ids = list(articles_map.keys())
    sample_ids = random.sample(article_ids, min(NUM_SINGLE_HOP, len(article_ids)))

    single_hop_tasks = []
    for aid in sample_ids:
        single_hop_tasks.append(generate_single_hop_question(aid, articles_map[aid]))

    single_hop_results = []
    for res in tqdm.tqdm(
        asyncio.as_completed(single_hop_tasks), total=len(single_hop_tasks)
    ):
        data = await res
        if data:
            single_hop_results.append(data)

    with open(SINGLE_HOP_DATASET, "w", encoding="utf-8") as f:
        json.dump(single_hop_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(single_hop_results)} single-hop examples to {SINGLE_HOP_DATASET}")

    print(f"\nGenerating {NUM_MULTI_HOP} multi-hop questions...")

    article_groups = get_related_article_groups_from_chunks(
        articles_map, df, NUM_MULTI_HOP
    )

    multi_hop_tasks = []
    for group in article_groups:
        multi_hop_tasks.append(generate_multi_hop_question(group))

    multi_hop_results = []
    for res in tqdm.tqdm(
        asyncio.as_completed(multi_hop_tasks), total=len(multi_hop_tasks)
    ):
        data = await res
        if data:
            multi_hop_results.append(data)

    with open(MULTI_HOP_DATASET, "w", encoding="utf-8") as f:
        json.dump(multi_hop_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(multi_hop_results)} multi-hop examples to {MULTI_HOP_DATASET}")


if __name__ == "__main__":
    asyncio.run(main())
