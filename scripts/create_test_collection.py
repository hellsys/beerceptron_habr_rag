"""
Создание тестовой коллекции с ограниченным количеством данных.
- 200 вопросов из single_hop датасета
- 200 вопросов из multi_hop датасета  
- Соответствующие чанки для этих вопросов
- Примерно столько же дополнительных документов для реалистичности
"""

import asyncio
import datetime
import json
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.asyncio import tqdm

from habr_rag.config import (
    CHUNKS_FILE,
    COLLECTION_PREFIX,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MULTI_HOP_DATASET,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    SINGLE_HOP_DATASET,
    DATASETS_DIR,
)
from habr_rag.embeddings import CustomBGEEmbeddings

load_dotenv()

# Конфигурация
QUESTIONS_PER_DATASET = 200  # Количество вопросов из каждого датасета
BATCH_SIZE = 50
MAX_CONCURRENT_UPLOADS = 5
EMBEDDING_SIZE = 1024


def load_chunks(filepath):
    """Загрузка чанков из pickle файла."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def sample_questions_and_get_ids(n_questions=200):
    """
    Выбирает n вопросов из каждого датасета и возвращает:
    - sampled_single_hop: список отобранных вопросов single_hop
    - sampled_multi_hop: список отобранных вопросов multi_hop
    - required_article_ids: set всех нужных article_id
    - required_chunk_ids: set всех нужных chunk_id
    """
    sampled_single_hop = []
    sampled_multi_hop = []
    required_article_ids = set()
    required_chunk_ids = set()
    
    # Single hop
    if os.path.exists(SINGLE_HOP_DATASET):
        with open(SINGLE_HOP_DATASET, "r", encoding="utf-8") as f:
            single_hop_data = json.load(f)
        
        # Берём случайные n вопросов
        sampled_single_hop = random.sample(
            single_hop_data, 
            min(n_questions, len(single_hop_data))
        )
        
        for item in sampled_single_hop:
            if "article_id" in item:
                required_article_ids.add(str(item["article_id"]))
            if "chunk_id" in item:
                required_chunk_ids.add(item["chunk_id"])
        
        print(f"Selected {len(sampled_single_hop)} questions from single_hop dataset")
    else:
        print(f"Warning: {SINGLE_HOP_DATASET} not found")
    
    # Multi hop
    if os.path.exists(MULTI_HOP_DATASET):
        with open(MULTI_HOP_DATASET, "r", encoding="utf-8") as f:
            multi_hop_data = json.load(f)
        
        # Берём случайные n вопросов
        sampled_multi_hop = random.sample(
            multi_hop_data,
            min(n_questions, len(multi_hop_data))
        )
        
        for item in sampled_multi_hop:
            if "source_article_ids" in item:
                for aid in item["source_article_ids"]:
                    required_article_ids.add(str(aid))
            if "source_chunk_ids" in item:
                for cid in item["source_chunk_ids"]:
                    required_chunk_ids.add(cid)
        
        print(f"Selected {len(sampled_multi_hop)} questions from multi_hop dataset")
    else:
        print(f"Warning: {MULTI_HOP_DATASET} not found")
    
    print(f"Total unique article IDs needed: {len(required_article_ids)}")
    print(f"Total unique chunk IDs needed: {len(required_chunk_ids)}")
    
    return sampled_single_hop, sampled_multi_hop, required_article_ids, required_chunk_ids


def select_documents(all_docs, required_article_ids, required_chunk_ids, noise_ratio=1.0):
    """
    Выбирает документы:
    1. Находим article_ids для всех required_chunk_ids
    2. Объединяем с required_article_ids
    3. Загружаем ВСЕ чанки для этих статей
    4. Добавляем noise документы
    """
    # Шаг 1: Найти article_ids для всех required_chunk_ids
    chunk_to_article = {}
    for doc in all_docs:
        chunk_id = doc.metadata.get("chunk_id")
        article_id = str(doc.metadata.get("id", ""))
        if chunk_id:
            chunk_to_article[chunk_id] = article_id
    
    # Расширяем required_article_ids статьями, содержащими нужные чанки
    extended_article_ids = set(required_article_ids)
    found_chunk_ids = set()
    missing_chunk_ids = set()
    
    for chunk_id in required_chunk_ids:
        if chunk_id in chunk_to_article:
            extended_article_ids.add(chunk_to_article[chunk_id])
            found_chunk_ids.add(chunk_id)
        else:
            missing_chunk_ids.add(chunk_id)
    
    print(f"Original required article IDs: {len(required_article_ids)}")
    print(f"Extended with articles from chunk_ids: {len(extended_article_ids)}")
    print(f"Found chunk_ids: {len(found_chunk_ids)}/{len(required_chunk_ids)}")
    if missing_chunk_ids:
        print(f"WARNING: Missing {len(missing_chunk_ids)} chunk_ids in chunked_docs.pkl!")
        print(f"  First 5 missing: {list(missing_chunk_ids)[:5]}")
    
    # Шаг 2: Выбираем все чанки для extended_article_ids
    priority_docs = []
    remaining_docs = []
    
    for doc in all_docs:
        article_id = str(doc.metadata.get("id", ""))
        if article_id in extended_article_ids:
            priority_docs.append(doc)
        else:
            remaining_docs.append(doc)
    
    print(f"Priority documents (all chunks for required articles): {len(priority_docs)}")
    print(f"Available remaining documents: {len(remaining_docs)}")
    
    # Шаг 3: Добавляем noise документы
    noise_count = int(len(priority_docs) * noise_ratio)
    noise_docs = random.sample(remaining_docs, min(noise_count, len(remaining_docs)))
    
    print(f"Adding {len(noise_docs)} noise documents (ratio {noise_ratio})")
    
    final_docs = priority_docs + noise_docs
    print(f"Total documents to upload: {len(final_docs)}")
    
    return final_docs, priority_docs, noise_docs


async def upload_batch(vector_store, batch, semaphore, max_retries=5, initial_delay=1):
    """Загрузка батча с retry логикой."""
    async with semaphore:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                await vector_store.aadd_documents(batch)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to upload batch after {max_retries} attempts: {e}")
                    raise e
                print(f"Error uploading batch (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2


async def main():
    random.seed(42)  # Для воспроизводимости
    
    print("=" * 60)
    print("Creating Test Collection")
    print("=" * 60)
    
    # 1. Загружаем все чанки
    print(f"\nLoading chunks from {CHUNKS_FILE}...")
    all_docs = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(all_docs)} documents total.")
    
    # Диагностика: проверяем структуру метаданных первого документа
    if all_docs:
        sample_doc = all_docs[0]
        print(f"\nSample document metadata keys: {list(sample_doc.metadata.keys())}")
        print(f"Sample metadata: {sample_doc.metadata}")
    
    # 2. Выбираем вопросы и получаем нужные ID
    print("\nSampling questions from datasets...")
    sampled_single, sampled_multi, required_article_ids, required_chunk_ids = \
        sample_questions_and_get_ids(QUESTIONS_PER_DATASET)
    
    # Диагностика: проверяем, есть ли required_chunk_ids в all_docs
    print("\n--- DIAGNOSTIC: Checking if required chunks exist in chunked_docs.pkl ---")
    all_chunk_ids_in_docs = {doc.metadata.get("chunk_id") for doc in all_docs if doc.metadata.get("chunk_id")}
    all_article_ids_in_docs = {str(doc.metadata.get("id", "")) for doc in all_docs}
    
    print(f"Total unique chunk_ids in chunked_docs.pkl: {len(all_chunk_ids_in_docs)}")
    print(f"Total unique article_ids in chunked_docs.pkl: {len(all_article_ids_in_docs)}")
    
    # Сколько required_chunk_ids есть в chunked_docs.pkl
    found_in_docs = required_chunk_ids & all_chunk_ids_in_docs
    missing_from_docs = required_chunk_ids - all_chunk_ids_in_docs
    print(f"Required chunk_ids found in chunked_docs.pkl: {len(found_in_docs)}/{len(required_chunk_ids)}")
    if missing_from_docs:
        print(f"MISSING chunk_ids (not in chunked_docs.pkl): {len(missing_from_docs)}")
        print(f"  First 5 missing: {list(missing_from_docs)[:5]}")
    
    # Сколько required_article_ids есть в chunked_docs.pkl
    found_articles = required_article_ids & all_article_ids_in_docs
    missing_articles = required_article_ids - all_article_ids_in_docs
    print(f"Required article_ids found in chunked_docs.pkl: {len(found_articles)}/{len(required_article_ids)}")
    if missing_articles:
        print(f"MISSING article_ids (not in chunked_docs.pkl): {len(missing_articles)}")
        print(f"  First 5 missing: {list(missing_articles)[:5]}")
    print("--- END DIAGNOSTIC ---\n")
    
    # 3. Сохраняем отобранные датасеты
    test_single_hop_path = DATASETS_DIR / "test_single_hop.json"
    test_multi_hop_path = DATASETS_DIR / "test_multi_hop.json"
    
    with open(test_single_hop_path, "w", encoding="utf-8") as f:
        json.dump(sampled_single, f, ensure_ascii=False, indent=2)
    print(f"\nSaved test single_hop dataset to {test_single_hop_path}")
    
    with open(test_multi_hop_path, "w", encoding="utf-8") as f:
        json.dump(sampled_multi, f, ensure_ascii=False, indent=2)
    print(f"Saved test multi_hop dataset to {test_multi_hop_path}")
    
    # 4. Выбираем документы
    print("\nSelecting documents...")
    final_docs, priority_docs, noise_docs = select_documents(
        all_docs, required_article_ids, required_chunk_ids, noise_ratio=1.0
    )
    
    # 5. Инициализируем embeddings и клиент
    print("\nInitializing embeddings...")
    embeddings = CustomBGEEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # 6. Создаём коллекцию
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    collection_name = f"{COLLECTION_PREFIX}_test_{today}"
    
    print(f"\nTarget collection: {collection_name}")
    
    if client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists. Deleting...")
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_SIZE, 
            distance=models.Distance.COSINE
        ),
        # Оптимизации для небольшой коллекции
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            on_disk=False,  # В RAM для скорости
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=5000,  # Быстрее индексируем
        ),
    )
    print(f"Created collection {collection_name}")
    
    # 7. Загружаем документы
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_collection_config=False,
    )
    
    print(f"\nUploading {len(final_docs)} documents in batches of {BATCH_SIZE}...")
    
    batches = [final_docs[i:i + BATCH_SIZE] for i in range(0, len(final_docs), BATCH_SIZE)]
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    tasks = [upload_batch(vector_store, batch, semaphore) for batch in batches]
    
    for f in tqdm.as_completed(tasks, total=len(batches), desc="Uploading batches"):
        await f
    
    # 8. Проверяем результат
    collection_info = client.get_collection(collection_name)
    print(f"\n{'=' * 60}")
    print("Upload Complete!")
    print(f"{'=' * 60}")
    print(f"Collection: {collection_name}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Indexed vectors: {collection_info.indexed_vectors_count}")
    print(f"\nTest datasets saved to:")
    print(f"  - {test_single_hop_path}")
    print(f"  - {test_multi_hop_path}")
    print(f"\nTo run evaluation, update evaluate_rag.py to use:")
    print(f"  collection_name = '{collection_name}'")
    print(f"  SINGLE_HOP_DATASET = '{test_single_hop_path}'")
    print(f"  MULTI_HOP_DATASET = '{test_multi_hop_path}'")


if __name__ == "__main__":
    asyncio.run(main())

