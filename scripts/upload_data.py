import asyncio
import datetime
import json
import os
import pickle
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from habr_rag
sys.path.append(str(Path(__file__).parent.parent))

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.asyncio import tqdm

from habr_rag.config import (
    BATCH_SIZE,
    CHUNKS_FILE,
    COLLECTION_PREFIX,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MAX_CHUNKS_TO_UPLOAD,
    MAX_CONCURRENT_UPLOADS,
    MULTI_HOP_DATASET,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    SINGLE_HOP_DATASET,
)
from habr_rag.embeddings import CustomBGEEmbeddings


def load_chunks(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_dataset_article_ids():
    article_ids = set()
    chunk_ids = set()

    if os.path.exists(SINGLE_HOP_DATASET):
        with open(SINGLE_HOP_DATASET, "r", encoding="utf-8") as f:
            single_hop_data = json.load(f)
            for item in single_hop_data:
                if "article_id" in item:
                    article_ids.add(item["article_id"])
                if "chunk_id" in item:
                    chunk_ids.add(item["chunk_id"])
        print(f"Loaded {len(single_hop_data)} entries from single-hop dataset")
    else:
        print(f"Warning: {SINGLE_HOP_DATASET} not found")

    if os.path.exists(MULTI_HOP_DATASET):
        with open(MULTI_HOP_DATASET, "r", encoding="utf-8") as f:
            multi_hop_data = json.load(f)
            for item in multi_hop_data:
                if "source_article_ids" in item:
                    for aid in item["source_article_ids"]:
                        article_ids.add(aid)
                if "source_chunk_ids" in item:
                    for cid in item["source_chunk_ids"]:
                        chunk_ids.add(cid)
        print(f"Loaded {len(multi_hop_data)} entries from multi-hop dataset")
    else:
        print(f"Warning: {MULTI_HOP_DATASET} not found")

    article_ids = {str(aid) for aid in article_ids}

    print(f"Total unique article IDs from datasets: {len(article_ids)}")
    print(f"Total unique chunk IDs from datasets: {len(chunk_ids)}")

    return article_ids, chunk_ids


def prioritize_chunks(docs, dataset_article_ids):
    priority_docs = []
    remaining_docs = []

    for doc in docs:
        article_id = str(doc.metadata.get("id", ""))
        if article_id in dataset_article_ids:
            priority_docs.append(doc)
        else:
            remaining_docs.append(doc)

    return priority_docs, remaining_docs


async def upload_batch(vector_store, batch, semaphore, max_retries=5, initial_delay=1):
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
                print(
                    f"Error uploading batch (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2


async def main():
    embeddings = CustomBGEEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    embedding_size = 1024

    print(f"Loading chunks from {CHUNKS_FILE}...")
    try:
        all_docs = load_chunks(CHUNKS_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Loaded {len(all_docs)} documents total.")

    print("\nLoading dataset article IDs...")
    dataset_article_ids, dataset_chunk_ids = load_dataset_article_ids()

    priority_docs, remaining_docs = prioritize_chunks(all_docs, dataset_article_ids)
    print(f"\nPriority documents (from datasets): {len(priority_docs)}")
    print(f"Remaining documents: {len(remaining_docs)}")

    if MAX_CHUNKS_TO_UPLOAD is not None:
        if len(priority_docs) >= MAX_CHUNKS_TO_UPLOAD:
            docs = priority_docs[:MAX_CHUNKS_TO_UPLOAD]
            print(
                f"\nLimit reached with priority docs only. Using {len(docs)} priority docs."
            )
        else:
            remaining_slots = MAX_CHUNKS_TO_UPLOAD - len(priority_docs)
            docs = priority_docs + remaining_docs[:remaining_slots]
            print(
                f"\nUsing all {len(priority_docs)} priority docs + {min(remaining_slots, len(remaining_docs))} remaining docs."
            )
    else:
        docs = priority_docs + remaining_docs
        print(f"\nNo limit set. Using all {len(docs)} documents.")

    print(f"Total documents to upload: {len(docs)}")

    today = datetime.datetime.now().strftime("%Y_%m_%d")
    collection_name = f"{COLLECTION_PREFIX}_{today}"
    print(f"Target collection: {collection_name}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists. Deleting...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_size, distance=models.Distance.COSINE
        ),
    )
    print(f"Created collection {collection_name}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_collection_config=False,
    )

    print(f"Uploading documents in batches of {BATCH_SIZE}...")

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    tasks = []

    for batch in batches:
        task = upload_batch(vector_store, batch, semaphore)
        tasks.append(task)

    for f in tqdm.as_completed(tasks, total=len(batches), desc="Uploading batches"):
        await f

    print("Upload complete.")


if __name__ == "__main__":
    asyncio.run(main())
