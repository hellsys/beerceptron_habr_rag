import argparse
import asyncio
import datetime
import json
import os
import pickle
from typing import List

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.asyncio import tqdm

load_dotenv()


class CustomBGEEmbeddings(Embeddings):
    """
    Custom embeddings class using `requests` directly.
    Replicates the working curl/requests example exactly, bypassing openai library issues.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "bge-m3"):
        self.api_key = api_key

        self.endpoint_url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call the embeddings API using requests."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(
                self.endpoint_url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                raise ValueError(f"Unexpected API response format: {data.keys()}")

        except requests.exceptions.HTTPError as e:
            print(f"\nAPI Error: {e}")
            print(f"Response text: {response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed([text])[0]


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
CHUNKS_FILE = os.path.join(script_dir, "data/processed/chunked_docs.pkl")
SINGLE_HOP_DATASET = os.path.join(parent_dir, "data/datasets/dataset_single_hop.json")
MULTI_HOP_DATASET = os.path.join(parent_dir, "data/datasets/dataset_multi_hop.json")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_PREFIX = "habr_rag_bge_m3"
BATCH_SIZE = 50
MAX_CONCURRENT_UPLOADS = 10

print(f"QDRANT_URL: {QDRANT_URL}")

MAX_CHUNKS_TO_UPLOAD = 5500000


def load_chunks(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_dataset_article_ids():
    """Load article IDs and chunk IDs from synthetic datasets."""
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
    """
    Split documents into priority (from datasets) and remaining.
    Returns (priority_docs, remaining_docs)
    """
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
    parser = argparse.ArgumentParser(description="Upload data to Qdrant")
    parser.add_argument(
        "--resume-from-batch",
        type=int,
        default=0,
        help="Batch index to resume from (0 to start over)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Target collection name (defaults to habr_rag_bge_m3_YYYY_MM_DD)",
    )
    args = parser.parse_args()

    # Initialize embeddings
    embeddings = CustomBGEEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://gpt.mwsapis.ru/projects/mws-ai-automation/openai/v1",
        model="bge-m3",
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
    if args.collection_name:
        collection_name = args.collection_name
    else:
        collection_name = f"{COLLECTION_PREFIX}_{today}"
    print(f"Target collection: {collection_name}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if args.resume_from_batch == 0:
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
    else:
        print(f"Resuming from batch {args.resume_from_batch}...")
        if not client.collection_exists(collection_name):
            print(f"Error: Collection {collection_name} does not exist. Cannot resume.")
            return

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_collection_config=False,
    )

    print(f"Uploading documents in batches of {BATCH_SIZE}...")

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]

    if args.resume_from_batch > 0:
        if args.resume_from_batch >= len(batches):
            print(
                f"Resume batch index {args.resume_from_batch} is >= total batches {len(batches)}. Nothing to upload."
            )
            return
        print(
            f"Skipping first {args.resume_from_batch} batches. {len(batches) - args.resume_from_batch} remaining."
        )
        batches = batches[args.resume_from_batch :]

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
