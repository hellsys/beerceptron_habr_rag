import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
from tqdm.asyncio import tqdm

from habr_rag.config import (
    COLLECTION_PREFIX,
    EVAL_METRICS_FILE,
    EVAL_RESULTS_FILE,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MULTI_HOP_DATASET,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    SINGLE_HOP_DATASET,
)
from habr_rag.embeddings import CustomBGEEmbeddings

# Retrieval Config
TOP_K = 10

# Helper Functions
async def retrieve_context(
    vector_store, query: str, top_k: int = 5, retries: int = 3, delay: float = 2.0
):
    """Retrieves documents from vector store with retry mechanism."""
    for attempt in range(retries):
        try:
            results = await vector_store.asimilarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            print(f"Retrieval error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise

def calculate_retrieval_metrics(retrieved_docs, target_chunk_ids, target_doc_ids=None):
    retrieved_chunk_ids = [doc.metadata.get("chunk_id") for doc, _ in retrieved_docs]
    retrieved_doc_ids = [str(doc.metadata.get("id")) for doc, _ in retrieved_docs]

    chunk_hits = any(tid in retrieved_chunk_ids for tid in target_chunk_ids)

    doc_hits = False
    if target_doc_ids:
        target_doc_ids_str = [str(tid) for tid in target_doc_ids]
        doc_hits = any(tid in retrieved_doc_ids for tid in target_doc_ids_str)

    return {
        "hit_rate_chunk": 1 if chunk_hits else 0,
        "hit_rate_doc": 1 if doc_hits else 0,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
    }

async def process_item(item, vector_store, dataset_type="single_hop"):
    question = item["question"]
    
    if dataset_type == "single_hop":
        target_chunk_ids = [item.get("chunk_id")]
        target_doc_ids = [item.get("article_id")]
    else:
        target_chunk_ids = item.get("source_chunk_ids", [])
        target_doc_ids = item.get("source_article_ids", [])
    
    retrieved_docs = await retrieve_context(vector_store, question, top_k=TOP_K)
    
    retrieval_metrics = calculate_retrieval_metrics(
        retrieved_docs, target_chunk_ids, target_doc_ids
    )
    
    return {
        "dataset_type": dataset_type,
        "question": question,
        "target_chunk_ids": target_chunk_ids,
        "target_doc_ids": target_doc_ids,
        "retrieved_chunk_ids": retrieval_metrics["retrieved_chunk_ids"],
        "retrieved_doc_ids": retrieval_metrics["retrieved_doc_ids"],
        "hit_rate_chunk": retrieval_metrics["hit_rate_chunk"],
        "hit_rate_doc": retrieval_metrics["hit_rate_doc"],
    }

async def main():
    print("Initializing components...")

    embeddings = CustomBGEEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    today = datetime.datetime.now().strftime("%Y_%m_%d")
    collection_name = f"{COLLECTION_PREFIX}_{today}"

    if not qdrant_client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist.")
        print("Existing collections:")
        collections = qdrant_client.get_collections().collections
        for c in collections:
            print(f" - {c.name}")
        print("Please ensure upload_data.py has been run.")
        return
    else:
        print(f"Using collection: {collection_name}")
        
        print("Ensuring payload index for 'id'...")
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_collection_config=False,
    )

    all_results = []

    if os.path.exists(SINGLE_HOP_DATASET):
        print(f"\nProcessing Single Hop Dataset ({SINGLE_HOP_DATASET})...")
        with open(SINGLE_HOP_DATASET, "r") as f:
            single_hop_data = json.load(f)

        tasks = [
            process_item(item, vector_store, "single_hop") for item in single_hop_data
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Single Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {SINGLE_HOP_DATASET} not found.")

    if os.path.exists(MULTI_HOP_DATASET):
        print(f"\nProcessing Multi Hop Dataset ({MULTI_HOP_DATASET})...")
        with open(MULTI_HOP_DATASET, "r") as f:
            multi_hop_data = json.load(f)

        tasks = [
            process_item(item, vector_store, "multi_hop") for item in multi_hop_data
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Multi Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {MULTI_HOP_DATASET} not found.")

    if not all_results:
        print("No results generated.")
        return

    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {EVAL_RESULTS_FILE}")

    df = pd.DataFrame(all_results)

    print("\n=== Retrieval Metrics Summary ===")

    print(f"Total Questions: {len(df)}")
    print(f"Overall Hit Rate (Chunk): {df['hit_rate_chunk'].mean():.4f}")
    print(f"Overall Hit Rate (Doc):   {df['hit_rate_doc'].mean():.4f}")

    if "dataset_type" in df.columns:
        summary = df.groupby("dataset_type")[["hit_rate_chunk", "hit_rate_doc"]].mean()
        print("\nBy Dataset Type:")
        print(summary)

        summary.to_csv(EVAL_METRICS_FILE)
        print(f"Metrics summary saved to {EVAL_METRICS_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
