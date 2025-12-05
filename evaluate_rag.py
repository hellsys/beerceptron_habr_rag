import asyncio
import json
import os
import datetime
import urllib.request
import math
from typing import List, Set
# Try to import dotenv, but don't fail if missing (environment might be set otherwise)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
from tqdm.asyncio import tqdm
from langchain_core.embeddings import Embeddings

# --- Configuration ---
# Adjust these paths if necessary
SINGLE_HOP_DATASET = "dataset_single_hop.json"
MULTI_HOP_DATASET = "dataset_multi_hop.json"
OUTPUT_FILE = "evaluation_results_retrieval.json"
METRICS_FILE = "evaluation_metrics_retrieval.csv"

# Model for generation and evaluation
# MODEL_NAME = "gpt-5.1-2025-11-13" # Not used for retrieval only
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_BASE_URL = "https://gpt.mwsapis.ru/projects/mws-ai-automation/openai/v1"

# Qdrant Config
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://590d8b86-25a0-42c7-bf89-43a3fe0387ac.eu-central-1-0.aws.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Try to match the collection naming convention from upload_data.py
# If you want to target a specific collection, change this.
TODAY = datetime.datetime.now().strftime("%Y_%m_%d")
# You might need to adjust this if you are validating a collection from a different date
COLLECTION_NAME = f"habr_rag_bge_m3_{TODAY}"

# Retrieval Config
TOP_K = 10

# --- Helper Classes ---

class CustomBGEEmbeddings(Embeddings):
    """
    Custom embeddings class using `urllib` directly to avoid requests dependency issues.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "bge-m3"):
        self.api_key = api_key
        # Ensure base_url doesn't end with slash and append /embeddings
        self.endpoint_url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call the embeddings API using urllib."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        try:
            data_json = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url, data=data_json, headers=headers
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                response_body = response.read().decode("utf-8")
                data = json.loads(response_body)
            
            if "data" in data:
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                raise ValueError(f"Unexpected API response format: {data.keys()}")
                
        except Exception as e:
            print(f"\nEmbedding API Error: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# --- RAG Functions ---

async def retrieve_context(
    vector_store, query: str, top_k: int = 5, retries: int = 3, delay: float = 2.0
):
    """Retrieves documents from vector store with retry mechanism."""
    for attempt in range(retries):
        try:
            # returns list of (Document, score)
            results = await vector_store.asimilarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            print(f"Retrieval error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                import asyncio

                await asyncio.sleep(delay)
            else:
                raise

# --- Evaluation Metrics ---

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    unique_relevant_count = len(set(retrieved_k) & relevant_ids)
    return unique_relevant_count / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    unique_relevant_count = len(set(retrieved_k) & relevant_ids)
    return unique_relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    if not relevant_ids:
        return 0.0

    precisions = []
    relevant_count = 0
    seen_relevant = set()

    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids and doc_id not in seen_relevant:
            relevant_count += 1
            precisions.append(relevant_count / i)
            seen_relevant.add(doc_id)

    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_ids)


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    dcg = 0.0
    seen_relevant = set()
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids and doc_id not in seen_relevant:
            dcg += 1.0 / math.log2(i + 1)
            seen_relevant.add(doc_id)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    retrieved_k = retrieved_ids[:k]
    return 1 if any(doc_id in relevant_ids for doc_id in retrieved_k) else 0


def calculate_retrieval_metrics(retrieved_docs, target_chunk_ids, target_doc_ids=None):
    """
    Calculates comprehensive retrieval metrics for both chunks and documents.
    target_chunk_ids: list of correct chunk IDs.
    target_doc_ids: list of correct document/article IDs (optional).
    """
    retrieved_chunk_ids = []
    retrieved_doc_ids = []

    for doc, _ in retrieved_docs:
        chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("chunk_uuid")
        if chunk_id:
            retrieved_chunk_ids.append(str(chunk_id))

        article_id = (
            doc.metadata.get("id")
            or doc.metadata.get("article_id")
            or doc.metadata.get("article_uuid")
        )
        if article_id:
            retrieved_doc_ids.append(str(article_id))

    target_chunk_set = {str(tid) for tid in (target_chunk_ids or []) if tid}
    target_doc_set = {str(tid) for tid in (target_doc_ids or []) if tid}

    k_values = [1, 3, 5, 10]

    metrics = {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
    }

    if target_chunk_ids:
        for k in k_values:
            metrics[f"hit@{k}_chunk"] = hit_at_k(retrieved_chunk_ids, target_chunk_set, k)
            metrics[f"precision@{k}_chunk"] = precision_at_k(
                retrieved_chunk_ids, target_chunk_set, k
            )
            metrics[f"recall@{k}_chunk"] = recall_at_k(
                retrieved_chunk_ids, target_chunk_set, k
            )
            metrics[f"ndcg@{k}_chunk"] = ndcg_at_k(retrieved_chunk_ids, target_chunk_set, k)

        metrics["mrr_chunk"] = reciprocal_rank(retrieved_chunk_ids, target_chunk_set)
        metrics["map_chunk"] = average_precision(retrieved_chunk_ids, target_chunk_set)
    else:
        for k in k_values:
            metrics[f"hit@{k}_chunk"] = 0
            metrics[f"precision@{k}_chunk"] = 0.0
            metrics[f"recall@{k}_chunk"] = 0.0
            metrics[f"ndcg@{k}_chunk"] = 0.0
        metrics["mrr_chunk"] = 0.0
        metrics["map_chunk"] = 0.0

    if target_doc_ids:
        for k in k_values:
            metrics[f"hit@{k}_doc"] = hit_at_k(retrieved_doc_ids, target_doc_set, k)
            metrics[f"precision@{k}_doc"] = precision_at_k(
                retrieved_doc_ids, target_doc_set, k
            )
            metrics[f"recall@{k}_doc"] = recall_at_k(retrieved_doc_ids, target_doc_set, k)
            metrics[f"ndcg@{k}_doc"] = ndcg_at_k(retrieved_doc_ids, target_doc_set, k)

        metrics["mrr_doc"] = reciprocal_rank(retrieved_doc_ids, target_doc_set)
        metrics["map_doc"] = average_precision(retrieved_doc_ids, target_doc_set)
    else:
        for k in k_values:
            metrics[f"hit@{k}_doc"] = 0
            metrics[f"precision@{k}_doc"] = 0.0
            metrics[f"recall@{k}_doc"] = 0.0
            metrics[f"ndcg@{k}_doc"] = 0.0
        metrics["mrr_doc"] = 0.0
        metrics["map_doc"] = 0.0

    metrics["hit_rate_chunk"] = metrics["hit@10_chunk"]
    metrics["hit_rate_doc"] = metrics["hit@10_doc"]

    return metrics

# --- Main Logic ---

async def process_item(item, vector_store, dataset_type="single_hop"):
    question = item["question"]
    # ground_truth = item["answer"] # Not needed for retrieval eval
    
    # Identify target chunks and docs
    if dataset_type == "single_hop":
        target_chunk_ids = [item.get("chunk_id")]
        target_doc_ids = [item.get("article_id")]
    else:
        target_chunk_ids = item.get("source_chunk_ids", [])
        target_doc_ids = item.get("source_article_ids", [])
    
    # 1. Retrieve
    retrieved_docs = await retrieve_context(vector_store, question, top_k=TOP_K)
    
    # DEBUG PRINT
    # print(f"DEBUG: Question: {question}")
    # if retrieved_docs:
    #     print(f"DEBUG: First retrieved doc metadata: {retrieved_docs[0][0].metadata}")
    #     print(f"DEBUG: Target Chunk IDs: {target_chunk_ids}")
    #     print(f"DEBUG: Target Doc IDs: {target_doc_ids}")
    
    # 2. Retrieval Metrics
    retrieval_metrics = calculate_retrieval_metrics(
        retrieved_docs, target_chunk_ids, target_doc_ids
    )
    
    # DEBUG: Print if no hit
    if retrieval_metrics.get("hit_rate_doc", 0) == 0 and dataset_type == "single_hop":
         # Only print once or rarely to avoid spam
         import random
         if random.random() < 0.01:
             print(f"\nDEBUG MISS:")
             print(f"Question: {question}")
             print(f"Target Chunk: {target_chunk_ids}")
             print(f"Target Doc: {target_doc_ids}")
             print("Top 3 Retrieved Metadata:")
             for i, (doc, score) in enumerate(retrieved_docs[:3]):
                 print(f"  {i+1}. Score: {score:.4f}, Meta: {doc.metadata}")
    
    result = {
        "dataset_type": dataset_type,
        "question": question,
        "target_chunk_ids": target_chunk_ids,
        "target_doc_ids": target_doc_ids,
    }
    result.update(retrieval_metrics)
    
    return result

async def main():
    print("Initializing components...")

    embeddings = CustomBGEEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Check if collection exists
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
        print("Existing collections:")
        collections = qdrant_client.get_collections().collections
        for c in collections:
            print(f" - {c.name}")
        print(
            "Please update COLLECTION_NAME in the script or run upload_data.py first."
        )
        return
    else:
        print(f"Using collection: {COLLECTION_NAME}")
        
        # Create payload index for 'id' to fix filtering errors
        print("Ensuring payload index for 'id'...")
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Index check complete.")

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        validate_collection_config=False,
    )

    all_results = []

    # --- Process Single Hop ---
    if os.path.exists(SINGLE_HOP_DATASET):
        print(f"\nProcessing Single Hop Dataset ({SINGLE_HOP_DATASET})...")
        with open(SINGLE_HOP_DATASET, "r") as f:
            single_hop_data = json.load(f)

        # Limit for testing? (Remove [:10] for full run)
        # single_hop_data = single_hop_data[:10]

        tasks = [
            process_item(item, vector_store, "single_hop") for item in single_hop_data
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Single Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {SINGLE_HOP_DATASET} not found.")

    # --- Process Multi Hop ---
    if os.path.exists(MULTI_HOP_DATASET):
        print(f"\nProcessing Multi Hop Dataset ({MULTI_HOP_DATASET})...")
        with open(MULTI_HOP_DATASET, "r") as f:
            multi_hop_data = json.load(f)

        # multi_hop_data = multi_hop_data[:10]

        tasks = [
            process_item(item, vector_store, "multi_hop") for item in multi_hop_data
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Multi Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {MULTI_HOP_DATASET} not found.")

    # --- Save and Aggregate ---
    if not all_results:
        print("No results generated.")
        return

    # Save raw results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

    # Calculate Aggregates
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("                    RETRIEVAL METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Questions: {len(df)}")
    print(f"TOP_K: {TOP_K}")

    metric_columns_chunk = []
    metric_columns_doc = []

    for k in [1, 3, 5, 10]:
        for metric in ["hit", "precision", "recall", "ndcg"]:
            col_chunk = f"{metric}@{k}_chunk"
            col_doc = f"{metric}@{k}_doc"
            if col_chunk in df.columns:
                metric_columns_chunk.append(col_chunk)
            if col_doc in df.columns:
                metric_columns_doc.append(col_doc)

    for metric in ["mrr", "map"]:
        if f"{metric}_chunk" in df.columns:
            metric_columns_chunk.append(f"{metric}_chunk")
        if f"{metric}_doc" in df.columns:
            metric_columns_doc.append(f"{metric}_doc")

    print("\n" + "-" * 70)
    print("CHUNK-LEVEL METRICS (exact chunk match)")
    print("-" * 70)

    print(f"\n{'Metric':<20} {'Overall':>10}", end="")
    if "dataset_type" in df.columns:
        for dt in sorted(df["dataset_type"].unique()):
            print(f" {dt:>12}", end="")
    print()
    print("-" * 70)

    for col in metric_columns_chunk:
        metric_name = col.replace("_chunk", "")
        overall = df[col].mean()
        print(f"{metric_name:<20} {overall:>10.4f}", end="")
        if "dataset_type" in df.columns:
            for dt in sorted(df["dataset_type"].unique()):
                val = df[df["dataset_type"] == dt][col].mean()
                print(f" {val:>12.4f}", end="")
        print()

    print("\n" + "-" * 70)
    print("DOCUMENT-LEVEL METRICS (article match)")
    print("-" * 70)

    print(f"\n{'Metric':<20} {'Overall':>10}", end="")
    if "dataset_type" in df.columns:
        for dt in sorted(df["dataset_type"].unique()):
            print(f" {dt:>12}", end="")
    print()
    print("-" * 70)

    for col in metric_columns_doc:
        metric_name = col.replace("_doc", "")
        overall = df[col].mean()
        print(f"{metric_name:<20} {overall:>10.4f}", end="")
        if "dataset_type" in df.columns:
            for dt in sorted(df["dataset_type"].unique()):
                val = df[df["dataset_type"] == dt][col].mean()
                print(f" {val:>12.4f}", end="")
        print()

    print("\n" + "=" * 70)

    if "dataset_type" in df.columns:
        all_metric_cols = metric_columns_chunk + metric_columns_doc
        summary = df.groupby("dataset_type")[all_metric_cols].mean()

        summary_transposed = summary.T
        summary_transposed["overall"] = df[all_metric_cols].mean()

        summary_transposed.to_csv(METRICS_FILE)
        print(f"\nMetrics summary saved to {METRICS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
