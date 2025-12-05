import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Set

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from tqdm.asyncio import tqdm

from habr_rag.config import (
    COLLECTION_PREFIX,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EVAL_METRICS_FILE,
    EVAL_RESULTS_FILE,
    MULTI_HOP_DATASET,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    SINGLE_HOP_DATASET,
    TEST_MULTI_HOP_DATASET,
    TEST_SINGLE_HOP_DATASET,
)
from habr_rag.embeddings import CustomBGEEmbeddings

TOP_K = 10
MAX_CONCURRENT_REQUESTS = 5
SEARCH_EF = 64

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


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


async def retrieve_context(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    embeddings,
    query: str,
    top_k: int = 5,
    retries: int = 3,
    delay: float = 2.0,
):
    async with semaphore:
        for attempt in range(retries):
            try:
                query_vector = await asyncio.to_thread(embeddings.embed_query, query)

                results = await qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                    search_params=models.SearchParams(
                        hnsw_ef=SEARCH_EF,
                        exact=False,
                    ),
                    timeout=60,
                )

                converted_results = []
                for point in results.points:

                    class DocResult:
                        def __init__(self, metadata):
                            self.metadata = metadata

                    payload = point.payload or {}
                    if "metadata" in payload:
                        metadata = payload["metadata"]
                    else:
                        metadata = payload

                    doc = DocResult(metadata)
                    score = point.score
                    converted_results.append((doc, score))

                return converted_results
            except Exception as e:
                # print(f"Retrieval error on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (attempt + 1))
                else:
                    raise


def calculate_retrieval_metrics(
    retrieved_docs, target_chunk_ids, target_doc_ids=None, debug=False
):
    retrieved_chunk_ids = []
    retrieved_doc_ids = []

    for doc, _ in retrieved_docs:
        chunk_id = (
            doc.metadata.get("chunk_id") or doc.metadata.get("chunk_uuid") or None
        )
        if chunk_id:
            retrieved_chunk_ids.append(str(chunk_id))

        article_id = (
            doc.metadata.get("id")
            or doc.metadata.get("article_id")
            or doc.metadata.get("article_uuid")
            or None
        )
        if article_id:
            retrieved_doc_ids.append(str(article_id))

    if debug and retrieved_docs:
        print("\n=== DEBUG: First retrieved document ===")
        print(f"Metadata keys: {list(retrieved_docs[0][0].metadata.keys())}")
        print(f"Full metadata: {retrieved_docs[0][0].metadata}")
        print(f"\nTarget chunk_ids: {target_chunk_ids}")
        print(f"Retrieved chunk_ids (first 5): {retrieved_chunk_ids[:5]}")
        print(f"\nTarget doc_ids: {target_doc_ids}")
        print(f"Retrieved doc_ids (first 5): {retrieved_doc_ids[:5]}")

    target_chunk_set = {str(tid) for tid in target_chunk_ids if tid}
    target_doc_set = {str(tid) for tid in (target_doc_ids or []) if tid}

    k_values = [1, 3, 5, 10]

    metrics = {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
    }

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

    for k in k_values:
        metrics[f"hit@{k}_doc"] = hit_at_k(retrieved_doc_ids, target_doc_set, k)
        metrics[f"precision@{k}_doc"] = precision_at_k(
            retrieved_doc_ids, target_doc_set, k
        )
        metrics[f"recall@{k}_doc"] = recall_at_k(retrieved_doc_ids, target_doc_set, k)
        metrics[f"ndcg@{k}_doc"] = ndcg_at_k(retrieved_doc_ids, target_doc_set, k)

    metrics["mrr_doc"] = reciprocal_rank(retrieved_doc_ids, target_doc_set)
    metrics["map_doc"] = average_precision(retrieved_doc_ids, target_doc_set)

    metrics["hit_rate_chunk"] = metrics["hit@10_chunk"]
    metrics["hit_rate_doc"] = metrics["hit@10_doc"]

    return metrics


async def process_item(
    item,
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    embeddings,
    dataset_type="single_hop",
    debug=False,
):
    question = item["question"]

    if dataset_type == "single_hop":
        target_chunk_ids = [item.get("chunk_id")]
        target_doc_ids = [item.get("article_id")]
    else:
        target_chunk_ids = item.get("source_chunk_ids", [])
        target_doc_ids = item.get("source_article_ids", [])

    try:
        retrieved_docs = await retrieve_context(
            qdrant_client, collection_name, embeddings, question, top_k=TOP_K
        )

        retrieval_metrics = calculate_retrieval_metrics(
            retrieved_docs, target_chunk_ids, target_doc_ids, debug=debug
        )

        result = {
            "dataset_type": dataset_type,
            "question": question,
            "target_chunk_ids": target_chunk_ids,
            "target_doc_ids": target_doc_ids,
        }
        result.update(retrieval_metrics)

        return result

    except Exception as e:
        print(f"Error processing question '{question[:50]}...': {e}")
        error_result = {
            "dataset_type": dataset_type,
            "question": question,
            "target_chunk_ids": target_chunk_ids,
            "target_doc_ids": target_doc_ids,
            "retrieved_chunk_ids": [],
            "retrieved_doc_ids": [],
            "error": str(e),
        }
        for k in [1, 3, 5, 10]:
            for suffix in ["chunk", "doc"]:
                error_result[f"hit@{k}_{suffix}"] = 0
                error_result[f"precision@{k}_{suffix}"] = 0.0
                error_result[f"recall@{k}_{suffix}"] = 0.0
                error_result[f"ndcg@{k}_{suffix}"] = 0.0
        for suffix in ["chunk", "doc"]:
            error_result[f"mrr_{suffix}"] = 0.0
            error_result[f"map_{suffix}"] = 0.0
            error_result[f"hit_rate_{suffix}"] = 0

        return error_result


async def main():
    debug = False
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test collection and test datasets (smaller, faster)",
    )
    parser.add_argument(
        "--collection", type=str, default=None, help="Override collection name"
    )
    args = parser.parse_args()

    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    if args.test:
        single_hop_path = TEST_SINGLE_HOP_DATASET
        multi_hop_path = TEST_MULTI_HOP_DATASET
        default_collection = "habr_rag_bge_m3_test_2025_12_05"
        print("=" * 60)
        print("RUNNING IN TEST MODE (smaller dataset)")
        print("=" * 60)
    else:
        single_hop_path = SINGLE_HOP_DATASET
        multi_hop_path = MULTI_HOP_DATASET
        default_collection = f"{COLLECTION_PREFIX}_2025_12_04"

    collection_name = args.collection if args.collection else default_collection

    print("Initializing components...")
    print(
        f"Config: TOP_K={TOP_K}, MAX_CONCURRENT={MAX_CONCURRENT_REQUESTS}, HNSW_EF={SEARCH_EF}"
    )

    embeddings = CustomBGEEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    print(f"QDRANT_URL: {QDRANT_URL}")

    sync_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    async_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    if not sync_client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist.")
        print("Existing collections:")
        collections = sync_client.get_collections().collections
        for c in collections:
            print(f" - {c.name}")
        if args.test:
            print(
                "\nTo create test collection, run: python scripts/create_test_collection.py"
            )
        else:
            print("Please ensure upload_data.py has been run.")
        return
    else:
        collection_info = sync_client.get_collection(collection_name)
        print(f"Using collection: {collection_name}")
        print(f"  Points count: {collection_info.points_count}")
        print(f"  Indexed vectors: {collection_info.indexed_vectors_count}")

    all_results = []

    if os.path.exists(single_hop_path):
        print(f"\nProcessing Single Hop Dataset ({single_hop_path})...")
        with open(single_hop_path, "r") as f:
            single_hop_data = json.load(f)

        print(f"Total items: {len(single_hop_data)}")

        tasks = []
        for idx, item in enumerate(single_hop_data):
            tasks.append(
                process_item(
                    item,
                    async_client,
                    collection_name,
                    embeddings,
                    "single_hop",
                    debug=debug,
                )
            )

        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Single Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {single_hop_path} not found.")

    if os.path.exists(multi_hop_path):
        print(f"\nProcessing Multi Hop Dataset ({multi_hop_path})...")
        with open(multi_hop_path, "r") as f:
            multi_hop_data = json.load(f)

        print(f"Total items: {len(multi_hop_data)}")

        tasks = [
            process_item(item, async_client, collection_name, embeddings, "multi_hop")
            for item in multi_hop_data
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating Multi Hop"
        ):
            all_results.append(await task)
    else:
        print(f"Dataset {multi_hop_path} not found.")

    await async_client.close()

    if not all_results:
        print("No results generated.")
        return

    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {EVAL_RESULTS_FILE}")

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

        summary_transposed.to_csv(EVAL_METRICS_FILE)
        print(f"\nMetrics summary saved to {EVAL_METRICS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
