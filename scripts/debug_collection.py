"""
Скрипт для отладки - проверяет, что реально находится в коллекции Qdrant
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from habr_rag.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_PREFIX

load_dotenv()

# Имя коллекции для проверки
collection_name = f"{COLLECTION_PREFIX}_test_2025_12_05"  # Измените на нужную

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if not client.collection_exists(collection_name):
    print(f"Collection {collection_name} does not exist!")
    print("Available collections:")
    for c in client.get_collections().collections:
        print(f"  - {c.name}")
    sys.exit(1)

print(f"Checking collection: {collection_name}")
print("=" * 60)

# Получаем несколько случайных точек
results = client.scroll(
    collection_name=collection_name,
    limit=5,
    with_payload=True,
    with_vectors=False
)

points, _ = results

if not points:
    print("Collection is empty!")
    sys.exit(1)

print(f"\nFound {len(points)} sample points\n")

for i, point in enumerate(points, 1):
    print(f"--- Point {i} (ID: {point.id}) ---")
    print(f"Payload keys: {list(point.payload.keys())}")
    print(f"Payload:")
    for key, value in point.payload.items():
        if isinstance(value, (list, dict)):
            print(f"  {key}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
            if key == 'metadata' and isinstance(value, dict):
                print(f"    Metadata keys: {list(value.keys())}")
                print(f"    Full metadata: {value}")
        else:
            print(f"  {key}: {value}")
    print()

# Проверяем конкретный chunk_id из датасета
print("\n" + "=" * 60)
print("Checking specific chunk_id from dataset...")

# Загружаем тестовый датасет
from habr_rag.config import TEST_SINGLE_HOP_DATASET
import json

if os.path.exists(TEST_SINGLE_HOP_DATASET):
    with open(TEST_SINGLE_HOP_DATASET, "r") as f:
        test_data = json.load(f)
    
    if test_data:
        sample_item = test_data[0]
        target_chunk_id = sample_item.get("chunk_id")
        target_article_id = sample_item.get("article_id")
        
        print(f"Looking for chunk_id: {target_chunk_id}")
        print(f"Looking for article_id: {target_article_id}")
        
        # Ищем по chunk_id
        search_results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchValue(value=target_chunk_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        found_points, _ = search_results
        if found_points:
            print(f"\n✓ Found chunk by chunk_id!")
            print(f"Payload: {found_points[0].payload}")
            if 'metadata' in found_points[0].payload:
                print(f"Metadata: {found_points[0].payload['metadata']}")
        else:
            print(f"\n✗ chunk_id '{target_chunk_id}' NOT FOUND")
            # Проверим, какие chunk_id есть в коллекции
            print("\nChecking what chunk_ids exist in collection...")
            sample_check = client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True
            )
            sample_points, _ = sample_check
            print(f"Sample of {len(sample_points)} points:")
            for p in sample_points[:3]:
                if 'metadata' in p.payload:
                    meta = p.payload['metadata']
                    print(f"  Point ID: {p.id}")
                    print(f"    chunk_id: {meta.get('chunk_id', 'NOT FOUND')}")
                    print(f"    article_id (id): {meta.get('id', 'NOT FOUND')}")
        
        # Ищем по article_id
        search_results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=str(target_article_id))
                    )
                ]
            ),
            limit=3,
            with_payload=True
        )
        
        found_points, _ = search_results
        if found_points:
            print(f"\n✓ Found {len(found_points)} chunks by article_id!")
            print(f"First chunk payload keys: {list(found_points[0].payload.keys())}")
            if "chunk_id" in found_points[0].payload:
                print(f"First chunk chunk_id: {found_points[0].payload['chunk_id']}")
        else:
            print(f"\n✗ article_id '{target_article_id}' NOT FOUND")

