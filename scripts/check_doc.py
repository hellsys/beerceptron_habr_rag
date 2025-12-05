import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models

from habr_rag.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_PREFIX

# Construct the collection name. Note: this might need adjustment if you want a specific date
# For now, let's assume we want to check today's collection or a hardcoded one if debugging.
# To make it robust, we could list collections and pick the latest, but for this script, manual edit is often expected.
# I will use a placeholder or the latest logic if possible.
# Since the original script had hardcoded date, I'll just put a comment.

import datetime
TODAY = datetime.datetime.now().strftime("%Y_%m_%d")
COLLECTION_NAME = f"{COLLECTION_PREFIX}_{TODAY}" 
# Override if checking a specific past collection:
# COLLECTION_NAME = "habr_rag_bge_m3_2025_12_04"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ID of the document to check
target_doc_id = "28359"

print(f"Checking for document ID: {target_doc_id} in collection: {COLLECTION_NAME}")

try:
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Index created/ensured.")
    except Exception as e:
        print(f"Index creation note: {e}")

    res = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=target_doc_id)
                )
            ]
        ),
        limit=5,
        with_payload=True
    )
    
    records, next_page_offset = res
    
    if records:
        print(f"\nSUCCESS: Found {len(records)} chunks for document {target_doc_id}.")
        print("Metadata sample from first chunk:")
        print(records[0].payload)
    else:
        print(f"\nFAILURE: Document {target_doc_id} NOT FOUND in collection.")

except Exception as e:
    print(f"Error during search: {e}")
