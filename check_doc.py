import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "https://590d8b86-25a0-42c7-bf89-43a3fe0387ac.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# WARNING: Check if this collection name matches what is used in upload_data.py
# Usually it has a date suffix, e.g., habr_rag_bge_m3_2025_12_04
COLLECTION_NAME = "habr_rag_bge_m3_2025_12_04" 

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ID of the document that was missed in retrieval (from your logs)
target_doc_id = "28359"  # Ensure string if IDs are stored as strings

print(f"Checking for document ID: {target_doc_id} in collection: {COLLECTION_NAME}")

try:
    # First, ensure index exists (just in case, though evaluate_rag.py now does this)
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Index created/ensured.")
    except Exception as e:
        print(f"Index creation note: {e}")

    # Search for the document using scroll with filter
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
        print("Possible reasons:")
        print("1. The document was filtered out during upload (e.g., due to MAX_CHUNKS_TO_UPLOAD limit).")
        print("2. The 'id' field type mismatch (string vs int).")

except Exception as e:
    print(f"Error during search: {e}")
