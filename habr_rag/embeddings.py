import json
from typing import List
import requests
from langchain_core.embeddings import Embeddings

class CustomBGEEmbeddings(Embeddings):
    """
    Custom embeddings class for BGE-M3 model.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "bge-m3"):
        self.api_key = api_key
        self.endpoint_url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            else:
                raise ValueError(f"Unexpected API response format: {data.keys()}")
                
        except requests.exceptions.RequestException as e:
            print(f"Embedding API Error: {e}")
            if hasattr(e.response, "text"):
                print(f"Response text: {e.response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

