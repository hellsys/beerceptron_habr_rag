from qdrant_client import QdrantClient
from app.core import config
from app.services.llm.provider import get_embeddings
import logging

log = logging.getLogger("app")

client = QdrantClient(
    url=config.QDRANT_API_URL,
    api_key=config.QDRANT_API_KEY
)

def get_chunks(text: str, collection_name: str, top_k: int = 5):
    """
    Возвращает похожие чанки из Qdrant
    """
    global log
    log.info(f"Querying Qdrant collection '{collection_name}' for top {top_k} chunks similar to the input text.")
    # log.info(f"Qdrant input text: {text}")

    try:
        emb = get_embeddings(text)
        log.info(f"Generate embedding for query text: {text}")

        chunks = client.query_points(
            collection_name=collection_name,
            query=emb,
            with_payload=True,
            limit=top_k
        ).points
        log.info(f"Retrieved {len(chunks)} chunks from Qdrant.")
        log.info(f"Chunks: {chunks}")
        search_result = [
            {
                'content': chunk.payload['metadata']['initial_text'],
                'metadata': {
                    'url': chunk.payload['metadata']['url'],
                    'title': chunk.payload['metadata']['title']
                }
            } for chunk in chunks
        ]
    #     search_result = [
    #         {
    #             'content': """Model Context Protocol (MCP)
    # Что это: Открытый протокол, разработанный компанией Anthropic, который стандартизирует способ, которым приложения на базе ИИ взаимодействуют с внешними системами, такими как CRM, почта, или базами данных.
    # Для чего: Позволяет ИИ-агентам не только отвечать на запросы, но и выполнять реальные действия во внешних системах, что делает их более полезными и автономными.
    # Принцип работы: Использует стандартный формат сообщений, аналогичный client-server архитектуре, для обмена данными между ИИ-моделями и другими инструментами. """,
    #             'metadata': {
    #                 'url': 'https://modelcontextprotocol.io/docs/getting-started/intro',
    #                 'title': 'What is the Model Context Protocol'
    #             }
    #         },
    #         {
    #             'content': """Sample chunk 2 content related to the query.""",
    #             'metadata': {
    #                 'url': 'http://example.com/article2',
    #                 'title': 'Example Article 2'
    #             }
    #         }
    #     ] # Заглушка для примера
        log.info(f"Returning {len(search_result)} chunks from Qdrant search.")
        log.debug(f"Chunks: {search_result}")
        return search_result
    except Exception as e:
        log = logging.getLogger("errors")
        log.error(f"Error querying Qdrant: {e}")
        raise