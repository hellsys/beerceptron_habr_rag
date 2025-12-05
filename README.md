# Habr RAG

RAG-система для поиска и генерации ответов на основе статей с Хабра.

## Источник данных

Используется публичный датасет [IlyaGusev/habr](https://huggingface.co/datasets/IlyaGusev/habr) с HuggingFace — коллекция статей с habr.com.

Загрузка:
```python
from datasets import load_dataset
dataset = load_dataset('IlyaGusev/habr', split="train")
```

## Структура проекта

```
data/
├── raw/                      # Метаданные статей (meta_cols.csv)
├── processed/                # Чанки (chunked_docs.pkl)
├── datasets/                 # Синтетические датасеты для оценки
└── results/                  # Результаты evaluation

habr_rag/
├── config.py                 # Конфигурация (пути, API endpoints)
├── embeddings.py             # Класс CustomBGEEmbeddings
└── splitter.py               # MarkdownSplitter для chunking

scripts/
├── generate_synthetic_dataset.py   # Генерация QA-датасетов
├── evaluate_rag.py                 # Оценка retrieval
└── upload_data.py                  # Загрузка в Qdrant

notebooks/
└── collection_creation.ipynb       # Создание чанков из исходных данных
```

## Chunking

Документы разбиваются на чанки с помощью `MarkdownSplitter` (`habr_rag/splitter.py`).

### Параметры

| Параметр | Значение |
|----------|----------|
| chunk_size | 500 |
| chunk_overlap | 50 |
| max_symbols | 1500 |

### Логика работы

1. **Препроцессинг**:
   - Удаление Markdown-ссылок (опционально)
   - Обработка code-блоков: JSON упрощается (оставляется только первый элемент массивов)
   - Таблицы: либо удаляются, либо заменяются placeholder'ами и разбиваются отдельно

2. **Разбиение**: используется `ExperimentalMarkdownSyntaxTextSplitter` из LangChain, который учитывает структуру Markdown (заголовки как разделители)

3. **Постпроцессинг**:
   - Если чанк превышает `max_symbols`, дополнительно разбивается через `RecursiveCharacterTextSplitter`
   - К каждому чанку добавляется header с метаданными статьи (title, author, hubs, flows, tags, labels)
   - Генерируется уникальный `chunk_id` (UUID)
   - Дедупликация по содержимому

### Метаданные чанка

```python
{
    "id": 123456,              # ID статьи
    "chunk_id": "uuid-...",    # UUID чанка
    "title": "...",
    "author": "...",
    "hubs": "[...]",
    "flows": "[...]",
    "tags": "[...]",
    "initial_text": "..."      # Текст до финального разбиения
}
```

## Синтетические датасеты

Для оценки retrieval генерируются два типа датасетов (`scripts/generate_synthetic_dataset.py`):

### Single-hop (800 примеров)

Вопрос по одному чанку одной статьи. LLM получает все чанки статьи и формулирует вопрос к конкретному факту.

```json
{
  "question": "...",
  "answer": "...",
  "chunk_id": "target-chunk-uuid",
  "article_id": 123456
}
```

### Multi-hop (800 примеров)

Вопрос, требующий информации из нескольких статей. Статьи группируются по общим тегам, LLM синтезирует вопрос, требующий объединения информации.

```json
{
  "question": "...",
  "answer": "...",
  "reasoning": "почему нужно несколько источников",
  "source_chunk_ids": ["uuid-1", "uuid-2"],
  "source_article_ids": [123, 456]
}
```

## Векторное хранилище

- **БД**: Qdrant (Яндекс Облако)
- **Эмбеддинги**: BGE-M3 (1024 dim)
- **Коллекция**: `habr_rag_bge_m3_{YYYY_MM_DD}`

Загрузка выполняется батчами по 50 документов с retry-механизмом (`upload_data.py`).

## Установка

```bash
pip install -r requirements.txt
```

## Переменные окружения

```env
OPENAI_API_KEY=...
QDRANT_API_KEY=...
QDRANT_URL=http(s)://...:6333
```

## Запуск

```bash
# Загрузка данных в Qdrant
python upload_data.py

# Генерация синтетических датасетов
python scripts/generate_synthetic_dataset.py

# Оценка retrieval
python scripts/evaluate_rag.py
```

## Объем данных

| Метрика | Значение |
|---------|----------|
| Статей | ~411 000 |
| Чанков | ~5 400 000 |
| Single-hop примеров | 800 |
| Multi-hop примеров | 800 |

## Результаты оценки Retrieval

Оценка на 400 вопросах (200 single-hop + 200 multi-hop), TOP_K=10.

### Chunk-level метрики (точное совпадение чанка)

| Metric | Overall | Multi-hop | Single-hop |
|--------|---------|-----------|------------|
| hit@1 | 0.6775 | 0.7450 | 0.6100 |
| precision@1 | 0.6775 | 0.7450 | 0.6100 |
| recall@1 | 0.3461 | 0.0822 | 0.6100 |
| ndcg@1 | 0.6775 | 0.7450 | 0.6100 |
| hit@3 | 0.8700 | 0.9350 | 0.8050 |
| precision@3 | 0.4758 | 0.6833 | 0.2683 |
| recall@3 | 0.5009 | 0.1968 | 0.8050 |
| ndcg@3 | 0.7166 | 0.7079 | 0.7252 |
| hit@5 | 0.9250 | 0.9700 | 0.8800 |
| precision@5 | 0.4055 | 0.6350 | 0.1760 |
| recall@5 | 0.5831 | 0.2862 | 0.8800 |
| ndcg@5 | 0.7193 | 0.6827 | 0.7559 |
| hit@10 | 0.9500 | 0.9900 | 0.9100 |
| precision@10 | 0.3138 | 0.5365 | 0.0910 |
| recall@10 | 0.6788 | 0.4475 | 0.9100 |
| ndcg@10 | 0.7066 | 0.6474 | 0.7658 |
| MRR | 0.7811 | 0.8435 | 0.7187 |
| MAP | 0.5408 | 0.3629 | 0.7187 |

### Document-level метрики (совпадение статьи)

| Metric | Overall | Multi-hop | Single-hop |
|--------|---------|-----------|------------|
| hit@1 | 0.9175 | 0.9100 | 0.9250 |
| precision@1 | 0.9175 | 0.9100 | 0.9250 |
| recall@1 | 0.6900 | 0.4550 | 0.9250 |
| ndcg@1 | 0.9175 | 0.9100 | 0.9250 |
| hit@3 | 0.9625 | 0.9700 | 0.9550 |
| precision@3 | 0.3642 | 0.4100 | 0.3183 |
| recall@3 | 0.7850 | 0.6150 | 0.9550 |
| ndcg@3 | 0.8089 | 0.6745 | 0.9433 |
| hit@5 | 0.9775 | 0.9850 | 0.9700 |
| precision@5 | 0.2315 | 0.2690 | 0.1940 |
| recall@5 | 0.8213 | 0.6725 | 0.9700 |
| ndcg@5 | 0.8266 | 0.7036 | 0.9497 |
| hit@10 | 0.9800 | 0.9900 | 0.9700 |
| precision@10 | 0.1255 | 0.1540 | 0.0970 |
| recall@10 | 0.8700 | 0.7700 | 0.9700 |
| ndcg@10 | 0.8463 | 0.7429 | 0.9497 |
| MRR | 0.9427 | 0.9424 | 0.9429 |
| MAP | 0.7900 | 0.6372 | 0.9429 |
