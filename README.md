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
