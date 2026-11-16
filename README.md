# Hong Kong Policy Document Hybrid Search System

A full-stack retrieval system for Hong Kong policy documents that combines **exact title matching** in PostgreSQL with **semantic vector search** in Milvus, and provides a **RAG-based AI assistant** for document summarization and question answering.

- **Backend:** Python + Flask (RESTful API)
- **Frontend:** Streamlit (a Vue.js migration is also discussed below)
- **Relational Database:** PostgreSQL (structured metadata & full text)
- **Vector Database:** Milvus (text chunks & embeddings)
- **Embedding Model:** Sentence-Transformers (e.g. `bge-small-zh-v1.5`, 512-dim)
- **LLM:** OpenAI-compatible Chat Completion API for RAG generation

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Database Design](#database-design)
3. [Data Collection](#data-collection)
4. [Importing Data into PostgreSQL](#importing-data-into-postgresql)
5. [Building the Vector Index in Milvus](#building-the-vector-index-in-milvus)
6. [How Retrieval Works](#how-retrieval-works)
7. [Backend (Flask API)](#backend-flask-api)
8. [Frontend (Streamlit)](#frontend-streamlit)
9. [Migrating the Frontend to Vue.js](#migrating-the-frontend-to-vuejs)
10. [AI Assistant (RAG Agent)](#ai-assistant-rag-agent)
11. [Project Structure](#project-structure)
12. [Getting Started](#getting-started)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The system follows a classic **hybrid retrieval + RAG** architecture:

```
                         ┌──────────────────────────────┐
                         │        Streamlit Frontend     │
                         │  search | filters | AI chat   │
                         └───────────────┬───────────────┘
                                         │ HTTP / JSON
                         ┌───────────────▼───────────────┐
                         │         Flask Backend          │
                         │  search API | AI API | detail  │
                         └───────┬───────────────┬────────┘
                                 │               │
              SQL / exact match  │               │ vector search
                      ┌──────────▼─────┐   ┌─────▼──────────┐
                      │   PostgreSQL   │   │     Milvus      │
                      │ docs | orgs |  │   │ chunks + HNSW   │
                      │   categories   │   │   embeddings    │
                      └────────────────┘   └─────────────────┘
                                 │               │
                                 └───────┬───────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │ Embedding model + LLM (RAG)   │
                         └───────────────────────────────┘
```

**Design principle:** PostgreSQL and Milvus are linked by a shared business key `doc_id`. PostgreSQL handles structured filtering and exact matching, while Milvus handles semantic recall. The two result sets are merged, de-duplicated and returned to the frontend.

---

## Database Design

### Relational schema (PostgreSQL)

Three normalized tables: two dictionary tables and one main document table.

#### `orgs` — issuing authorities

| Column | Type | Constraints | Description |
|---|---|---|---|
| `org_id` | SERIAL | PRIMARY KEY | Authority ID |
| `org_name` | VARCHAR(128) | UNIQUE NOT NULL | Authority name |

#### `categories` — policy categories

| Column | Type | Constraints | Description |
|---|---|---|---|
| `cat_id` | SERIAL | PRIMARY KEY | Category ID |
| `cat_name` | VARCHAR(128) | UNIQUE NOT NULL | Category name |

#### `documents` — main document table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `doc_id` | SERIAL | PRIMARY KEY | Document ID |
| `title` | VARCHAR(512) | UNIQUE NOT NULL | Document title |
| `org_id` | INT | NOT NULL, FK → `orgs(org_id)` | Issuing authority |
| `cat_id` | INT | NOT NULL, FK → `categories(cat_id)` | Policy category |
| `publish_date` | DATE | NOT NULL | Publication date |
| `source_url` | VARCHAR(1024) | NULL | Official source link |
| `summary` | TEXT | NULL | Short summary |
| `content` | TEXT | NOT NULL | Full text |
| `create_time` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Ingestion time |

**Indexes:**

- Primary keys on all three tables.
- Unique constraint `uk_doc_title` on `documents(title)` — enables `ON CONFLICT DO NOTHING` to skip duplicates during import.
- B-tree indexes on `documents(org_id)`, `documents(cat_id)`, `documents(publish_date)` to accelerate filtering.

### Vector collection (Milvus)

Collection `document_chunks`:

| Field | Type | Description |
|---|---|---|
| `id` | INT64 | Auto-increment primary key |
| `doc_id` | INT | Business foreign key to `documents.doc_id` |
| `chunk_text` | VARCHAR(2048) | Text chunk |
| `vector` | FLOAT_VECTOR | Embedding (512-dim for `bge-small-zh-v1.5`) |

- Index: **HNSW** (`M = 16`, `ef_construction = 32`)
- Metric: **COSINE** (embeddings are L2-normalized)

### ER relationships

- One `orgs` row → many `documents` (one-to-many).
- One `categories` row → many `documents` (one-to-many).
- One `documents` row → many `document_chunks` (one-to-many).

---

## Data Collection

1. **Identify sources.** Collect policy documents from official Hong Kong government websites and department portals (immigration, education, labour, etc.), recording the title, issuing authority, publication date, source URL and full text.
2. **Normalize authorities and categories.** Map each document to a canonical authority name and category name so the dictionary tables stay consistent.
3. **Export to an intermediate format.** Store the raw corpus as JSON/CSV files, one record per document:

```json
{
  "title": "Example Policy Title",
  "org_name": "Immigration Department",
  "cat_name": "Immigration",
  "publish_date": "2025-03-01",
  "source_url": "https://www.immd.gov.hk/...",
  "summary": "Short summary...",
  "content": "Full text of the document..."
}
```

4. **Clean the text.** Remove navigation boilerplate, duplicate headers/footers and broken encoding; keep the encoding UTF-8.

---

## Importing Data into PostgreSQL

The ingestion script performs the following steps:

1. Ensure the schema exists (create tables and indexes if absent).
2. Insert dictionary values with `INSERT ... ON CONFLICT DO NOTHING`, then read back their `org_id` / `cat_id`.
3. Insert each document with the title unique constraint, again using `ON CONFLICT DO NOTHING` so re-running the importer is idempotent and duplicates are skipped.

Example SQL pattern:

```sql
INSERT INTO orgs (org_name) VALUES (%s)
ON CONFLICT (org_name) DO NOTHING;

INSERT INTO documents (title, org_id, cat_id, publish_date, source_url, summary, content)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (title) DO NOTHING
RETURNING doc_id;
```

After import, every document has a stable `doc_id` that is reused when building the vector index.

---

## Building the Vector Index in Milvus

1. **Create the collection** `document_chunks` with the schema above and an HNSW/COSINE index.
2. **Chunk each document.** Split `content` into overlapping chunks of roughly 200–500 characters (or tokens) so each chunk remains semantically complete.
3. **Embed chunks.** Encode all chunks with Sentence-Transformers using `normalize_embeddings=True` so cosine similarity is directly comparable.
4. **Insert in batches.** Each inserted record contains `doc_id`, `chunk_text` and the embedding vector.
5. **Flush and load.** Flush the collection and load it into memory before serving searches.

Chunking rules:

- Keep chunk size small enough for precise vector semantics but large enough to preserve context.
- Preserve `doc_id` on every chunk so search hits can be traced back to the source document.
- When multiple chunks of the same document match, aggregate by `doc_id` and keep the highest score.

---

## How Retrieval Works

A search request runs **two retrieval channels in parallel**:

### 1. Exact title match (PostgreSQL)

- Filter by authority, category and publication date using ordinary `WHERE` clauses.
- Match the keyword against the title with a case-insensitive `ILIKE` pattern.
- For larger corpora, replace `ILIKE` with a PostgreSQL GIN full-text index.

### 2. Semantic search (Milvus)

1. Encode the query with the same embedding model (normalized).
2. Run an HNSW cosine search, optionally applying a Milvus `filter` expression on `doc_id` when the user selects a restricted scope.
3. Aggregate chunk hits by `doc_id`, keeping the highest similarity and the best matching snippet.
4. Fetch document metadata from PostgreSQL.

### 3. Merge and respond

- Exact matches are returned first (high precision), followed by semantic results (high recall).
- Results are de-duplicated and grouped into `exact` and `semantic` lists.
- The frontend also stores the returned `doc_id` list as the AI assistant's document scope.

Important Milvus result detail: in the returned hits, `score` is a **top-level field**, while `doc_id` and `chunk_text` live inside `entity`:

```python
score = hit["score"]
doc_id = hit["entity"]["doc_id"]
chunk_text = hit["entity"]["chunk_text"]
```

---

## Backend (Flask API)

The backend is a lightweight RESTful service organized with blueprints:

- **Document routes:** filters (authorities/categories), hybrid search, document detail.
- **AI routes:** single-document summarization and RAG chat.
- **Common utilities:** unified JSON success/error responses.

Error responses must use the correct HTTP status code:

```python
def error_response(msg="error", code=400):
    return jsonify({"code": code, "msg": msg, "data": None}), code
```

Run Flask in debug mode during development so the server auto-reloads after file changes.

---

## Frontend (Streamlit)

The Streamlit app uses a two-column layout:

- **Left column:** keyword search box, sidebar filters (authority, category, date range), grouped results with collapsible panels, "View Full Text" and "Summarize with AI" buttons.
- **Right column:** AI assistant panel with settings, fixed-height chat area, message input and "Clear Chat".

### Persisting search state

Every button click in Streamlit reruns the whole script. To prevent search results from disappearing, persist all search state in `st.session_state`:

- `search_keyword`, `search_results`, `has_searched`
- `current_doc_ids` (scope handed to the AI assistant)
- `chat_history`

Search is executed only when the search button is clicked; rendering always reads from `st.session_state`, so clicking other buttons no longer clears the results.

The fixed-height chat area uses the native Streamlit container:

```python
chat_container = st.container(height=400, border=True)
with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
```

The frontend also defensively handles empty/non-JSON responses so a backend failure shows an error message instead of raising `AttributeError: 'NoneType' object has no attribute 'get'`.


## AI Assistant (RAG Agent)

The assistant supports two modes:

### 1. Document summarization

- Fetch the full document from PostgreSQL by `doc_id`.
- Build a summarization prompt and call the LLM.
- Return the generated summary to the frontend.

### 2. RAG question answering

1. **Embed the question** with the same embedding model.
2. **Retrieve relevant chunks** from Milvus:
   - If a document scope is provided, add a `filter` expression such as `doc_id in [1, 2, 3]`.
   - If no scope is provided (global mode), search the **entire collection** — do not return early just because `doc_ids` is empty.
3. **Build the prompt.** Place the retrieved chunks in the context section and instruct the LLM to answer only from the provided context.
4. **Generate and return** the answer; the frontend appends it to the chat history.

Reference retrieval function:

```python
def retrieve_relevant_chunks(question, doc_ids, top_k=3):
    query_vec = embedding_model.encode([question], normalize_embeddings=True)[0].tolist()

    search_kwargs = {
        "collection_name": Config.COLLECTION_NAME,
        "data": [query_vec],
        "limit": top_k,
        "output_fields": ["doc_id", "chunk_text"],
        "search_params": {"metric_type": "COSINE", "params": {"ef": 32}},
    }

    if doc_ids:
        if len(doc_ids) == 1:
            filter_expr = f"doc_id == {doc_ids[0]}"
        else:
            filter_expr = f"doc_id in [{', '.join(map(str, doc_ids))}]"
        search_kwargs["filter"] = filter_expr

    res = milvus_client.search(**search_kwargs)

    chunks = []
    for hit in res[0]:
        entity = hit.get("entity", {})
        chunks.append({
            "doc_id": entity.get("doc_id"),
            "content": entity.get("chunk_text", ""),
            "score": hit.get("score", 0.0),
        })
    return chunks
```

LLM configuration (API base URL, API key, model name) is centralized in the project config.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── config.py              # DB / Milvus / LLM configuration
│   ├── models/                # DB connections and clients
│   ├── routes/
│   │   ├── doc.py             # search, filters, detail
│   │   └── ai.py              # summarization and RAG chat
│   ├── services/              # embedding, chunking, Milvus helpers
│   └── utils.py               # unified response helpers
├── scripts/
│   ├── import_pg.py           # import corpus into PostgreSQL
│   └── build_index.py         # chunk, embed and load into Milvus
├── data/                      # raw JSON/CSV corpus
├── streamlit_app.py           # Streamlit frontend
├── run.py                     # Flask entry point
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL
- Milvus (standalone is sufficient)
- Node.js 18+ (only if using the Vue frontend)

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Edit `app/config.py` (or use environment variables) with:

- PostgreSQL connection string
- Milvus host/port and collection name
- Embedding model name
- LLM base URL, API key and model name

### 3. Prepare the databases

Start PostgreSQL and Milvus, create the database, then run the ingestion and index scripts:

```bash
python scripts/import_pg.py          # create tables and import documents
python scripts/build_index.py        # chunk, embed and load Milvus
```

### 4. Start the backend

```bash
python run.py
```

The Flask API runs at `http://127.0.0.1:5000`.

### 5. Start the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Open the URL printed in the terminal, run a search, and test the AI assistant.

### 6. (Optional) Start the Vue frontend

```bash
cd frontend
npm install
npm run dev
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/doc/orgs` | List all issuing authorities |
| GET | `/api/doc/categories` | List all policy categories |
| POST | `/api/doc/search` | Hybrid search (keyword + filters) |
| GET | `/api/doc/detail/<doc_id>` | Get full document content |
| POST | `/api/ai/summarize` | Summarize one document |
| POST | `/api/ai/chat` | RAG question answering |

### Search request example

```json
POST /api/doc/search
{
  "keyword": "visa",
  "org_id": null,
  "cat_id": null,
  "start_date": "2020-01-01",
  "end_date": "2026-12-31"
}
```

### Chat request example

```json
POST /api/ai/chat
{
  "question": "How do I apply for a visa extension?",
  "doc_ids": [1, 2, 3]
}
```

An empty `doc_ids` array means global search across the entire vector collection.

---


## License

This project is released under the MIT License.
