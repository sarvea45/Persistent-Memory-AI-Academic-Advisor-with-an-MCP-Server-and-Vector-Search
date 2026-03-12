# AI Academic Advisor with Persistent Memory MCP Server

A production-ready AI academic advisor agent with long-term, context-aware memory, built using the **Memory, Control, and Process (MCP)** architectural pattern. The system combines a relational database (SQLite) and a vector database (ChromaDB) to give the LLM agent persistent memory that survives across sessions.

---

## Architecture Overview

```
User ──► LLM Agent ──► MCP Server (FastAPI)
                            ├── memory_write ──► SQLite + ChromaDB
                            ├── memory_read ──► SQLite
                            └── memory_retrieve_by_context ──► ChromaDB (vector search)
                                                                     ↑
                                                         Sentence Transformers
                                                         (all-MiniLM-L6-v2)
```

See `docs/memory_architecture.png` for a full visual diagram.

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| MCP Server | FastAPI + Python | Exposes memory tools via HTTP |
| Relational DB | SQLite + SQLAlchemy | Structured storage for conversations, preferences, milestones |
| Vector DB | ChromaDB | Semantic search over past memories |
| Embeddings | sentence-transformers | Converts text to vectors for similarity search |
| Agent | Python + Requests | Orchestrates tool calls to the MCP server |
| LLM | Claude API or Ollama | Generates advisor responses |

---

## Prerequisites

- Docker & Docker Compose
- (Optional) Claude API key or local Ollama installation

---

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd academic-advisor
cp .env.example .env
# Edit .env if you want to add a CLAUDE_API_KEY or OLLAMA_BASE_URL
```

### 2. Start the MCP server

```bash
docker-compose up --build
```

The MCP server will be available at `http://localhost:8000`.

### 3. Verify it's running

```bash
curl http://localhost:8000/health
# → {"status": "ok"}

curl http://localhost:8000/tools
# → {"tools": [...]}
```

### 4. Run the agent (interactive mode)

```bash
docker-compose --profile agent run agent_service python agent.py student_001
```

Or run the agent locally (pointing to the containerized MCP server):

```bash
cd agent
pip install -r requirements.txt
MCP_SERVER_URL=http://localhost:8000 python agent.py student_001
```

---

## API Reference

### Health Check
```
GET /health
→ {"status": "ok"}
```

### List Tools
```
GET /tools
→ {"tools": [{"name": "memory_write", ...}, ...]}
```

### Write Memory
```
POST /invoke/memory_write
Content-Type: application/json

{
  "memory_type": "conversation",
  "data": {
    "user_id": "student_001",
    "turn_id": 1,
    "role": "user",
    "content": "I want to study machine learning."
  }
}
→ 201 {"status": "success", "memory_id": "student_001_1"}
```

Supported `memory_type` values: `conversation`, `preference`, `milestone`.

### Read Memory
```
POST /invoke/memory_read
Content-Type: application/json

{
  "user_id": "student_001",
  "query_type": "last_n_turns",
  "params": {"n": 5}
}
→ 200 {"results": [...]}
```

Supported `query_type` values: `last_n_turns`, `milestones`, `preferences`.

### Retrieve by Context (Semantic Search)
```
POST /invoke/memory_retrieve_by_context
Content-Type: application/json

{
  "user_id": "student_001",
  "query_text": "What subjects does the student find interesting?",
  "top_k": 3
}
→ 200 {"results": [{"content": "...", "metadata": {...}, "score": 0.87}]}
```

---

## Memory Schemas

Defined in `mcp_server/memory_schemas.py` using Pydantic v2:

| Schema | Key Fields |
|--------|-----------|
| `Conversation` | `user_id`, `turn_id`, `role`, `content`, `timestamp` |
| `UserPreferences` | `user_id`, `preferences` (dict) |
| `Milestone` | `user_id`, `milestone_id`, `description`, `status`, `date_achieved` |

---

## Database Schema

SQLite tables managed by SQLAlchemy:

- **conversations** — conversation history with indexes on `user_id` and `(user_id, turn_id)`
- **user_preferences** — JSON-encoded preference dictionaries
- **milestones** — academic goals and their completion status

Data is persisted in the `./data/` directory, which is mounted as a Docker volume.

---

## Environment Variables

See `.env.example` for all variables. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/app/data/advisor_memory.db` | SQLite file path |
| `CHROMA_DB_PATH` | `/app/data/chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `MCP_SERVER_URL` | `http://mcp_server:8000` | MCP server URL (for agent) |
| `CLAUDE_API_KEY` | _(optional)_ | Anthropic API key |
| `OLLAMA_BASE_URL` | _(optional)_ | Ollama server URL |

---

## Design Decisions

### Why Hybrid Memory?
- **SQLite** excels at structured queries: "Give me the last 5 messages from user X"
- **ChromaDB** excels at semantic retrieval: "What did this user say about their career goals?"
- Together, they cover both exact lookup and fuzzy/semantic retrieval needs.

### Idempotent Writes
The `memory_write` tool uses upsert semantics — writing the same `(user_id, turn_id)` pair twice updates the record rather than creating a duplicate.

### Stateless Server
The MCP server holds no in-memory state. All state lives in the databases, enabling horizontal scaling.

### RAG Pattern
Before each LLM response, the agent calls `memory_retrieve_by_context` to inject relevant past memories into the system prompt, implementing Retrieval-Augmented Generation (RAG).

---

## Project Structure

```
academic-advisor/
├── docker-compose.yml
├── .env.example
├── .env
├── submission.json
├── README.md
├── docs/
│   └── memory_architecture.png
├── mcp_server/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py              # FastAPI application & tool endpoints
│   ├── memory_schemas.py    # Pydantic v2 schemas
│   ├── database.py          # SQLAlchemy models & session management
│   └── vector_store.py      # ChromaDB & embedding utilities
├── agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── agent.py             # Academic advisor agent
└── data/                    # Persisted database files (auto-created)
    ├── advisor_memory.db
    └── chroma_db/
```

---

## Testing

After `docker-compose up --build`:

```bash
# Health
curl http://localhost:8000/health

# Write a conversation turn
curl -X POST http://localhost:8000/invoke/memory_write \
  -H "Content-Type: application/json" \
  -d '{"memory_type":"conversation","data":{"user_id":"test","turn_id":1,"role":"user","content":"I love quantum physics."}}'

# Semantic search
curl -X POST http://localhost:8000/invoke/memory_retrieve_by_context \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","query_text":"What subjects interest this student?","top_k":3}'
```