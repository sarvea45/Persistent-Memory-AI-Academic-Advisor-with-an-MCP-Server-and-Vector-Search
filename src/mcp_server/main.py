import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict

from database import init_db, get_session, ConversationTable, UserPreferencesTable, MilestoneTable
from vector_store import store_embedding, search_similar, get_vector_count
from memory_schemas import (
    MemoryWriteRequest,
    MemoryReadRequest,
    MemoryRetrieveRequest,
    Conversation,
    UserPreferences,
    Milestone
)

app = FastAPI(title="AI Academic Advisor MCP Server")

# Initialize the database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("Database initialized.")


# ────────────────────────────── Health ──────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ────────────────────────────── Tools listing ──────────────────────────────

TOOLS = [
    {
        "name": "memory_write",
        "description": "Persist a memory object (conversation turn, user preference, or milestone) to SQLite and optionally index it in ChromaDB for semantic search."
    },
    {
        "name": "memory_read",
        "description": "Retrieve structured records from SQLite. Supports query types like 'last_n_turns' to fetch recent conversation history for a user."
    },
    {
        "name": "memory_retrieve_by_context",
        "description": "Perform semantic (vector) search over past memories stored in ChromaDB. Returns the most relevant memories given a natural-language query."
    }
]


@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}


# ────────────────────────────── memory_write ──────────────────────────────

@app.post("/invoke/memory_write", status_code=201)
def invoke_memory_write(request: MemoryWriteRequest):
    session = get_session()
    memory_id = str(uuid.uuid4())

    try:
        if request.memory_type == "conversation":
            conv = Conversation(**request.data)
            # Upsert to avoid duplicates
            existing = (
                session.query(ConversationTable)
                .filter_by(user_id=conv.user_id, turn_id=conv.turn_id)
                .first()
            )
            if existing:
                existing.role = conv.role
                existing.content = conv.content
                existing.timestamp = conv.timestamp
                memory_id = f"{conv.user_id}_{conv.turn_id}"
            else:
                row = ConversationTable(
                    user_id=conv.user_id,
                    turn_id=conv.turn_id,
                    role=conv.role,
                    content=conv.content,
                    timestamp=conv.timestamp
                )
                session.add(row)
                memory_id = f"{conv.user_id}_{conv.turn_id}"
            session.commit()

            # Index in vector store
            store_embedding(
                doc_id=memory_id,
                text=conv.content,
                metadata={
                    "user_id": conv.user_id,
                    "turn_id": str(conv.turn_id),
                    "role": conv.role,
                    "memory_type": "conversation"
                }
            )

        elif request.memory_type == "preference":
            pref = UserPreferences(**request.data)
            existing = (
                session.query(UserPreferencesTable)
                .filter_by(user_id=pref.user_id)
                .first()
            )
            if existing:
                existing.preferences = json.dumps(pref.preferences)
            else:
                row = UserPreferencesTable(
                    user_id=pref.user_id,
                    preferences=json.dumps(pref.preferences)
                )
                session.add(row)
            session.commit()
            memory_id = f"pref_{pref.user_id}"

        elif request.memory_type == "milestone":
            ms = Milestone(**request.data)
            existing = (
                session.query(MilestoneTable)
                .filter_by(user_id=ms.user_id, milestone_id=ms.milestone_id)
                .first()
            )
            if existing:
                existing.description = ms.description
                existing.status = ms.status
                existing.date_achieved = ms.date_achieved
            else:
                row = MilestoneTable(
                    user_id=ms.user_id,
                    milestone_id=ms.milestone_id,
                    description=ms.description,
                    status=ms.status,
                    date_achieved=ms.date_achieved
                )
                session.add(row)
            session.commit()
            memory_id = f"{ms.user_id}_{ms.milestone_id}"

            # Index milestone description in vector store
            store_embedding(
                doc_id=memory_id,
                text=ms.description,
                metadata={
                    "user_id": ms.user_id,
                    "milestone_id": ms.milestone_id,
                    "status": ms.status,
                    "memory_type": "milestone"
                }
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown memory_type: {request.memory_type}")

        return {"status": "success", "memory_id": memory_id}

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        session.close()


# ────────────────────────────── memory_read ──────────────────────────────

@app.post("/invoke/memory_read")
def invoke_memory_read(request: MemoryReadRequest):
    session = get_session()
    try:
        if request.query_type == "last_n_turns":
            n = int(request.params.get("n", 10))
            rows = (
                session.query(ConversationTable)
                .filter_by(user_id=request.user_id)
                .order_by(ConversationTable.turn_id.desc())
                .limit(n)
                .all()
            )
            results = [
                {
                    "user_id": r.user_id,
                    "turn_id": r.turn_id,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None
                }
                for r in reversed(rows)
            ]
            return {"results": results}

        elif request.query_type == "milestones":
            rows = (
                session.query(MilestoneTable)
                .filter_by(user_id=request.user_id)
                .all()
            )
            results = [
                {
                    "user_id": r.user_id,
                    "milestone_id": r.milestone_id,
                    "description": r.description,
                    "status": r.status,
                    "date_achieved": r.date_achieved.isoformat() if r.date_achieved else None
                }
                for r in rows
            ]
            return {"results": results}

        elif request.query_type == "preferences":
            row = (
                session.query(UserPreferencesTable)
                .filter_by(user_id=request.user_id)
                .first()
            )
            if row:
                return {"results": [{"user_id": row.user_id, "preferences": json.loads(row.preferences)}]}
            return {"results": []}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown query_type: {request.query_type}")

    finally:
        session.close()


# ────────────────────────────── memory_retrieve_by_context ──────────────────────────────

@app.post("/invoke/memory_retrieve_by_context")
def invoke_memory_retrieve_by_context(request: MemoryRetrieveRequest):
    results = search_similar(
        query_text=request.query_text,
        user_id=request.user_id,
        top_k=request.top_k
    )
    return {"results": results}


# ────────────────────────────── vector count (for testing) ──────────────────────────────

@app.get("/vector_count")
def vector_count():
    return {"count": get_vector_count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ────────────────────────────── bulk memory_write (batch) ──────────────────────────────

@app.post("/invoke/memory_write_batch", status_code=201)
def invoke_memory_write_batch(requests_list: list[MemoryWriteRequest]):
    """
    Batch write multiple memories at once using efficient embedding batching.
    Ideal for indexing large amounts of historical conversation data.
    """
    from vector_store import store_embeddings_batch

    session = get_session()
    memory_ids = []
    texts_to_embed = []
    embed_ids = []
    embed_metadatas = []

    try:
        for request in requests_list:
            if request.memory_type == "conversation":
                conv = Conversation(**request.data)
                existing = session.query(ConversationTable).filter_by(
                    user_id=conv.user_id, turn_id=conv.turn_id).first()
                memory_id = f"{conv.user_id}_{conv.turn_id}"
                if existing:
                    existing.role = conv.role
                    existing.content = conv.content
                    existing.timestamp = conv.timestamp
                else:
                    session.add(ConversationTable(
                        user_id=conv.user_id, turn_id=conv.turn_id,
                        role=conv.role, content=conv.content, timestamp=conv.timestamp
                    ))
                memory_ids.append(memory_id)
                texts_to_embed.append(conv.content)
                embed_ids.append(memory_id)
                embed_metadatas.append({
                    "user_id": conv.user_id, "turn_id": str(conv.turn_id),
                    "role": conv.role, "memory_type": "conversation"
                })

        session.commit()

        # Batch embed and store all at once
        if texts_to_embed:
            store_embeddings_batch(embed_ids, texts_to_embed, embed_metadatas)

        return {"status": "success", "memory_ids": memory_ids, "count": len(memory_ids)}

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        session.close()