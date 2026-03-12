"""
AI Academic Advisor Agent
Uses the MCP server's memory tools for context-aware conversations.
Uses Groq API for LLM responses.
"""
import os
import requests
from groq import Groq

MCP_BASE_URL = os.environ.get("MCP_SERVER_URL", "http://mcp_server:8000")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

groq_client = Groq(api_key=GROQ_API_KEY)


def write_memory(memory_type: str, data: dict) -> dict:
    resp = requests.post(f"{MCP_BASE_URL}/invoke/memory_write",
                         json={"memory_type": memory_type, "data": data})
    return resp.json()


def read_memory(user_id: str, query_type: str = "last_n_turns", params: dict = None) -> dict:
    resp = requests.post(f"{MCP_BASE_URL}/invoke/memory_read",
                         json={"user_id": user_id, "query_type": query_type, "params": params or {"n": 10}})
    return resp.json()


def retrieve_context(user_id: str, query_text: str, top_k: int = 3) -> dict:
    resp = requests.post(f"{MCP_BASE_URL}/invoke/memory_retrieve_by_context",
                         json={"user_id": user_id, "query_text": query_text, "top_k": top_k})
    return resp.json()


def build_system_prompt(context_memories: list) -> str:
    context_str = ""
    if context_memories:
        context_str = "\n".join([f"- {m['content']}" for m in context_memories])
        context_str = f"\nRelevant past context from memory:\n{context_str}\n"
    return f"""You are a helpful and empathetic AI academic advisor. Your role is to:
1. Help students plan their academic journey
2. Track their goals, milestones, and preferences
3. Provide personalized guidance based on their history
4. Suggest courses, resources, and strategies tailored to their interests
{context_str}
Be warm, encouraging, and specific in your advice."""


def chat_with_groq(messages: list, system_prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].message.content


def run_advisor(user_id: str):
    print(f"\n🎓 AI Academic Advisor - Session for user: {user_id}")
    print("Type 'quit' to exit.\n")
    turn_id = 1
    conversation_history = []

    # Load existing conversation history
    history = read_memory(user_id, "last_n_turns", {"n": 10})
    if history.get("results"):
        existing_turns = history["results"]
        for turn in existing_turns:
            conversation_history.append({"role": turn["role"], "content": turn["content"]})
            turn_id = max(turn_id, turn["turn_id"] + 1)
        print(f"📚 Loaded {len(existing_turns)} previous conversation turns.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! Your session has been saved.")
            break
        if not user_input:
            continue

        # Retrieve relevant context from vector store
        context = retrieve_context(user_id, user_input, top_k=3)
        context_memories = context.get("results", [])

        system_prompt = build_system_prompt(context_memories)
        conversation_history.append({"role": "user", "content": user_input})

        # Persist user message
        write_memory("conversation", {
            "user_id": user_id,
            "turn_id": turn_id,
            "role": "user",
            "content": user_input
        })
        turn_id += 1

        # Generate response via Groq
        try:
            response = chat_with_groq(conversation_history, system_prompt)
        except Exception as e:
            response = f"[Error generating response: {e}]"

        print(f"\nAdvisor: {response}\n")
        conversation_history.append({"role": "assistant", "content": response})

        # Persist assistant response
        write_memory("conversation", {
            "user_id": user_id,
            "turn_id": turn_id,
            "role": "assistant",
            "content": response
        })
        turn_id += 1


if __name__ == "__main__":
    import sys
    uid = sys.argv[1] if len(sys.argv) > 1 else "student_001"
    run_advisor(uid)