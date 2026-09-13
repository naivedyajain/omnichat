"""FastAPI app: chat, file upload, session memory, tool loop."""
import json
import re
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles

from app.llm import chat
from app.rag import ingest, retrieve, has_documents
from app.tools import TOOLS, TOOLS_DESCRIPTION

app = FastAPI()

# In-memory conversation history, keyed by a fixed single-session id for the demo.
HISTORY = []


def _build_system_prompt(keys: dict, context: str) -> str:
    prompt = "You are a helpful assistant."
    prompt += "\n\n" + TOOLS_DESCRIPTION
    if context:
        prompt += "\n\nRelevant context from the user's uploaded document:\n" + context
    return prompt


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = body.get("message", "")
    provider = body.get("provider", "openai")
    keys = body.get("keys", {})

    print("\n" + "=" * 60)
    print(f"[Chat Step 1] New /api/chat request received.")
    print(f"[Chat Step 1] Provider: '{provider}'")
    print(f"[Chat Step 1] Message: '{message}'")
    print(f"[Chat Step 1] Keys present: {[k for k, v in keys.items() if v]}")

    # RAG: pull context if a document was uploaded.
    context = ""
    if has_documents():
        print("[Chat Step 2] RAG store contains documents. Performing vector search...")
        context = retrieve(message, keys.get("openai", ""))
        print(f"[Chat Step 2] RAG context retrieved length: {len(context)} chars.")
    else:
        print("[Chat Step 2] RAG inactive or no documents stored.")

    system_prompt = _build_system_prompt(keys, context)

    # Build the message list: system + history + new user message.
    HISTORY.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": system_prompt}] + HISTORY

    print(f"[Chat Step 3] Sending request to LLM provider '{provider}' (history depth: {len(HISTORY)} messages)...")
    try:
        reply = chat(provider, messages, keys)
        print(f"[Chat Step 3] LLM initial reply received (length: {len(reply)} chars).")
    except Exception as e:
        err_msg = f"ERROR: {e}"
        print(f"[Chat Step 3] LLM call failed with error: {err_msg}")
        return {"reply": err_msg}

    # Tool loop: if the model asked for a tool, run it and ask again.
    tool_used = None
    if '"tool"' in reply:
        print("[Chat Step 4] Tool call JSON pattern detected in LLM response.")
        try:
            match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}', reply, re.DOTALL)
            if not match:
                match = re.search(r'\{.*?"tool".*?\}', reply, re.DOTALL)
            if match:
                raw_tool_json = match.group(0)
                print(f"[Chat Step 4] Extracted tool JSON: {raw_tool_json}")
                call = json.loads(raw_tool_json)
                tool_name = call.get("tool")
                query = call.get("query", "")
                print(f"[Chat Step 4] Tool name: '{tool_name}', Query: '{query}'")

                if tool_name in TOOLS:
                    tool_used = tool_name
                    print(f"[Chat Step 5] Executing tool function '{tool_name}'...")
                    result = TOOLS[tool_name](query, keys)
                    print(f"[Chat Step 5] Tool function '{tool_name}' execution complete (result length: {len(result)} chars).")

                    # Feed the tool result back and ask for a final answer.
                    print(f"[Chat Step 6] Sending tool output back to LLM provider '{provider}' for synthesis...")
                    followup = messages + [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": f"Tool result:\n{result}\n\nAnswer the original question using this."},
                    ]
                    reply = chat(provider, followup, keys)
                    print(f"[Chat Step 6] Final LLM response generated (length: {len(reply)} chars).")
        except Exception as e:
            reply = f"ERROR running tool: {e}"
            print(f"[Chat Step 5] Error during tool execution loop: {e}")
    else:
        print("[Chat Step 4] No tool requested by LLM.")

    HISTORY.append({"role": "assistant", "content": reply})
    print(f"[Chat Step 7] Request cycle complete. Returning response to client.")
    print("=" * 60 + "\n")
    return {"reply": reply, "tool_used": tool_used}


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...), openai_key: str = Form("")):
    print(f"[Upload Step] File upload request: filename='{file.filename}'")
    content = await file.read()
    try:
        msg = ingest(file.filename, content, openai_key)
        print(f"[Upload Step] Ingestion status: {msg}")
        return {"status": msg}
    except Exception as e:
        err_msg = f"ERROR: {e}"
        print(f"[Upload Step] Ingestion error: {err_msg}")
        return {"status": err_msg}


@app.post("/api/reset")
async def reset_endpoint():
    print("[Reset Step] Resetting session conversation history.")
    HISTORY.clear()
    return {"status": "Conversation reset."}


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
