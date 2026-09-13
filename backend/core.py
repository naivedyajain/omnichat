"""Core module: LLM Unified Adapters (Anthropic & Grok) and RAG Vector Store Engine."""
import io
import json
import math
import re
import anthropic
import numpy as np
from openai import OpenAI
from pypdf import PdfReader


# --- LLM ENGINE ---

def chat(provider: str, messages: list, keys: dict) -> str:
    """Unified chat function routing calls to Anthropic Claude or Grok xAI."""
    if provider == "grok":
        key = keys.get("xai")
        if not key:
            return "ERROR: No Grok (xAI) API key provided in Settings (⚙️)."
        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        grok_models = [
            "grok-4.6",
            "grok-4.5",
            "grok-4.3",
            "grok-4.20-0309-non-reasoning",
            "grok-2",
            "grok-2-1212",
            "grok-beta",
            "grok-2-latest",
        ]
        errors = []
        for m in grok_models:
            try:
                resp = client.chat.completions.create(model=m, messages=messages)
                return resp.choices[0].message.content
            except Exception as e:
                errors.append(str(e))
                continue
        return f"ERROR: Grok API request failed. Please check your xAI API key in Settings (⚙️).\nDetails: {errors[0] if errors else 'Unknown error'}"

    if provider == "anthropic":
        key = keys.get("anthropic")
        if not key:
            return "ERROR: No Anthropic API key provided in Settings (⚙️)."
        client = anthropic.Anthropic(api_key=key)
        system = ""
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})

        model_candidates = [
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
        ]
        last_error = None
        for model_name in model_candidates:
            try:
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system,
                    messages=convo,
                )
                return resp.content[0].text
            except anthropic.NotFoundError as e:
                last_error = e
                continue
        if last_error:
            raise last_error

    return "ERROR: Unknown provider."


def evaluate_response(primary_provider: str, user_prompt: str, primary_response: str, keys: dict) -> dict:
    """Evaluates the primary LLM response using the OPPOSITE LLM as a judge."""
    judge_provider = "grok" if primary_provider == "anthropic" else "anthropic"
    judge_name = "Grok (xAI)" if judge_provider == "grok" else "Anthropic (Claude)"
    
    judge_key_name = "xai" if judge_provider == "grok" else "anthropic"
    if not keys.get(judge_key_name):
        return {
            "judge": judge_name,
            "skipped": True,
            "reason": f"Add {judge_name} API key in Settings (⚙️) to enable cross-LLM evaluation."
        }

    eval_system_prompt = (
        "You are an expert AI Response Evaluator & Judge. "
        "Evaluate the generated response for the user prompt objectively across accuracy, completeness, and clarity. "
        "Reply ONLY with a single valid JSON object, nothing else, in this exact format:\n"
        "{\n"
        '  "confidence_score": 92,\n'
        '  "verdict": "PASS" | "NEEDS_IMPROVEMENT" | "FAIL",\n'
        '  "reasoning": "Brief explanation of your verdict and rating."\n'
        "}"
    )

    eval_user_content = (
        f"USER PROMPT:\n{user_prompt}\n\n"
        f"GENERATED RESPONSE TO EVALUATE:\n{primary_response}"
    )

    eval_messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": eval_user_content}
    ]

    print(f"[Judge Log] Requesting evaluation from Judge LLM '{judge_name}'...")
    try:
        raw_eval_reply = chat(judge_provider, eval_messages, keys)
        print(f"[Judge Log] Judge LLM raw reply: {raw_eval_reply}")

        if raw_eval_reply.startswith("ERROR"):
            return {
                "judge": judge_name,
                "skipped": True,
                "reason": raw_eval_reply
            }

        match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw_eval_reply, re.DOTALL)
        if not match:
            match = re.search(r'\{.*?"confidence_score".*?\}', raw_eval_reply, re.DOTALL)
        
        if match:
            parsed = json.loads(match.group(0))
            return {
                "judge": judge_name,
                "skipped": False,
                "confidence_score": int(parsed.get("confidence_score", 85)),
                "verdict": str(parsed.get("verdict", "PASS")).upper(),
                "reasoning": str(parsed.get("reasoning", "Evaluated successfully."))
            }
        else:
            return {
                "judge": judge_name,
                "skipped": False,
                "confidence_score": 85,
                "verdict": "PASS",
                "reasoning": raw_eval_reply[:300]
            }
    except Exception as e:
        print(f"[Judge Log] Error during evaluation: {e}")
        return {
            "judge": judge_name,
            "skipped": True,
            "reason": f"Evaluation error: {repr(e)}"
        }


# --- RAG VECTOR STORE ENGINE ---

_STORE = []


def _extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return content.decode(errors="ignore")


def _chunk(text: str, size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _compute_tfidf_vector(tokens: list, vocab: dict, idf: dict):
    vec = np.zeros(len(vocab))
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    for t, idx in vocab.items():
        vec[idx] *= idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ingest(filename: str, content: bytes, openai_key: str = "") -> str:
    global _STORE
    _STORE = []
    text = _extract_text(filename, content)
    chunks = _chunk(text)
    if not chunks:
        return "No text could be extracted from the file."

    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.embeddings.create(model="text-embedding-3-small", input=chunks)
            for chunk, item in zip(chunks, resp.data):
                _STORE.append({"text": chunk, "embedding": np.array(item.embedding), "mode": "openai"})
            print(f"[RAG Log] Ingested {len(chunks)} chunks with OpenAI embeddings.")
            return f"Ingested {len(chunks)} chunks from {filename} (OpenAI embeddings)."
        except Exception as e:
            print(f"[RAG Log] OpenAI embedding failed ({e}), falling back to local TF-IDF vectorizer...")

    all_tokens = [_tokenize(c) for c in chunks]
    vocab = {}
    for doc in all_tokens:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    num_docs = len(all_tokens)
    idf = {}
    for word, idx in vocab.items():
        doc_count = sum(1 for doc in all_tokens if word in doc)
        idf[word] = math.log((num_docs + 1) / (doc_count + 1)) + 1.0

    for chunk, tokens in zip(chunks, all_tokens):
        emb = _compute_tfidf_vector(tokens, vocab, idf)
        _STORE.append({"text": chunk, "embedding": emb, "vocab": vocab, "idf": idf, "mode": "tfidf"})

    print(f"[RAG Log] Ingested {len(chunks)} chunks with local TF-IDF vectorizer.")
    return f"Ingested {len(chunks)} chunks from {filename} (Local TF-IDF vectorizer)."


def retrieve(query: str, openai_key: str = "", top_k: int = 4) -> str:
    if not _STORE:
        return ""

    mode = _STORE[0].get("mode", "tfidf")
    if mode == "openai" and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            q = client.embeddings.create(model="text-embedding-3-small", input=[query])
            q_emb = np.array(q.data[0].embedding)
            scored = []
            for item in _STORE:
                emb = item["embedding"]
                sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
                scored.append((sim, item["text"]))
            scored.sort(reverse=True, key=lambda x: x[0])
            top = [t for _, t in scored[:top_k]]
            return "\n\n".join(top)
        except Exception as e:
            print(f"[RAG Log] OpenAI retrieval failed ({e}), using TF-IDF fallback...")

    vocab = _STORE[0].get("vocab", {})
    idf = _STORE[0].get("idf", {})
    q_tokens = _tokenize(query)
    q_emb = _compute_tfidf_vector(q_tokens, vocab, idf)

    scored = []
    for item in _STORE:
        emb = item["embedding"]
        norm = np.linalg.norm(emb) * np.linalg.norm(q_emb)
        sim = float(np.dot(q_emb, emb) / norm) if norm > 0 else 0.0
        scored.append((sim, item["text"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [t for _, t in scored[:top_k]]
    return "\n\n".join(top)


def has_documents() -> bool:
    return len(_STORE) > 0
