"""In-memory RAG: chunk, embed (OpenAI or TF-IDF fallback), retrieve by cosine similarity."""
import math
import re
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import io

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
    return re.findall(r'\w+', text.lower())


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
    _STORE = []  # reset on each new upload for a clean demo
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

    # Fallback TF-IDF embedding (No OpenAI key required!)
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

    # TF-IDF retrieval fallback
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
