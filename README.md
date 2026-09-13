# Tu Puch Mai bataunga! — Multi-LLM Bot Harness

A web-based multi-LLM bot harness supporting **Anthropic Claude** & **Grok (xAI)**, real-time web search (Tavily), read-only Gmail access over IMAP, and document Q&A (RAG with local TF-IDF fallback).

Built with **FastAPI** + **Plain HTML/CSS/JS** with zero-disk persistence (all keys & histories are stored in browser session memory / `localStorage`).

---

## Features

- **Multi-LLM Provider Switching**: Instantly switch between Anthropic (`claude-sonnet-4-6`) and Grok (`grok-4.6`) with candidate model fallback.
- **Document RAG (Q&A)**: Upload PDFs or text files for instant Q&A powered by an in-memory TF-IDF vectorizer (no OpenAI key required!).
- **Web Search Connector**: Real-time web search via Tavily API with full raw JSON response logging.
- **Read-Only Gmail Connector**: Fetches recent emails over IMAP with automatic credential regex sanitization (stripping non-breaking spaces `\xa0`, hyphens, and whitespace).
- **Responsive Dark UI**: Collapsible sidebar, animated thinking indicator, clickable starter prompt cards, thumbs up/down reaction buttons, and `localStorage` credential persistence.

---

## Folder Structure

```text
omnichat-harness/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server & route endpoints
│   ├── llm.py               # Unified LLM provider adapter (Anthropic & Grok)
│   ├── rag.py               # In-memory document chunking & vector retrieval
│   ├── tools/
│   │   ├── __init__.py      # Tool registry & prompt instructions
│   │   ├── web_search.py    # Tavily web search integration
│   │   └── email.py         # Read-only Gmail IMAP access & sanitization
│   └── static/
│       └── index.html       # Single-page frontend UI
├── README.md
└── requirements.txt         # Dependency manifest
```

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Uvicorn server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in your browser. Open **Settings** (⚙️) to enter your Anthropic or Grok API keys.

---

## How to Deploy on Render

1. Create a new GitHub repository (e.g. `tu-puch-mai-bataunga`).
2. Push this codebase to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
3. Log into **[Render.com](https://render.com)**:
   - Click **New +** &rarr; **Web Service**.
   - Connect your GitHub repository.
   - Set **Environment**: `Python 3`
   - Set **Build Command**: `pip install -r requirements.txt`
   - Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service**.
