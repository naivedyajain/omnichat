# Tu Puch Mai bataunga! — Multi-LLM Bot Harness (v1.0)

A web-based multi-LLM bot harness supporting **Anthropic Claude** & **Grok (xAI)**, real-time web search (Tavily), read-only Gmail access over IMAP, and document Q&A (RAG with local TF-IDF vectorizer fallback).

Built with **FastAPI** + **Plain HTML/CSS/JS** with zero-disk persistence (all keys & histories are stored in browser session memory / `localStorage`).

---

## Clean Project Structure

```text
omnichat-harness/
├── backend/
│   ├── app.py              # Main FastAPI application & API routes (/api/chat, /api/upload, /api/reset)
│   ├── core.py             # Unified LLM provider adapter (Anthropic & Grok) + RAG Vector Engine
│   └── tools.py            # Integrated Tool Registry (Web Search & Gmail IMAP)
├── frontend/
│   └── index.html          # "Tu Puch Mai bataunga!" Single-page UI
├── README.md               # Setup & deployment documentation
└── requirements.txt        # Dependency manifest
```

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Uvicorn server
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in your browser. Open **Settings** (⚙️) to enter your Anthropic or Grok API keys.

---

## How to Deploy on Render

1. Create a new GitHub repository (e.g. `tu-puch-mai-bataunga`).
2. Push this codebase to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of clean 3-tier architecture"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
3. Log into **[Render.com](https://render.com)**:
   - Click **New +** &rarr; **Web Service**.
   - Connect your GitHub repository.
   - Set **Environment**: `Python 3`
   - Set **Build Command**: `pip install -r requirements.txt`
   - Set **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service**.
