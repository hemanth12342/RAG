# 📄 AI Document Analyst

A full-stack RAG (Retrieval-Augmented Generation) app that lets you upload PDF, DOCX, or Excel files and chat with their contents using Groq's high-speed LLM.

```
┌─────────────────────┐     API calls      ┌────────────────────────┐
│  Frontend (Vercel)  │ ─────────────────► │  Backend (Render)      │
│  HTML + CSS + JS    │                    │  FastAPI + LangChain   │
└─────────────────────┘                    │  FAISS + Groq LLM      │
                                           └────────────────────────┘
```

## 🚀 Quick Start (Local)

### 1. Backend

```bash
cd backend

# Create virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com/keys

# Run
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Frontend

No build step needed. Just open `frontend/index.html` in a browser, or serve it:

```bash
# Option A — Python
cd frontend && python -m http.server 3000

# Option B — Node (if installed)
cd frontend && npx serve .
```

> **Important**: Make sure `BACKEND_URL` in `frontend/app.js` is set to `http://localhost:8000` for local dev.

---

## ☁️ Deployment

### Backend → Render

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo, set **Root Directory** to `backend`
4. Render auto-detects `render.yaml` — no manual config needed
5. In **Environment → Environment Variables**, add:
   - `GROQ_API_KEY` = your key from https://console.groq.com/keys
6. Deploy. Note your Render URL (e.g., `https://rag-doc-analyst-api.onrender.com`)

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import your GitHub repo, set **Root Directory** to `frontend`
3. Framework preset: **Other**
4. Before deploying, update `BACKEND_URL` in `frontend/app.js`:
   ```js
   const BACKEND_URL = "https://rag-doc-analyst-api.onrender.com";
   ```
5. Deploy. Your app goes live at `https://your-project.vercel.app`

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 · CSS3 (Glassmorphism) · Vanilla JS |
| Backend | Python 3.11 · FastAPI · Uvicorn |
| LLM | Groq (llama3-8b-8192) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (in-memory, per session) |
| Orchestration | LangChain |
| File Formats | PDF · DOCX · Excel |

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload & process a document |
| `POST` | `/chat` | Ask a question about the document |
| `DELETE` | `/session/{id}` | Clear a session from memory |

## ⚠️ Known Limitations

- **Ephemeral storage**: Render free tier resets memory on each deploy/restart. Re-upload your document after a cold start.
- **Session-based**: Each upload creates a new session. Multiple tabs = multiple sessions.
- **File size**: Large files may hit Render's 512MB RAM limit on the free tier.
