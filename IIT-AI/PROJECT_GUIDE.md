# Enterprise Knowledge Assistant with Advanced RAG
### GenAI Development Program — Capstone Project (Project 2)

This document is the single source of truth for evaluating this submission. It
covers the project objectives, how to run it end-to-end, and what every file
in the repository does.

---

## 1. Objective

Build a production-style RAG (Retrieval-Augmented Generation) application
that goes beyond a basic "chat with PDF" demo. The assistant answers employee
questions using a private set of company documents (leave policy, IT policy,
code of conduct, FAQs), and demonstrates:

- Multi-format document ingestion and chunking
- Embedding generation and local vector storage
- Hybrid retrieval (semantic + keyword search)
- Reranking of retrieved passages
- Conversation memory across follow-up questions
- Source citation for every answer
- Guardrails against hallucination
- A usable chat interface (Streamlit)

## 2. Business Scenario

**Employee Knowledge Assistant** for a fictional company, Acme Corp. Employees
ask natural-language questions about HR and IT policies and get grounded
answers with the source document referenced, instead of searching through
PDFs/Word docs manually.

## 3. Architecture

```
Documents (.txt / .docx / .pdf)
        |
        v
  Document Loader  (src/loaders.py)
        |
        v
  Chunking          (RecursiveCharacterTextSplitter)
        |
        v
  Embeddings        (OpenAI text-embedding-3-small)
        |
        v
  FAISS Vector Store (persisted in vector_store/)
        |
        +-- Vector / Semantic Search --+
        |                              +--> Hybrid Results (EnsembleRetriever)
Query --+-- BM25 Keyword Search --------+
                                        |
                                        v
                             Cross-Encoder Reranking
                                        |
                                        v
                            Relevant Context (top-k)
                                        |
                                        v
                     LLM + Conversation Memory
             (question condensing + grounded-answer prompt)
                                        |
                                        v
                     Grounded Answer + Source Citations
                                        |
                                        v
                             Streamlit Chat UI
```

**Why this design satisfies "advanced RAG":**

- **Hybrid search** — combines dense vector similarity (captures meaning) with
  BM25 keyword matching (captures exact terms like "VPN", "carry-forward"),
  merged via `EnsembleRetriever`.
- **Reranking** — a cross-encoder model re-scores the hybrid candidates
  against the query so the most relevant chunks reach the LLM, not just the
  ones that scored well on the first-pass retrievers.
- **Memory** — follow-up questions ("What about carry-forward?") are rewritten
  into standalone questions using prior conversation turns before retrieval.
- **Grounding** — the answer prompt forces the model to only use retrieved
  context and to explicitly say when information isn't available.

## 4. Repository Structure and File Responsibilities

```
IIT-AI/
├── app.py
├── config/
│   ├── config.json
│   └── settings.py
├── data/
├── scripts/
│   └── make_sample_docx.py
├── src/
│   ├── loaders.py
│   ├── ingest.py
│   ├── retrievers.py
│   ├── reranker.py
│   └── rag_chain.py
├── vector_store/            (generated after ingestion, not committed)
├── requirements.txt
├── .env.example
└── README.md
```

| File | Responsibility |
|---|---|
| `app.py` | Streamlit entry point. Renders the chat UI, keeps per-session chat history and memory, calls the RAG chain for each question, and displays the answer with its sources. |
| `config/config.json` | All tunable, non-secret settings: LLM model name, temperature, max tokens, embedding model, chunk size/overlap, retrieval top-k values, hybrid search weights, reranker model, folder paths, memory window size. |
| `config/settings.py` | Loads `config.json` into typed Pydantic models (`AppConfig`) and loads the OpenAI API key from `.env` via `pydantic-settings` (`Secrets`). Exposes cached `get_config()` / `get_secrets()` accessors used everywhere else. |
| `data/` | Sample company documents used as the private knowledge base: `leave_policy.txt`, `it_policy.txt`, `company_faq.txt`, `code_of_conduct.docx`. Demonstrates ingestion of 2+ formats (.txt and .docx). |
| `scripts/make_sample_docx.py` | One-time helper used to generate `code_of_conduct.docx` as a valid Word file without extra dependencies. Not used at runtime. |
| `src/loaders.py` | Reads every supported file in `data/`, dispatches to the correct LangChain loader based on extension (`TextLoader`, `PyPDFLoader`, `Docx2txtLoader`), tags each document with its source filename, and skips/logs unreadable files instead of crashing the batch. |
| `src/ingest.py` | The ingestion pipeline: loads raw documents, splits them into overlapping chunks, generates embeddings, builds a FAISS index, and persists it to `vector_store/`. Run manually whenever `data/` changes. |
| `src/retrievers.py` | Loads the persisted FAISS index and builds the hybrid retriever — a FAISS similarity retriever combined with a BM25 keyword retriever via LangChain's `EnsembleRetriever`, weighted per `config.json`. |
| `src/reranker.py` | Loads a sentence-transformers cross-encoder model and re-scores hybrid search candidates against the query, keeping only the top-k most relevant chunks. Falls back to the unranked top-k if the model can't be loaded (e.g. offline). |
| `src/rag_chain.py` | The orchestration layer: `ConversationMemory` stores recent Q&A turns; `RagChain` condenses follow-up questions into standalone form, retrieves + reranks context, and calls the LLM with a grounding prompt that forces citation-worthy, hallucination-free answers. |
| `requirements.txt` | Pinned Python dependencies for the virtual environment. |
| `.env.example` | Template for the required `.env` file (holds `OPENAI_API_KEY`). Never commit the real `.env`. |
| `README.md` | Quick-start version of this guide. |

## 5. Setup Instructions

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Step 1 — Create and activate a virtual environment

```powershell
cd c:\IIT-AI
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3 — Configure secrets

```powershell
copy .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=sk-your-actual-key
```

### Step 4 — (Optional) Adjust configuration

Edit `config/config.json` to change the LLM model, temperature, chunk size,
number of retrieved chunks, hybrid search weighting, or the reranker model.

## 6. How to Run the Project

### Step 1 — Build the knowledge base (run once, and again whenever `data/` changes)

```powershell
python -m src.ingest
```

This reads all files in `data/`, chunks them, generates embeddings, and saves
a FAISS index to `vector_store/`.

### Step 2 — Launch the assistant

```powershell
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open
it in a browser.

### Step 3 — Ask questions

Try the following in the chat box:

1. `What is the leave policy?`
2. `What about carry-forward?` — a follow-up; the assistant should
   understand this refers to leave carry-forward, thanks to conversation
   memory.
3. `How do I reset my VPN password?`
4. `What is the company's travel reimbursement policy?`
5. `What is the process for filing an ethics complaint?`
6. `What is the company's stock price?` — an out-of-scope question; the
   assistant should respond that it could not find the information, instead
   of guessing.

Use the sidebar **"Clear conversation"** button to reset chat history and
memory.

## 7. Sample Input / Output

**Input documents:** `leave_policy.txt`, `it_policy.txt`, `company_faq.txt`,
`code_of_conduct.docx`.

**Sample conversation:**

```
User: What is the leave policy?
AI:   Employees receive 18 days of paid annual leave per year, credited on a
      pro-rata basis for new joiners. Sick leave is 10 days per year, with a
      medical certificate required beyond 2 consecutive days.
      Sources: leave_policy.txt

User: What about carry-forward?
AI:   Up to 5 unused annual leave days can be carried forward to the next
      year and must be used within Q1, after which they lapse.
      Sources: leave_policy.txt, company_faq.txt

User: What is the company's stock price?
AI:   I could not find this information in the available company documents.
      Sources: (none)
```

## 8. Requirement Coverage Summary

| Functional Requirement | Where It's Implemented |
|---|---|
| Document ingestion, 2+ formats | `src/loaders.py` (.txt, .docx, .pdf) |
| Chunking | `src/ingest.py` (`RecursiveCharacterTextSplitter`) |
| Embeddings + local vector store | `src/ingest.py` (OpenAI embeddings + FAISS) |
| Basic semantic retrieval | `src/retrievers.py` (`FAISS.as_retriever`) |
| Hybrid search | `src/retrievers.py` (`EnsembleRetriever` with FAISS + BM25) |
| Reranking | `src/reranker.py` (cross-encoder) |
| Conversational memory | `src/rag_chain.py` (`ConversationMemory` + question condensing) |
| Source citations | `src/rag_chain.py` returns `sources`; shown in `app.py` |
| Hallucination mitigation | Grounding instructions in the answer prompt (`ANSWER_PROMPT`) |
| Streamlit UI (chat, history, reset, sources, errors) | `app.py` |
| Config-driven model/temperature/etc. | `config/config.json` + `config/settings.py` |
| Secrets via `.env` | `.env.example` + `pydantic-settings` |

## 9. Known Limitations / Notes for the Examiner

- The cross-encoder reranker model is downloaded from Hugging Face on first
  use; if the machine has no internet access at that moment, the app
  automatically falls back to unranked hybrid results so it keeps working.
- The vector store (`vector_store/`) is generated locally and intentionally
  excluded from version control — it must be rebuilt with `python -m
  src.ingest` after cloning.
- Sample documents are illustrative/fictional (Acme Corp) and created for
  demonstration purposes only.
