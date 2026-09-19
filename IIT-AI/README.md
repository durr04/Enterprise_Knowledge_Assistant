# Enterprise Knowledge Assistant with Advanced RAG

GenAI Development Program — Final Capstone Submission (Project 2).

An Employee Knowledge Assistant that answers questions about company policies
(Leave Policy, IT Policy, Code of Conduct, FAQs) using an advanced
Retrieval-Augmented Generation (RAG) pipeline: **hybrid search (vector + BM25)
→ reranking → grounded LLM answer with source citations**, wrapped in a
conversational Streamlit UI with memory.

## Architecture

```
Documents (.txt / .docx / .pdf)
        │
        ▼
  Document Loader (src/loaders.py)
        │
        ▼
  Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
  Embeddings (OpenAI text-embedding-3-small)
        │
        ▼
  FAISS Vector Store (persisted locally in vector_store/)
        │
        ├── Vector/Semantic Search ──┐
        │                            ├─► Hybrid Results (EnsembleRetriever)
Query ──┴── BM25 Keyword Search ──────┘
                                       │
                                       ▼
                             Cross-Encoder Reranking
                                       │
                                       ▼
                              Relevant Context (top-k)
                                       │
                                       ▼
                        LLM + Conversation Memory
                     (question condensing + grounded answer prompt)
                                       │
                                       ▼
                        Grounded Answer + Source Citations
                                       │
                                       ▼
                              Streamlit Chat UI
```

## Project Structure

```
IIT-AI/
├── app.py                  # Streamlit UI entry point
├── config/
│   ├── config.json         # Model, temperature, chunking, retrieval params
│   └── settings.py         # Pydantic settings loader (.env + config.json)
├── data/                   # Source company documents (.txt, .docx)
├── scripts/
│   └── make_sample_docx.py # Helper used to generate the sample .docx
├── src/
│   ├── loaders.py           # Multi-format document loader
│   ├── ingest.py            # Chunk -> embed -> persist FAISS index
│   ├── retrievers.py        # Hybrid retriever (FAISS + BM25 ensemble)
│   ├── reranker.py          # Cross-encoder reranking
│   └── rag_chain.py         # Conversational RAG chain w/ memory + citations
├── vector_store/           # Persisted FAISS index (generated, gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Create and activate a virtual environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure your API key**

   Copy `.env.example` to `.env` and add your OpenAI API key:

   ```
   OPENAI_API_KEY=sk-...
   ```

4. **Tune model/retrieval settings (optional)**

   Edit `config/config.json` to change the LLM model, temperature, chunk
   size, number of retrieved chunks, hybrid search weights, etc.

## Running the Application

1. **Ingest the documents** (builds the local FAISS vector store from
   `data/`). Re-run this any time documents in `data/` change:

   ```powershell
   python -m src.ingest
   ```

2. **Launch the Streamlit app**

   ```powershell
   streamlit run app.py
   ```

3. Open the URL shown in the terminal (typically `http://localhost:8501`)
   and start asking questions, e.g.:
   - "What is the leave policy?"
   - "What about carry-forward?" (follow-up — memory resolves "it" refers to leave)
   - "How do I reset my VPN password?"
   - "What is the travel reimbursement policy?"

## Sample Input / Output

**Input documents** (in `data/`): `leave_policy.txt`, `it_policy.txt`,
`company_faq.txt`, `code_of_conduct.docx`.

**Sample conversation:**

```
User: What is the leave policy?
AI:   Employees receive 18 days of paid annual leave per year, credited on a
      pro-rata basis for new joiners. Sick leave is 10 days per year...
      Sources: leave_policy.txt

User: What about carry-forward?
AI:   Up to 5 unused annual leave days can be carried forward to the next
      year and must be used within Q1, after which they lapse.
      Sources: leave_policy.txt, company_faq.txt
```

## How Each Requirement Is Addressed

| Requirement | Implementation |
|---|---|
| Document ingestion (≥2 formats) | `src/loaders.py` supports `.txt`, `.docx`, `.pdf` |
| Chunking + Embeddings + Vector store | `src/ingest.py` using `RecursiveCharacterTextSplitter` + OpenAI embeddings + FAISS |
| Basic vector retrieval | `FAISS.as_retriever()` in `src/retrievers.py` |
| Hybrid search (vector + BM25) | `EnsembleRetriever` combining FAISS + `BM25Retriever` |
| Reranking | Cross-encoder (`sentence-transformers`) in `src/reranker.py`, with graceful fallback |
| Conversational memory | `ConversationMemory` (sliding window) + LLM-based question condensing in `src/rag_chain.py` |
| Source citations | Returned alongside every answer, shown in the UI |
| Hallucination mitigation | System prompt instructs the LLM to only use retrieved context and state clearly when information isn't found |
| Streamlit UI | `app.py` — chat interface, history, reset button, source display, error handling |
| Config-driven (model/temperature) | `config/config.json` + `config/settings.py` |
| Secrets via `.env` | `.env.example` + `pydantic-settings` |

## Notes

- The reranker downloads a small cross-encoder model
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) from Hugging Face on first use. If
  no internet access is available, it falls back to unranked hybrid results
  automatically.
- `vector_store/` and `.env` are excluded from version control via
  `.gitignore`.
