import logging

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config.settings import AppConfig, Secrets

logger = logging.getLogger(__name__)


def load_vector_store(config: AppConfig, secrets: Secrets) -> FAISS:
    if not config.vector_store_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {config.vector_store_dir}. "
            "Run 'python -m src.ingest' first."
        )
    embeddings = OpenAIEmbeddings(model=config.embeddings.model, api_key=secrets.openai_api_key)
    return FAISS.load_local(
        str(config.vector_store_dir), embeddings, allow_dangerous_deserialization=True
    )


def build_hybrid_retriever(vector_store: FAISS, config: AppConfig) -> EnsembleRetriever:
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": config.retrieval.vector_top_k}
    )

    all_docs = list(vector_store.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.retrieval.bm25_top_k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[
            config.retrieval.hybrid_vector_weight,
            config.retrieval.hybrid_bm25_weight,
        ],
    )
    return hybrid_retriever
