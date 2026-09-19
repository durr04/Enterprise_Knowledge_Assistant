import logging
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config.settings import get_config, get_secrets
from src.loaders import load_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    config = get_config()
    secrets = get_secrets()

    logger.info("Loading documents from %s", config.data_dir)
    raw_documents = load_documents(config.data_dir)
    if not raw_documents:
        logger.error("No documents were loaded. Aborting ingestion.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_documents)
    logger.info("Split %d documents into %d chunks", len(raw_documents), len(chunks))

    embeddings = OpenAIEmbeddings(
        model=config.embeddings.model,
        api_key=secrets.openai_api_key,
    )

    logger.info("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    config.vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(config.vector_store_dir))
    logger.info("Vector store saved to %s", config.vector_store_dir)


if __name__ == "__main__":
    run_ingestion()
