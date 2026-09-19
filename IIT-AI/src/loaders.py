import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def load_documents(data_dir: Path) -> List[Document]:
    documents: List[Document] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", file_path.name)
            continue
        try:
            docs = _load_single_file(file_path)
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
            logger.info("Loaded %s (%d chunks pre-split)", file_path.name, len(docs))
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path.name, exc)

    return documents


def _load_single_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(file_path), encoding="utf-8").load()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(file_path)).load()

    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(file_path)).load()

    raise ValueError(f"Unsupported file extension: {suffix}")
