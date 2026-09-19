import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from config.settings import AppConfig

logger = logging.getLogger(__name__)


@lru_cache
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, candidates: List[Document], config: AppConfig) -> List[Document]:
    if not candidates:
        return []

    top_k = config.retrieval.rerank_top_k
    try:
        encoder = _get_cross_encoder(config.reranker.model)
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
    except Exception as exc:
        logger.warning("Reranker unavailable (%s); returning top candidates unranked.", exc)
        return candidates[:top_k]
