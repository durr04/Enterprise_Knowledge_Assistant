import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_JSON_PATH = PROJECT_ROOT / "config" / "config.json"


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 800


class EmbeddingsConfig(BaseModel):
    model: str = "text-embedding-3-small"


class ChunkingConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 120


class RetrievalConfig(BaseModel):
    vector_top_k: int = 8
    bm25_top_k: int = 8
    hybrid_vector_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5
    rerank_top_k: int = 4


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    vector_store_dir: str = "vector_store"


class MemoryConfig(BaseModel):
    max_history_turns: int = 6


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    paths: PathsConfig = PathsConfig()
    memory: MemoryConfig = MemoryConfig()

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.data_dir

    @property
    def vector_store_dir(self) -> Path:
        return PROJECT_ROOT / self.paths.vector_store_dir


def _load_json_config() -> AppConfig:
    if not CONFIG_JSON_PATH.exists():
        return AppConfig()
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    return AppConfig(**raw)


@lru_cache
def get_config() -> AppConfig:
    return _load_json_config()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
