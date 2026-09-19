import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import AppConfig, Secrets
from src.reranker import rerank
from src.retrievers import build_hybrid_retriever, load_vector_store

logger = logging.getLogger(__name__)

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up question to be a standalone question that captures all "
            "necessary context. If the follow-up question is already standalone, "
            "return it unchanged. Only output the rewritten question, nothing else.",
        ),
        ("placeholder", "{history}"),
        ("human", "Follow-up question: {question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal Employee Knowledge Assistant for a company. "
            "Answer the user's question using ONLY the information in the "
            "provided context. \n\n"
            "Rules:\n"
            "- If the answer is not present in the context, respond exactly with: "
            "\"I could not find this information in the available company documents.\"\n"
            "- Do not invent, assume or use outside knowledge.\n"
            "- Keep the answer concise and professional.\n"
            "- Do not mention the word 'context' in your answer.\n\n"
            "Context:\n{context}",
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)


@dataclass
class RagResult:
    answer: str
    sources: List[str] = field(default_factory=list)


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._messages: List = []

    def as_messages(self) -> List:
        return self._messages[-(self.max_turns * 2):]

    def add_turn(self, question: str, answer: str) -> None:
        self._messages.append(HumanMessage(content=question))
        self._messages.append(AIMessage(content=answer))

    def clear(self) -> None:
        self._messages = []


class RagChain:
    def __init__(self, config: AppConfig, secrets: Secrets):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=secrets.openai_api_key,
        )
        vector_store = load_vector_store(config, secrets)
        self.retriever = build_hybrid_retriever(vector_store, config)

    def _condense_question(self, question: str, memory: ConversationMemory) -> str:
        if not memory.as_messages():
            return question
        chain = CONDENSE_QUESTION_PROMPT | self.llm
        response = chain.invoke({"history": memory.as_messages(), "question": question})
        return response.content.strip()

    def ask(self, question: str, memory: ConversationMemory) -> RagResult:
        standalone_question = self._condense_question(question, memory)
        logger.info("Standalone question: %s", standalone_question)

        candidates = self.retriever.invoke(standalone_question)
        reranked_docs = rerank(standalone_question, candidates, self.config)

        if not reranked_docs:
            answer = "I could not find this information in the available company documents."
            memory.add_turn(question, answer)
            return RagResult(answer=answer, sources=[])

        context = "\n\n---\n\n".join(doc.page_content for doc in reranked_docs)
        chain = ANSWER_PROMPT | self.llm
        response = chain.invoke(
            {
                "context": context,
                "history": memory.as_messages(),
                "question": question,
            }
        )
        answer = response.content.strip()

        sources = sorted({doc.metadata.get("source", "unknown") for doc in reranked_docs})
        memory.add_turn(question, answer)
        return RagResult(answer=answer, sources=sources)
