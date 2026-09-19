import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import logging

import streamlit as st

from config.settings import get_config, get_secrets
from src.rag_chain import ConversationMemory, RagChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_rag_chain() -> RagChain:
    config = get_config()
    secrets = get_secrets()
    return RagChain(config, secrets)


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=get_config().memory.max_history_turns)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Enterprise Knowledge Assistant")
        st.caption("Ask questions about company policies (Leave, IT, Code of Conduct, FAQs).")
        if st.button("🔄 Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()


def render_chat_history() -> None:
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                st.caption("Sources: " + ", ".join(turn["sources"]))


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🧠 Employee Knowledge Assistant")
    render_chat_history()

    user_question = st.chat_input("Ask a question about company policies...")
    if not user_question:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            try:
                chain = get_rag_chain()
                result = chain.ask(user_question, st.session_state.memory)
                st.markdown(result.answer)
                if result.sources:
                    st.caption("Sources: " + ", ".join(result.sources))
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result.answer, "sources": result.sources}
                )
            except FileNotFoundError as exc:
                error_msg = (
                    f"⚠️ {exc}\n\nPlease run `python -m src.ingest` first to build the "
                    "knowledge base."
                )
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            except Exception as exc:
                error_msg = f"⚠️ Something went wrong while answering your question: {exc}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
