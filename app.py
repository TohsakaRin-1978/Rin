import streamlit as st

from core.chatbot_engine import ChatbotEngine
from core.document_loader import load_txt_files, load_uploaded_txt
from core.text_splitter import split_text
from core.vector_store import VectorStore
from database.db import get_recent_history, init_db, save_message


st.set_page_config(
    page_title="Educational NLP Chatbot",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        }
        .hero-card {
            background: linear-gradient(135deg, #07111f 0%, #123a72 55%, #1d4ed8 100%);
            color: white;
            padding: 30px 30px 26px 30px;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.20);
            margin-bottom: 16px;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            opacity: 0.94;
            line-height: 1.65;
            max-width: 880px;
        }
        .hero-badges {
            margin-top: 14px;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 20px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
        }
        .info-chip {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.12);
            color: #1d4ed8;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .feature-card {
            background: white;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            height: 100%;
        }
        .feature-kicker {
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .feature-title {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .feature-text {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .section-desc {
            color: #475569;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

st.title("Educational NLP Chatbot")
st.caption("A lightweight educational question-answering system based on local knowledge retrieval and RAG")

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Educational NLP Chatbot</div>
        <div class="hero-subtitle">
            Local TXT knowledge base, semantic retrieval, conversational answering, and SQLite chat history.
            The interface keeps the required document workflow intact while presenting it in a cleaner product-style layout.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

badge_cols = st.columns(4)
for col, label in zip(
    badge_cols,
    ["Local TXT Knowledge Base", "Semantic Retrieval", "SQLite Chat History", "FastAPI Optional API"],
):
    with col:
        st.markdown(f'<span class="info-chip">{label}</span>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotEngine(st.session_state.vector_store)


with st.sidebar:
    st.markdown('<div class="section-title">System Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Manage the knowledge base, upload TXT files, and clear the current chat.</div>', unsafe_allow_html=True)
    user_id = st.text_input("User ID", value="student_001")

    if st.button("Refresh Knowledge Base Status"):
        st.session_state["kb_status_refresh"] = True

    if st.button("Build Knowledge Base"):
        docs = load_txt_files("data/knowledge_base")
        all_chunks = []
        for doc in docs:
            all_chunks.extend(split_text(doc["content"]))
        count = st.session_state.vector_store.add_documents(all_chunks)
        st.success(f"Knowledge base built successfully. Total chunks: {count}")

    uploaded_file = st.file_uploader("Upload course material", type=["txt"])
    if uploaded_file is not None:
        content = load_uploaded_txt(uploaded_file)
        chunks = split_text(content)
        count = st.session_state.vector_store.add_documents(chunks)
        st.success(f"Uploaded document added successfully. Total chunks: {count}")

    if st.button("Clear Current Chat"):
        st.session_state.messages = []
        st.success("Current chat cleared successfully.")

    st.markdown('<div class="section-title">Knowledge Base Files</div>', unsafe_allow_html=True)
    docs = load_txt_files("data/knowledge_base")
    if docs:
        for doc in docs:
            st.markdown(f'<span class="info-chip">{doc["filename"]}</span>', unsafe_allow_html=True)
    else:
        st.write("No knowledge base files found.")

    st.markdown('<div class="section-title">Project Data Paths</div>', unsafe_allow_html=True)
    st.code("data/knowledge_base\ndata/chroma_db\ndata/chatbot.db", language="text")


recent_history = get_recent_history(user_id, limit=5)
knowledge_files = len(docs)
turn_count = len(recent_history)
message_count = len(st.session_state.messages)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Knowledge Files", knowledge_files)
with col2:
    st.metric("Chat Turns", turn_count)
with col3:
    st.metric("Current Session Messages", message_count)

overview_left, overview_right = st.columns([1.35, 1])
with overview_left:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-kicker">System Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-title">A local educational question-answering workflow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="feature-text">'
        'The system loads TXT course materials from <code>data/knowledge_base</code>, splits them into overlapping chunks, '
        'stores semantic vectors in ChromaDB, generates responses with a lightweight transformer pipeline, and keeps '
        'conversation history in SQLite for multi-turn context.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
with overview_right:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-kicker">Demonstration Focus</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-title">Built for thesis defense and demo presentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="feature-text">'
        'The layout highlights the project architecture, the local knowledge base files, the retrieved source panel, '
        'and the persisted chat history so the implementation is easy to explain on stage.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

tab_chat, tab_knowledge, tab_history = st.tabs(["Chat", "Knowledge Base", "History"])

with tab_chat:
    with st.container(border=True):
        st.markdown('<div class="section-title">Conversation Area</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Ask questions about the course knowledge and review the retrieved sources for each answer.</div>', unsafe_allow_html=True)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask a question about the course knowledge...")

        if question:
            if not question.strip():
                st.warning("Please enter a valid question.")
            else:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)

                history = get_recent_history(user_id, limit=5)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving knowledge and generating response..."):
                        answer, sources = st.session_state.chatbot.answer(question, history)
                        st.write(answer)
                        with st.expander("Retrieved knowledge sources"):
                            if sources:
                                for i, source in enumerate(sources, start=1):
                                    st.markdown(f"**Source {i}:**")
                                    st.write(source)
                            else:
                                st.write("No knowledge sources were retrieved.")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_message(user_id, question, answer)

with tab_knowledge:
    with st.container(border=True):
        st.markdown('<div class="section-title">Knowledge Base Management</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Build the vector knowledge base from local TXT files or upload new course materials.</div>', unsafe_allow_html=True)

        left, right = st.columns([1.2, 1])
        with left:
            st.write("Local knowledge base files")
            if docs:
                for doc in docs:
                    st.markdown(f"- `{doc['filename']}`")
            else:
                st.write("No knowledge base files found.")
        with right:
            st.write("Documented data folders")
            st.code("data/knowledge_base\ndata/chroma_db\ndata/chatbot.db", language="text")
            st.caption("These folders match the document's required local project layout.")

with tab_history:
    with st.container(border=True):
        st.markdown('<div class="section-title">Recent Stored History</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Conversation history is stored in SQLite and reused for multi-turn context.</div>', unsafe_allow_html=True)
        if recent_history:
            for index, (user_msg, bot_msg) in enumerate(recent_history, start=1):
                with st.expander(f"Turn {index}"):
                    st.markdown(f"**User:** {user_msg}")
                    st.markdown(f"**Assistant:** {bot_msg}")
        else:
            st.write("No stored chat history yet.")
