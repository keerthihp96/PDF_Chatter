# app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

load_dotenv()

# ── Caching — skip LLM call for repeated questions ────────────────────────────
set_llm_cache(InMemoryCache())

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Chat with your PDF")
st.caption("Powered by Groq LLaMA 3.3 + LangChain + FAISS — 100% Free")


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat display history

if "memory" not in st.session_state:
    st.session_state.memory = InMemoryChatMessageHistory()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "embeddings" not in st.session_state:
    try:
    # Load once and cache — avoids re-downloading model on every interaction
        with st.spinner("🔄 Loading embedding model (one-time setup ~30s)..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {str(e)}")
        st.stop()



# ── LLM Setup ─────────────────────────────────────────────────────────────────
def get_llm():
    """Returns Groq LLM with fallback model."""
    primary = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
        max_retries=3
    )
    fallback = ChatGroq(
        model="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
        max_retries=2
    )
    return primary.with_fallbacks([fallback])


# ── PDF Processing ────────────────────────────────────────────────────────────
def process_pdf(uploaded_file) -> int:
    """
    Full RAG pipeline:
    1. Load PDF pages
    2. Split into chunks
    3. Embed with HuggingFace
    4. Store in FAISS
    Returns: total number of chunks created
    """

    # Save to temp file — PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Step 1 — Load PDF
    loader    = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Step 3 — Build FAISS vector store
    st.session_state.vectorstore = FAISS.from_documents(
        chunks,
        st.session_state.embeddings
    )

    # Clean up temp file
    os.unlink(tmp_path)

    return len(chunks)


# ── RAG Answer Generation ─────────────────────────────────────────────────────
def get_answer(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from FAISS
    2. Build prompt with context + memory
    3. Get answer from Groq LLM
    Returns: { "answer": str, "sources": list }
    """

    # Step 1 — Retrieve top 4 relevant chunks
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    source_docs = retriever.invoke(question)

    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?') + 1}]:\n{doc.page_content}"
        for doc in source_docs
    ])

    # Step 2 — Build prompt with system instructions
    system_prompt = f"""You are a helpful assistant that answers questions 
based strictly on the provided PDF content.

Rules:
- Answer ONLY from the PDF context provided below
- If the answer is not in the context, say 
  "I couldn't find that in the PDF"
- Always mention which page(s) your answer comes from
- Be concise and conversational
- Never make up information

PDF Context:
{context}
"""

    # Build messages with conversation history
    messages = [SystemMessage(content=system_prompt)]

    # Add last 6 messages from memory for follow-up support
    for msg in st.session_state.memory.messages[-6:]:
        messages.append(msg)

    # Add current question
    messages.append(HumanMessage(content=question))

    # Step 3 — Get answer from LLM
    llm      = get_llm()
    response = llm.invoke(messages)
    answer   = response.content.strip()

    # Save to memory
    st.session_state.memory.add_user_message(question)
    st.session_state.memory.add_ai_message(answer)

    return {
        "answer":  answer,
        "sources": source_docs
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF — textbook, report, contract, research paper"
    )

    if uploaded_file:
        # Only reprocess if new PDF uploaded
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("📖 Reading and indexing PDF..."):
                num_chunks = process_pdf(uploaded_file)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.memory   = InMemoryChatMessageHistory()

            st.success(f"✅ Ready! Created {num_chunks} chunks")

    # Show active PDF info
    if st.session_state.pdf_name:
        st.info(f"📄 **{st.session_state.pdf_name}**")

    # Clear chat button
    if st.session_state.vectorstore:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory   = InMemoryChatMessageHistory()
            st.rerun()

    st.divider()

    # Show source pages toggle
    show_sources = st.toggle("📖 Show Source Pages", value=True)

    st.divider()

    # How it works
    st.markdown("**⚙️ How it works:**")
    st.markdown("1. PDF split into 1000 char chunks")
    st.markdown("2. Chunks embedded with MiniLM")
    st.markdown("3. Stored in FAISS locally")
    st.markdown("4. Question retrieves top 4 chunks")
    st.markdown("5. Groq LLaMA answers from chunks")
    st.markdown("6. Memory keeps chat context")

    st.divider()
    st.markdown("**🆓 Free Stack:**")
    st.markdown("- LLM : Groq LLaMA 3.3 70B")
    st.markdown("- Embed: HuggingFace MiniLM")
    st.markdown("- Store: FAISS (local)")


# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload a PDF from the sidebar to get started!")
    st.markdown("### 💡 You can ask things like:")
    st.markdown("- *What is this document about?*")
    st.markdown("- *Summarize the key points*")
    st.markdown("- *What does page 3 say about X?*")
    st.markdown("- *What are the main conclusions?*")
    st.stop()

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source pages for past assistant messages
        if (msg["role"] == "assistant"
                and show_sources
                and msg.get("sources")):
            with st.expander("📖 Source Pages Used"):
                seen = []
                for doc in msg["sources"]:
                    page = doc.metadata.get("page", "?")
                    if page not in seen:
                        seen.append(page)
                        st.markdown(f"**Page {int(page) + 1}:**")
                        st.caption(
                            doc.page_content[:300].strip() + "..."
                        )
                        st.divider()

# Chat input
if user_input := st.chat_input("Ask anything about your PDF..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role":    "user",
        "content": user_input
    })

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            try:
                result  = get_answer(user_input)
                answer  = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                # Show source pages
                if show_sources and sources:
                    with st.expander("📖 Source Pages Used"):
                        seen = []
                        for doc in sources:
                            page = doc.metadata.get("page", "?")
                            if page not in seen:
                                seen.append(page)
                                st.markdown(f"**Page {int(page) + 1}:**")
                                st.caption(
                                    doc.page_content[:300].strip() + "..."
                                )
                                st.divider()

            except Exception as e:
                answer  = f"❌ Error: {str(e)}"
                sources = []
                st.error(answer)

    # Save to display history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources
    })