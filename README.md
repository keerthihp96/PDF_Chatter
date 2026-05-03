# 📄 Chat with PDF
> Ask questions about any PDF in plain English — powered by **Groq LLaMA 3.3 70B**, **LangChain**, and **FAISS**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-v1-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Demo

Upload any PDF → Ask questions → Get instant answers with source pages cited.

---

## 🧠 Architecture Overview

```mermaid
graph TD
    A[👤 User] -->|Uploads PDF| B[Streamlit UI]
    A -->|Asks Question| B
    B -->|Loads PDF| C[PyPDFLoader]
    C -->|Raw Pages| D[RecursiveTextSplitter]
    D -->|1000 char chunks| E[HuggingFace MiniLM Embeddings]
    E -->|Vectors| F[(FAISS Vector Store)]
    B -->|User Question| G[FAISS Retriever]
    F -->|Indexed Vectors| G
    G -->|Top 4 Relevant Chunks| H[Prompt Builder]
    H -->|Context + History + Question| I[Groq LLaMA 3.3 70B]
    I -->|Answer| J[Streamlit Chat UI]
    J -->|Displays| K[Answer + Source Pages]
    I -->|Saves to| L[InMemoryChatHistory]
    L -->|Past messages| H
```

---

## 🔄 RAG Pipeline

```mermaid
flowchart LR
    subgraph INDEXING ["📥 Indexing Phase (runs once on upload)"]
        A[PDF File] --> B[PyPDFLoader]
        B --> C[Text Splitter\n1000 chars / 200 overlap]
        C --> D[HuggingFace Embeddings\nall-MiniLM-L6-v2]
        D --> E[(FAISS\nVector Store)]
    end

    subgraph RETRIEVAL ["🔍 Retrieval Phase (runs on every question)"]
        F[User Question] --> G[Embed Question]
        G --> H[Similarity Search\nTop 4 chunks]
        E --> H
        H --> I[Relevant Context]
    end

    subgraph GENERATION ["💬 Generation Phase"]
        I --> J[Prompt Builder\nContext + Memory + Question]
        J --> K[Groq LLaMA 3.3 70B]
        K --> L[Natural Language Answer]
        L --> M[Source Pages Cited]
    end

    INDEXING --> RETRIEVAL --> GENERATION
```

---

## 💬 Conversation Memory Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Memory as InMemoryChatHistory
    participant FAISS as FAISS Store
    participant LLM as Groq LLaMA

    User->>UI: Ask Question 1
    UI->>FAISS: Retrieve top 4 chunks
    FAISS-->>UI: Relevant context
    UI->>LLM: System prompt + Context + Q1
    LLM-->>UI: Answer 1
    UI->>Memory: Save Q1 + Answer 1

    User->>UI: Ask Follow-up Question 2
    UI->>Memory: Load chat history
    Memory-->>UI: Q1 + Answer 1
    UI->>FAISS: Retrieve top 4 chunks
    FAISS-->>UI: Relevant context
    UI->>LLM: System prompt + Context + History + Q2
    LLM-->>UI: Answer 2 (aware of Q1)
    UI->>Memory: Save Q2 + Answer 2
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| 🖥️ UI | Streamlit | Chat interface |
| 🤖 LLM | Groq LLaMA 3.3 70B | Answer generation |
| 🔗 Framework | LangChain v1 | Orchestration |
| 📐 Embeddings | HuggingFace MiniLM | Text → Vectors |
| 🗄️ Vector Store | FAISS | Similarity search |
| 📄 PDF Loader | PyPDF | Extract text |
| 🧠 Memory | InMemoryChatHistory | Conversation context |
| ⚡ Caching | InMemoryCache | Skip repeated LLM calls |

---

## 📁 Project Structure

```
chat_with_pdf/
├── .env                  # API keys (never commit)
├── .env.example          # Template for API keys
├── .gitignore            # Excludes .env and .venv
├── pyproject.toml        # Dependencies
├── app.py                # Main Streamlit application
└── README.md             # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Free [Groq API key](https://console.groq.com)

### Step 1 — Clone the repo
```bash
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf
```

### Step 2 — Create virtual environment
```bash
uv venv --python 3.11
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```

### Step 3 — Install dependencies
```bash
uv sync
```

### Step 4 — Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` 🚀

---

## 💡 Usage

1. **Upload a PDF** using the sidebar file uploader
2. **Wait** for indexing to complete (shows chunk count)
3. **Ask questions** in plain English in the chat input
4. **View source pages** by expanding "📖 Source Pages Used"
5. **Ask follow-ups** — the app remembers conversation history
6. **Clear chat** using the sidebar button to start fresh

### Example Questions
```
"What is this document about?"
"Summarize the key points"
"What does it say about X on page 5?"
"What are the main conclusions?"
"Tell me more about the first point you mentioned"
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 Chat Interface | Full conversational UI with message history |
| 🧠 Memory | Remembers previous questions for follow-ups |
| 📖 Source Pages | Shows exactly which pages were used to answer |
| ⚡ LLM Caching | Skips LLM call for repeated questions |
| 🔄 Fallback Model | Switches to Mixtral if primary model fails |
| 🔒 Stays in PDF | Never answers outside the document context |
| ❌ Error Handling | Catches and displays errors gracefully |
| 🗑️ Clear Chat | Reset conversation while keeping PDF loaded |

---

## 🔒 Security

- API keys stored in `.env` — **never committed to git**
- `.env` is listed in `.gitignore`
- Use `.env.example` as a template with no real values

---

## 🚧 Limitations

- PDF must contain selectable text (scanned PDFs not supported)
- Vector store is in-memory — resets on app restart
- Free Groq tier has rate limits — may slow on heavy usage

---

## 🗺️ Future Roadmap

```mermaid
graph LR
    A[✅ Current\nChat with PDF] --> B[🔜 Multiple PDFs\nupload & switch]
    B --> C[🔜 Persistent Storage\nChromaDB]
    C --> D[🔜 Agentic PDF\nReAct Agent]
    D --> E[🔜 Multi-modal\nImages in PDF]
```



## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration framework
- [Groq](https://groq.com) — Ultra-fast free LLM inference
- [HuggingFace](https://huggingface.co) — Free embedding models
- [FAISS](https://faiss.ai) — Facebook AI similarity search
- [Streamlit](https://streamlit.io) — Python web UI framework

Author
Keerthi Vinukonda
LinkedIn Profile : https://www.linkedin.com/in/keerthi-v-4022a8263/
GitHub link : https://github.com/keerthihp96
