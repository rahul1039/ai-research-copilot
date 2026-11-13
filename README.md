# 📚 AI Research Copilot
### Confidence-Aware Retrieval-Augmented Generation (RAG) for Structured Document Intelligence

AI Research Copilot is an intelligent document-question-answering system that allows users to upload multiple PDF documents and retrieve relevant information through natural-language queries.

The system combines semantic search, vector-based retrieval, and Large Language Model (LLM)-powered answer generation to provide context-grounded responses. It also includes an answer-verification mechanism to assess whether generated responses are supported by the retrieved document content.

## 🚀 Key Features

- **Multi-PDF Upload:** Upload and process multiple PDF documents through a Streamlit interface.
- **Document Ingestion:** Extract document content using PyPDFLoader.
- **Intelligent Chunking:** Split documents into manageable text chunks with configurable overlap.
- **Semantic Search:** Generate vector embeddings using Sentence Transformers.
- **Vector Database:** Store and retrieve document embeddings using FAISS.
- **Context-Grounded Generation:** Generate answers using retrieved document context and Mistral through Ollama.
- **Confidence Verification:** Evaluate generated answers against retrieved document content.
- **Hallucination Reduction:** Restrict responses to the supplied document context and return a fallback when information is unavailable.
- **Interactive Interface:** Ask natural-language questions and review answers with confidence indicators.

## 🏗️ System Architecture

```text
             PDF Documents
                   |
                   ▼
           Document Ingestion
              (PyPDFLoader)
                   |
                   ▼
            Text Chunking
         (Recursive Splitter)
                   |
                   ▼
          Embedding Generation
       (all-MiniLM-L6-v2)
                   |
                   ▼
             FAISS Index
                   |
                   ▼
          Semantic Retrieval
                   |
                   ▼
          Relevant Document
                Context
                   |
                   ▼
          Mistral LLM (Ollama)
                   |
                   ▼
            Answer Generation
                   |
                   ▼
          Confidence Verification
                   |
                   ▼
           Grounded Response
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Mistral |
| LLM Runtime | Ollama |
| Embedding Model | Sentence Transformers |
| Embedding Model Name | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Framework | LangChain |
| Document Processing | PyPDFLoader |

## 📂 Project Structure

```text
ai-research-copilot/
│
├── generation/
│   ├── answer_generator.py
│   ├── generator.py
│   └── prompt_templates.py
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
│
├── memory/
│   └── conversation_memory.py
│
├── retrieval/
│   ├── hybrid_search.py
│   ├── retriever.py
│   └── vector_store.py
│
├── app.py
├── streamlit_app.py
├── config.py
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rahul1039/-ai-research-copilot.git
cd -ai-research-copilot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Install Ollama from:

https://ollama.com/

Download the Mistral model:

```bash
ollama pull mistral
```

Ensure Ollama is running before launching the application.

### 5. Run the Application

Launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Alternatively, run the command-line application:

```bash
python app.py
```

## 💡 How It Works

1. Upload one or more PDF documents.
2. The system extracts text from the uploaded documents.
3. Extracted text is divided into smaller chunks.
4. The embedding model converts text chunks into vector representations.
5. FAISS indexes the vectors for similarity-based retrieval.
6. Relevant document chunks are retrieved based on the user's query.
7. Mistral generates an answer using the retrieved context.
8. A verification step evaluates whether the generated answer is supported by the retrieved content.
9. The application displays the response along with a confidence indicator.

## 🔍 Example Use Case

**Input:** Upload a research paper or technical PDF.

**Question:** What methodology was used in the research?

**Output:** A context-grounded response generated from the uploaded document.

If the requested information is unavailable in the retrieved document context, the system provides a fallback response instead of intentionally relying on outside knowledge.

## 🔮 Future Enhancements

- Advanced document parsing and information extraction.
- Improved hybrid retrieval and ranking strategies.
- Better confidence calibration and answer verification.
- Support for additional document formats.
- Enhanced conversational memory.
- Improved retrieval performance for large document collections.

## 👨‍💻 Author

**Rahul Kumar Parida**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

- GitHub: [@rahul1039](https://github.com/rahul1039)

---

⭐ If you find this project interesting, consider giving the repository a star!

**AI Research Copilot** — Making document research smarter through retrieval, semantic search, and grounded AI generation.
