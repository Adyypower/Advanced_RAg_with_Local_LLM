# 🤖 Local RAG with Web Search & TinyLlama

A powerful **Local Retrieval-Augmented Generation (RAG) System** that runs entirely on your machine. It intelligently routes queries between a **local knowledge base** (powered by ChromaDB) and **live web search** (powered by Tavily), using **TinyLlama-1.1B** as the brain.

Built with **LangChain (LCEL)** and **Streamlit**.

![Architecture](system_architecture.md)

## ✨ Features

- **🔒 Fully Local LLM**: Runs on `TinyLlama-1.1B-Chat`, optimized for consumer hardware (CPU/GPU).
- **🧠 Intelligent Routing**: Automatically decides if a question needs local documents or live web info.
- **📂 Document Ingestion**: Supports `.txt` and `.pdf` uploads to build your personal knowledge base.
- **🌐 Live Code/Web Search**: Fetches real-time info using Tavily API for questions about news, weather, etc.
- **⚡ Fast Retrieval**: Uses `ChromaDB` and `all-MiniLM-L6-v2` embeddings for millisecond-latency searches.
- **🖥️ Dual Interface**:
    - **Web UI**: Interactive Chat Interface using Streamlit.
    - **CLI**: Terminal-based chat for quick testing.

## 🛠️ Tech Stack

- **LLM**: TinyLlama-1.1B-Chat-v1.0 (via HuggingFace)
- **Framework**: LangChain (LCEL), Streamlit
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Web Search**: Tavily API

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Adyypower/Advanced_RAg_with_Local_LLM.git
    cd Advanced_RAg_with_Local_LLM
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your keys:
    ```ini
    TAVILY_API_KEY=your_tavily_api_key_here
    # HUGGINGFACE_API_KEY is optional if model is cached locally
    ```

## 📖 Usage

### 1. Run the Web Interface (Recommended)
Launch the full Chat UI:
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.
- **Upload Files**: Use the sidebar to upload `.txt` or `.pdf` documents.
- **Chat**: Ask questions. The system will auto-route:
    - *"What is RAG?"* -> **Local Knowledge Base**
    - *"Weather in New York?"* -> **Web Search**

### 2. Run CLI Mode
For quick testing in the terminal:
```bash
python main.py
```

## 📂 Project Structure

```
├── app.py                 # Streamlit Web Application
├── main.py                # CLI Application & Logic
├── llm_model.py           # TinyLlama Model Loader
├── ingestion.py           # Document Processing & ChromaDB Logic
├── webSearch.py           # Tavily Web Search Tool
├── system_architecture.md # Detailed System Diagrams
├── docs/                  # Directory for uploaded documents
└── requirements.txt       # Python Dependencies
```

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
