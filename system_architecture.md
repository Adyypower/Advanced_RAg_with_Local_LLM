# Local RAG System Architecture & Code Explanation

## Overview
This project is a **Local Retrieval-Augmented Generation (RAG) System** that runs entirely on your machine. It combines a local Large Language Model (TinyLlama), a local Vector Database (ChromaDB), and a Router to intelligent decide whether to answer using local documents or fetch live information from the web (using Tavily).

## System Architecture

The system uses a **Router-based Architecture** implemented with **LangChain LCEL (LangChain Expression Language)**.

```mermaid
graph TD
    UserQuery[User Query] --> Router{Router (LLM)}
    Router -- "Topics: RAG, Docs, Technical" --> LocalBranch[Local RAG Chain]
    Router -- "Topics: Weather, News, General" --> WebBranch[Web Search Chain]
    
    subgraph "Local RAG Chain"
        LocalBranch --> Retriever[ChromaDB Retriever]
        Retriever --> Context[Retrieved Docs]
        Context --> Prompt[RAG Prompt]
        Prompt --> Generator[TinyLlama LLM]
    end
    
    subgraph "Web Search Chain"
        WebBranch --> Tavily[Web Search Tool]
        Tavily --> WebContext[Search Results]
        WebContext --> Prompt
        Prompt --> Generator
    end
    
    Generator --> FinalAnswer[Final Answer]
```

## Key Components

### 1. The Brain: TinyLlama (1.1B)
*   **File**: `llm_model.py`
*   **Why**: We use `TinyLlama-1.1B-Chat` because it is strictly small enough to run on a standard CPU with reasonable speed.
*   **Optimization**: 
    *   `return_full_text=False`: Prevents the model from repeating the prompt.
    *   `max_new_tokens=128`: Keeps responses concise to save time.
    *   `device`: Auto-detects CPU or CUDA (GPU).

### 2. The Storage: ChromaDB & Embeddings
*   **File**: `ingestion.py`
*   **Embeddings**: We use `sentence-transformers/all-MiniLM-L6-v2`. This model converts text into numbers (vectors). It's very fast and effective.
*   **Vector Store**: `ChromaDB` stores these vectors. When you ask a question, we search this database for the most similar vectors (text chunks).
*   **Ingestion Logic**:
    *   Documents are loaded from `docs/` folder using `DirectoryLoader`.
    *   Text is split into chunks of 1000 characters (overlap 200) to fit into the model's memory.

### 3. The Eyes: Web Search (Tavily)
*   **File**: `webSearch.py`
*   **Tool**: `TavilySearchResults`
*   **Optimization**: We limited `k=1` (top 1 result) and truncated content to 500 characters. This is crucial because TinyLlama has a small "context window" (2048 tokens). If we feed it too much text, it crashes or hallucinates.

### 4. The Logic: Main Router
*   **File**: `main.py` (and upcoming `app.py`)
*   **Router**: A specialized prompt asks the LLM to classify the input as `WEB` or `LOCAL`.
*   **Branching**: We use `RunnableBranch` to direct the flow. This is "ReAct-lite"—instead of a complex agent loop that might fail on small models, we have a deterministic definition of where valid answers come from.

## How It Works (Step-by-Step)

1.  **Ingestion**: You run the ingestion to load your `.txt` files into the database.
2.  **Query**: You ask "What is the RAG system?"
3.  **Routing**: 
    *   The Router sees "RAG system" and outputs `LOCAL`.
    *   The `LocalChain` activates.
4.  **Retrieval**: ChromaDB finds the definition of RAG you put in `test.txt`.
5.  **Generation**: TinyLlama receives: "Context: [Found definition...] Question: What is RAG?". It answers using that context.
6.  **Alternative**: You ask "Weather in Delhi".
    *   Router sees "Weather" -> `WEB`.
    *   `WebChain` activates, calls Tavily API, gets the temperature, and TinyLlama summarizes it.

## Revision & Code Maintenance

*   **To change the Model**: Edit `MODEL_ID` in `llm_model.py`.
*   **To change the Database**: Edit `CHROMA_DB_DIR` in `ingestion.py`.
*   **To debug Web Limits**: If responses get cut off, check the `format_web_docs` function in `main.py`/`app.py` and adjust the character limit.
