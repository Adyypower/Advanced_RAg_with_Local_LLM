import streamlit as st
import os
import sys

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Import local modules
from llm_model import get_local_llm
from ingestion import get_retriever, ingest_from_strings, ingest_from_directory
from webSearch import get_web_search_tool

# --- Page Config ---
st.set_page_config(page_title="Local RAG Chat", layout="wide")

st.title("🤖 Local RAG System")
st.markdown("Interacts with **Local Documents** or **Web Search** automatically.")

# --- Sidebar: Ingestion & Config ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    
    # 1. File Uploader
    uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)
    
    if st.button("Process & Ingest Files"):
        if uploaded_files:
            # Save uploaded files to docs/ temp directory
            if not os.path.exists("docs"):
                os.makedirs("docs")
            
            for uploaded_file in uploaded_files:
                # Determine path
                file_path = os.path.join("docs", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.info("Files saved. Ingesting...")
            # Run ingestion
            ingest_from_directory()
            st.success("Ingestion Complete! Vector DB updated.")
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    
    # 2. Paragraph Input
    st.subheader("📝 Add Quick Text")
    user_text = st.text_area("Paste text/paragraph here:")
    if st.button("Ingest Text"):
        if user_text.strip():
            st.info("Ingesting text...")
            ingest_from_strings([user_text])
            st.success("Text added to Knowledge Base.")
        else:
            st.warning("Text area cannot be empty.")

# --- Main App Logic (Router) ---

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RAG Logic Setup ---

@st.cache_resource
def load_resources():
    """
    Load heavy resources (LLM, Retriever, Search Tool).
    Cached to prevent reloading on every interaction.
    """
    print("Loading Resources...")
    retriever = get_retriever()
    llm = get_local_llm()
    return retriever, llm

def setup_rag_chain(retriever, llm):
    """
    Sets up the LCEL chain. 
    """
    print("Building Chains...")
    
    # Helper formatters
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_web_docs(docs):
        # Strictly truncate content
        return "\n\n".join(f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content[:500]}..." for doc in docs)

    # 1. Router Pormpt
    router_template = """<|system|>
You are an intelligent router. Your job is to classify the user's question into one of two categories: 'WEB' or 'LOCAL'.

- Use 'WEB' if the question is about weather, temperature, news, current events, or general world facts.
- Use 'LOCAL' if the question is about 'RAG', 'Retrieval Augmented Generation', 'this system', 'documents', or specific technical details in the database.

Do not output anything else. Just 'WEB' or 'LOCAL'.</s>
<|user|>
{question}</s>
<|assistant|>
"""
    router_prompt = PromptTemplate.from_template(router_template)
    
    # 2. Router Chain
    # Fix: Explicitly wrap input in dict to avoid ambiguity
    router_chain = (
        {"question": RunnablePassthrough()}
        | router_prompt 
        | llm 
        | StrOutputParser() 
        | RunnableLambda(lambda x: x.strip().upper())
    )
    
    # 3. RAG/Answer Prompt
    rag_template = """<|system|>
You are a helpful assistant. Answer based on the context provided.
Context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    # 4. Branches
    # Web Branch
    def run_web_search(query):
        search_tool = get_web_search_tool()
        try:
            results = search_tool.invoke({"query": query})
            docs = []
            for res in results:
                 docs.append(Document(page_content=res.get('content'), metadata={'source': res.get('url')}))
            return docs
        except:
            return []

    web_chain = (
        {"context": RunnableLambda(lambda x: run_web_search(x["question"])) | format_web_docs, "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Local Branch
    local_chain = (
        {"context": retriever | format_docs, "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 5. Full Branching Logic
    # We pass the full dict from final_chain key-mapping to this branch
    full_chain = RunnableBranch(
        (lambda x: "WEB" in x["topic"], web_chain),
        local_chain # Default to local
    )

    # Master Chain
    final_chain = {
        "topic": router_chain,
        "question": RunnablePassthrough()
    } | full_chain
    
    return final_chain

# Load resources once
retriever, llm = load_resources()

# Build chain (lightweight)
rag_chain = setup_rag_chain(retriever, llm)

# --- Chat Input Handler ---
if prompt := st.chat_input("Ask something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Checking Router)"):
            try:
                # We can stream this easily
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
