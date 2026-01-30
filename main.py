import os
import sys

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Import our local modules
from llm_model import get_local_llm
from ingestion import get_retriever
from webSearch import get_web_search_tool

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_web_docs(docs):
    # Strictly truncate content to fit in TinyLlama's context text
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content[:500]}..." for doc in docs)

def main():
    print("--- Starting RAG System with Router (LCEL) ---")
    
    # 1. Initialize Components
    print("Initializing Reader, Searcher, and Generator...")
    retriever = get_retriever()
    llm = get_local_llm()
    
    # 2. Define Prompts
    
    # ROUTER PROMPT: Decides source
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
    
    # RAG PROMPT
    rag_template = """<|system|>
You are a helpful assistant. Answer based on the context provided.
Context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    # 3. Chains
    
    # Router Chain
    # We clean the output to ensure it's just the keyword
    router_chain = (
        router_prompt 
        | llm 
        | StrOutputParser() 
        | RunnableLambda(lambda x: x.strip().upper())
    )

    # Web Search Chain
    def run_web_search(query):
        print(f"   [Router] Routing to WEB SEARCH for: '{query}'")
        search_tool = get_web_search_tool()
        try:
            results = search_tool.invoke({"query": query})
            # Adapt tool output to list of documents format expected by formatter
            from langchain_core.documents import Document
            docs = []
            for res in results:
                 docs.append(Document(page_content=res.get('content'), metadata={'source': res.get('url')}))
            return docs
        except Exception as e:
            print(f"   [Error] Web Search failed: {e}")
            return []

    web_chain = (
        {"context": RunnableLambda(lambda x: run_web_search(x["question"])) | format_web_docs, "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Local RAG Chain
    def run_local_retrieval(query):
        print(f"   [Router] Routing to LOCAL ARCHIVE for: '{query}'")
        return query

    local_chain = (
        {"context": RunnableLambda(lambda x: run_local_retrieval(x["question"])) | retriever | format_docs, "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Main Branch Logic
    # If router says "WEB", go to web_chain
    # Else (default), go to local_chain
    full_chain = RunnableBranch(
        (lambda x: "WEB" in x["topic"], web_chain),
        local_chain
    )

    # Wrapper to run router first then pass topic + question to branch
    final_chain = {
        "topic": router_chain,
        "question": RunnablePassthrough()
    } | full_chain

    print("\nSystem Ready! Type 'exit' to quit.\n")
    
    # 4. Interactive Loop
    while True:
        try:
            query = input("User: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            if not query.strip():
                continue
            
            print("Assistant: ", end="", flush=True)
            
            # Since TinyLlama might be slow, we just invoke() to keep it simple, or stream if possible.
            # We'll stream the final output.
            response = final_chain.invoke(query)
            print(response)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
