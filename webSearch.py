import os
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

# Load environment variables (expecting TAVILY_API_KEY)
load_dotenv()

def get_web_search_tool():
    """
    Initializes and returns the Tavily search tool.
    """
    # k=3 means it will return top 3 search results
    return TavilySearchResults(k=1)

def perform_web_search(query: str) -> List[Document]:
    """
    Performs a web search using Tavily and returns the results as a list of Documents.
    
    Args:
        query (str): The search query.
        
    Returns:
        List[Document]: A list of documents containing the search results.
                        Each document has 'page_content' and metadata (url, title).
    """
    tool = get_web_search_tool()
    
    print(f"--- Searching Web for: '{query}' ---")
    try:
        # tool.invoke returns a list of dictionaries: [{'url': ..., 'content': ...}]
        results = tool.invoke({"query": query})
        
        documents = []
        for res in results:
            # Create a Document for each result
            doc = Document(
                page_content=res.get("content", ""),
                metadata={
                    "source": res.get("url", ""),
                    "title": res.get("title", "No Title")
                }
            )
            documents.append(doc)
            
        print(f"Found {len(documents)} results from web.")
        return documents
        
    except Exception as e:
        print(f"Error during web search: {e}")
        return []

if __name__ == "__main__":
    # Test the search
    search_query = "What is the latest version of LangChain?"
    docs = perform_web_search(search_query)
    
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
