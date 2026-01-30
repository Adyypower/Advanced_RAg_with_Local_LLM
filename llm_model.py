import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Configuration
# Using TinyLlama because it's small (1.1B), fast, and good for local testing.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_DIR = "./model_cache"

def get_local_llm():
    """
    Initializes and returns a local HuggingFace LLM pipeline compatible with LangChain.
    
    Features:
    - Downloads model to './model_cache' on first run.
    - Uses 'cpu' or 'cuda' automatically.
    - Configured for text generation.
    """
    
    system_device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Initializing Local LLM ({MODEL_ID}) on {device_name} ---")
    print(f"Model cache directory: {os.path.abspath(CACHE_DIR)}")

    # 1. Load Tokenizer
    # We specify cache_dir so it saves to our local folder
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        cache_dir=CACHE_DIR,
        local_files_only=False # Allow download if not present
    )

    # 2. Load Model
    # We specify cache_dir so it saves to our local folder
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=False
    )

    # 3. Create Pipeline
    # strict_param_check=False allows passing extra generation params if needed
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=128, # Limit output length
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        device=system_device,
        return_full_text=False # Return only the generated response, not the prompt
    )

    # 4. Wrap in LangChain
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    print("Local LLM initialized successfully.")
    return local_llm

if __name__ == "__main__":
    # Test the LLM
    print("Testing Local LLM...")
    llm = get_local_llm()
    
    # TinyLlama Chat requires a specific prompt format
    query = "What are the benefits of running LLMs locally?"
    prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
    
    print(f"\nPrompt: {prompt}\n")
    
    # Simple invoke
    response = llm.invoke(prompt)
    print(f"Response:\n{response}")
