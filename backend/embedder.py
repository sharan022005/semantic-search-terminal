import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend import config

# Load model lazily to speed up module import and startup
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.MODEL_NAME)
    return _model

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    doc_chunks = splitter.split_text(text)
    return [c.replace("\n", " ").strip() for c in doc_chunks if c.strip()]

def chunk_pages_parent_child(pages: list[dict]) -> list[dict]:
    # 1. Create Parent chunks (larger window size for context retrieval)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 2. Create Child chunks (smaller window size for precise vector matching)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_data = []
    parent_global_index = 0
    
    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]
        
        parent_texts = parent_splitter.split_text(page_text)
        for p_idx, parent_text in enumerate(parent_texts):
            parent_text_cleaned = parent_text.replace("\n", " ").strip()
            if not parent_text_cleaned:
                continue
            child_texts = child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                child_text_cleaned = child_text.replace("\n", " ").strip()
                if child_text_cleaned:
                    chunks_data.append({
                        "child_text": child_text_cleaned,
                        "parent_text": parent_text_cleaned,
                        "parent_index": parent_global_index,
                        "child_index": c_idx,
                        "page_num": page_num
                    })
            parent_global_index += 1
            
    return chunks_data

def chunk_document_parent_child(text: str) -> list[dict]:
    return chunk_pages_parent_child([{"page_num": 1, "text": text}])

def get_embedding(text: str) -> list[float]:
    model = get_model()
    # Return embedding as a standard float list for compatibility with ChromaDB
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()
