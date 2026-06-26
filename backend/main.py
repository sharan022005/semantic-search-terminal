import os
import re
import uuid
import datetime
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from backend import config, database, parser, embedder

app = FastAPI(
    title="Semantic Search Backend",
    description="FastAPI service storing documents in MongoDB & ChromaDB with semantic search",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token normalization and synonym expansion helper functions
def normalize_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ing") and len(token) > 4:
        return token[:-3]
    if token.endswith("ed") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 2:
        return token[:-1]
    return token

def expand_query_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for token in tokens:
        syns = config.SYNONYMS.get(token, [])
        expanded.extend(normalize_token(s) for s in syns)
    return expanded

# Pydantic schemas
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    document_id: Optional[str] = None

class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    filename: str
    score: float
    match_type: str  # "semantic" or "lexical"
    parent_text: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    semantic_matches_found: bool
    is_fallback: Optional[bool] = False

class DocumentMetadataResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    upload_time: str
    status: str

@app.get("/", tags=["System"])
async def root_index():
    return {
        "message": "Semantic Search Backend API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.post("/upload", response_model=DocumentMetadataResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    try:
        # 1. Read file bytes
        file_bytes = await file.read()
        
        # 2. Extract pages (contains page_num and text)
        pages = parser.extract_pages(file_bytes, filename)
        if not pages or all(not p["text"].strip() for p in pages):
            raise HTTPException(status_code=400, detail="Document contains no readable text.")
            
        def get_text_hash(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
            
        # 3. Check if document already exists in MongoDB
        existing_doc = await database.documents_metadata.find_one({"filename": filename})
        
        if existing_doc:
            doc_id = existing_doc["_id"]
            # Retrieve existing page list
            # Older uploads might not have "pages" metadata list, default to empty list
            existing_pages = existing_doc.get("pages", [])
            existing_pages_dict = {p["page_num"]: p for p in existing_pages}
            
            updated_pages = []
            total_chunk_count = 0
            
            # Loop through new pages and determine updates
            for page in pages:
                page_num = page["page_num"]
                page_text = page["text"]
                page_hash = get_text_hash(page_text)
                
                # Check if page is unchanged
                if page_num in existing_pages_dict and existing_pages_dict[page_num]["hash"] == page_hash:
                    # Page is unchanged: keep existing metadata and do not re-embed
                    updated_pages.append(existing_pages_dict[page_num])
                    total_chunk_count += len(existing_pages_dict[page_num]["chunk_ids"])
                    print(f"Page {page_num} unchanged for {filename}. Skipping re-embedding.")
                else:
                    # Page changed or is new:
                    # Delete old chunks for this page from ChromaDB if they exist
                    if page_num in existing_pages_dict:
                        old_chunk_ids = existing_pages_dict[page_num]["chunk_ids"]
                        if old_chunk_ids:
                            try:
                                database.chroma_collection.delete(ids=old_chunk_ids)
                            except Exception:
                                # Fallback deletion by metadata if ids fail or collection structure changed
                                try:
                                    database.chroma_collection.delete(
                                        where={
                                            "$and": [
                                                {"document_id": doc_id},
                                                {"page_num": page_num}
                                            ]
                                        }
                                    )
                                except Exception:
                                    pass
                    
                    # Chunk the page text
                    page_chunks = embedder.chunk_pages_parent_child([page])
                    page_chunk_count = len(page_chunks)
                    
                    if page_chunk_count > 0:
                        # Generate embeddings
                        embeddings = [embedder.get_embedding(item["child_text"]) for item in page_chunks]
                        
                        # Generate unique chunk IDs
                        chroma_ids = [f"{doc_id}_p{page_num}_c{i}_{str(uuid.uuid4())[:8]}" for i in range(page_chunk_count)]
                        chroma_metadatas = [
                            {
                                "document_id": doc_id,
                                "filename": filename,
                                "page_num": page_num,
                                "chunk_index": i,
                                "parent_text": item["parent_text"],
                                "parent_index": item["parent_index"]
                            } for i, item in enumerate(page_chunks)
                        ]
                        
                        database.chroma_collection.add(
                            ids=chroma_ids,
                            embeddings=embeddings,
                            metadatas=chroma_metadatas,
                            documents=[item["child_text"] for item in page_chunks]
                        )
                        
                        updated_pages.append({
                            "page_num": page_num,
                            "hash": page_hash,
                            "chunk_ids": chroma_ids
                        })
                        total_chunk_count += page_chunk_count
                        print(f"Page {page_num} modified/new for {filename}. Generated {page_chunk_count} chunks.")
                    else:
                        updated_pages.append({
                            "page_num": page_num,
                            "hash": page_hash,
                            "chunk_ids": []
                        })
            
            # Clean up deleted pages (if the new document is shorter than the old one)
            max_new_page_num = len(pages)
            for old_page_num, old_page_data in existing_pages_dict.items():
                if old_page_num > max_new_page_num:
                    old_chunk_ids = old_page_data["chunk_ids"]
                    if old_chunk_ids:
                        try:
                            database.chroma_collection.delete(ids=old_chunk_ids)
                        except Exception:
                            try:
                                database.chroma_collection.delete(
                                    where={
                                        "$and": [
                                            {"document_id": doc_id},
                                            {"page_num": old_page_num}
                                        ]
                                    }
                                )
                            except Exception:
                                pass
                        print(f"Removed page {old_page_num} chunks from ChromaDB for {filename}.")
            
            # Update MongoDB
            doc_meta = {
                "filename": filename,
                "chunk_count": total_chunk_count,
                "upload_time": datetime.datetime.utcnow().isoformat(),
                "status": "processed",
                "pages": updated_pages
            }
            await database.documents_metadata.update_one({"_id": doc_id}, {"$set": doc_meta})
            
        else:
            # Document does not exist: perform a complete new upload
            doc_id = str(uuid.uuid4())
            pages_meta_list = []
            total_chunk_count = 0
            
            all_chroma_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []
            
            for page in pages:
                page_num = page["page_num"]
                page_text = page["text"]
                page_hash = get_text_hash(page_text)
                
                # Chunk page
                page_chunks = embedder.chunk_pages_parent_child([page])
                page_chunk_count = len(page_chunks)
                
                page_chroma_ids = []
                if page_chunk_count > 0:
                    # Generate embeddings
                    embeddings = [embedder.get_embedding(item["child_text"]) for item in page_chunks]
                    chroma_ids = [f"{doc_id}_p{page_num}_c{i}_{str(uuid.uuid4())[:8]}" for i in range(page_chunk_count)]
                    chroma_metadatas = [
                        {
                            "document_id": doc_id,
                            "filename": filename,
                            "page_num": page_num,
                            "chunk_index": i,
                            "parent_text": item["parent_text"],
                            "parent_index": item["parent_index"]
                        } for i, item in enumerate(page_chunks)
                    ]
                    
                    all_chroma_ids.extend(chroma_ids)
                    all_embeddings.extend(embeddings)
                    all_metadatas.extend(chroma_metadatas)
                    all_documents.extend([item["child_text"] for item in page_chunks])
                    
                    page_chroma_ids = chroma_ids
                    total_chunk_count += page_chunk_count
                    
                pages_meta_list.append({
                    "page_num": page_num,
                    "hash": page_hash,
                    "chunk_ids": page_chroma_ids
                })
            
            # Batch upload to ChromaDB if there are chunks
            if all_chroma_ids:
                database.chroma_collection.add(
                    ids=all_chroma_ids,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                    documents=all_documents
                )
            
            # Save metadata in MongoDB
            doc_meta = {
                "_id": doc_id,
                "filename": filename,
                "chunk_count": total_chunk_count,
                "upload_time": datetime.datetime.utcnow().isoformat(),
                "status": "processed",
                "pages": pages_meta_list
            }
            await database.documents_metadata.insert_one(doc_meta)
            print(f"Created new document {filename} with ID {doc_id} and {total_chunk_count} chunks.")
            
        return DocumentMetadataResponse(
            id=doc_id,
            filename=filename,
            chunk_count=total_chunk_count,
            upload_time=doc_meta["upload_time"],
            status=doc_meta["status"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

# Lazy load cross encoder
_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    query = request.query
    top_k = request.top_k or config.TOP_K
    score_threshold = request.score_threshold or config.SCORE_THRESHOLD
    
    try:
        # 1. Get query embedding
        query_vector = embedder.get_embedding(query)
        
        # 2. Construct metadata filters if document_id is provided
        where_clause = {}
        if request.document_id:
            where_clause["document_id"] = request.document_id
            
        # Retrieve more candidates than requested to allow effective re-ranking
        candidate_count = max(top_k * 3, 10)
        
        # 3. Search ChromaDB
        chroma_res = database.chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_count,
            where=where_clause if where_clause else None
        )
        
        results = []
        semantic_matches_found = False
        
        # ChromaDB results format: list of lists
        if chroma_res and chroma_res["documents"] and len(chroma_res["documents"][0]) > 0:
            documents = chroma_res["documents"][0]
            metadatas = chroma_res["metadatas"][0]
            ids = chroma_res["ids"][0]
            
            # 4. Perform Cross-Encoder Re-ranking
            cross_encoder = get_cross_encoder()
            pairs = [[query, doc] for doc in documents]
            cross_scores = cross_encoder.predict(pairs)
            
            import numpy as np
            candidates = []
            for i in range(len(documents)):
                # Map raw logit output to [0, 1] using standard sigmoid
                raw_score = float(cross_scores[i])
                sigmoid_score = 1.0 / (1.0 + np.exp(-raw_score))
                
                candidates.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "document_id": metadatas[i]["document_id"],
                    "filename": metadatas[i]["filename"],
                    "score": round(sigmoid_score, 4),
                    "parent_text": metadatas[i].get("parent_text", ""),
                    "match_type": "semantic"
                })
                
            # Sort by Cross-Encoder score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Print debug information to the terminal
            print(f"\n--- DEBUG SEARCH RESULTS FOR QUERY: '{query}' ---")
            print(f"Total candidates retrieved from ChromaDB: {len(candidates)}")
            for idx, item in enumerate(candidates):
                print(f"Rank {idx+1}: Score={item['score']} | Doc={item['filename']} | ChunkID={item['chunk_id']}")
                print(f"   Text snippet: {item['text'][:100]}...")
            print("--------------------------------------------------\n")
            
            is_fallback = False

            # Filter by threshold and format results
            for item in candidates:
                if item["score"] >= score_threshold:
                    semantic_matches_found = True
                    results.append(SearchResultItem(
                        chunk_id=item["chunk_id"],
                        text=item["text"],
                        document_id=item["document_id"],
                        filename=item["filename"],
                        score=item["score"],
                        match_type="semantic",
                        parent_text=item["parent_text"]
                    ))
            
            # Limit to top_k
            results = results[:top_k]

            # Fallback logic if no results met the threshold
            if not results and candidates:
                is_fallback = True
                fallback_candidates = candidates[:top_k]
                for item in fallback_candidates:
                    results.append(SearchResultItem(
                        chunk_id=item["chunk_id"],
                        text=item["text"],
                        document_id=item["document_id"],
                        filename=item["filename"],
                        score=item["score"],
                        match_type="semantic",
                        parent_text=item["parent_text"]
                    ))
                    
        return SearchResponse(
            query=query,
            results=results,
            semantic_matches_found=semantic_matches_found,
            is_fallback=is_fallback
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/documents/{document_id}/chunks", tags=["Documents"])
async def get_document_chunks(document_id: str):
    try:
        # Retrieve chunks from ChromaDB matching document_id
        res = database.chroma_collection.get(
            where={"document_id": document_id}
        )
        
        if not res or not res["ids"]:
            return []
            
        chunks = []
        for i in range(len(res["ids"])):
            chunks.append({
                "chunk_id": res["ids"][i],
                "text": res["documents"][i],
                "chunk_index": res["metadatas"][i]["chunk_index"],
                "parent_text": res["metadatas"][i].get("parent_text", "")
            })
            
        # Sort chunks by chunk_index
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document chunks: {str(e)}")

@app.get("/documents", response_model=List[DocumentMetadataResponse], tags=["Documents"])
async def list_documents():
    try:
        cursor = database.documents_metadata.find()
        docs = []
        async for doc in cursor:
            docs.append(DocumentMetadataResponse(
                id=doc["_id"],
                filename=doc["filename"],
                chunk_count=doc["chunk_count"],
                upload_time=doc["upload_time"],
                status=doc["status"]
            ))
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    try:
        # 1. Delete from MongoDB
        delete_res = await database.documents_metadata.delete_one({"_id": document_id})
        if delete_res.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        # 2. Delete from ChromaDB
        database.chroma_collection.delete(where={"document_id": document_id})
        
        return {"status": "success", "message": f"Document {document_id} and all its chunks successfully deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
