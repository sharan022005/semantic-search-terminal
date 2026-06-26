import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Mock MongoDB and ChromaDB before importing the app
with patch("motor.motor_asyncio.AsyncIOMotorClient"), \
     patch("chromadb.PersistentClient") as mock_chroma_client:
    
    # Configure mock ChromaDB collection
    mock_collection = MagicMock()
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    
    from backend.main import app
    from backend import parser, embedder

client = TestClient(app)

def test_extract_text_txt():
    content = b"Hello world, this is a test."
    extracted = parser.extract_text(content, "test.txt")
    assert extracted == "Hello world, this is a test."

def test_chunk_text():
    content = "Hello " * 100  # Will create a large text
    chunks = embedder.chunk_text(content)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)

@patch("backend.database.documents_metadata")
@patch("backend.embedder.get_embedding")
def test_upload_endpoint(mock_get_embedding, mock_mongo_col):
    # Mock embedder and mongo db
    mock_get_embedding.return_value = [0.1] * 384
    mock_mongo_col.insert_one = AsyncMock()
    
    file_content = b"This is some document text to test upload."
    response = client.post(
        "/upload",
        files={"file": ("test_doc.txt", file_content, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.txt"
    assert data["chunk_count"] > 0
    assert data["status"] == "processed"
    assert "id" in data

@patch("backend.main.get_cross_encoder")
@patch("backend.database.documents_metadata")
@patch("backend.embedder.get_embedding")
def test_search_endpoint(mock_get_embedding, mock_mongo_col, mock_get_cross_encoder):
    mock_get_embedding.return_value = [0.1] * 384
    
    # Mock CrossEncoder predict to avoid HuggingFace model download during tests
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [2.0]  # logit 2.0 corresponds to ~0.88 similarity score
    mock_get_cross_encoder.return_value = mock_encoder
    
    # Mock chroma query return format
    mock_query_res = {
        "documents": [["This is a matching chunk text."]],
        "distances": [[0.1]],
        "metadatas": [[{"document_id": "doc_123", "filename": "doc.txt", "chunk_index": 0, "parent_text": "Parent text context"}]],
        "ids": [["doc_123_chunk_0"]]
    }
    
    with patch("backend.database.chroma_collection.query", return_value=mock_query_res):
        response = client.post(
            "/search",
            json={"query": "test query", "top_k": 1, "score_threshold": 0.1}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "This is a matching chunk text."
        assert data["results"][0]["match_type"] == "semantic"
        assert data["results"][0]["parent_text"] == "Parent text context"
        assert data["semantic_matches_found"] is True

@patch("backend.database.documents_metadata")
def test_list_documents(mock_mongo_col):
    # Mock MongoDB async cursor
    mock_cursor = MagicMock()
    mock_docs = [
        {"_id": "1", "filename": "a.txt", "chunk_count": 2, "upload_time": "2026-06-15", "status": "processed"},
        {"_id": "2", "filename": "b.txt", "chunk_count": 3, "upload_time": "2026-06-15", "status": "processed"}
    ]
    
    # Implement async iterator mock
    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.cursor = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.cursor >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.cursor]
            self.cursor += 1
            return item
            
    mock_mongo_col.find.return_value = AsyncIterator(mock_docs)
    
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["filename"] == "a.txt"
    assert data[1]["filename"] == "b.txt"

def test_get_document_chunks():
    mock_chroma_get_res = {
        "ids": ["doc_123_chunk_1", "doc_123_chunk_0"],
        "documents": ["Chunk 2 content", "Chunk 1 content"],
        "metadatas": [{"chunk_index": 1}, {"chunk_index": 0}]
    }
    
    with patch("backend.database.chroma_collection.get", return_value=mock_chroma_get_res):
        response = client.get("/documents/doc_123/chunks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Verify sorting by chunk_index
        assert data[0]["chunk_index"] == 0
        assert data[0]["text"] == "Chunk 1 content"
        assert data[1]["chunk_index"] == 1
        assert data[1]["text"] == "Chunk 2 content"
