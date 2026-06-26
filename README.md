# Semantic Search Terminal

A full-stack Semantic Search application built with FastAPI and React. This application allows users to upload documents (PDFs, text files, images) and perform advanced semantic searches over their content using state-of-the-art sentence transformers and cross-encoder re-ranking.

## Features

- **Document Upload & Parsing:** Support for PDF parsing (via PyMuPDF) and OCR for images (via EasyOCR).
- **Semantic Search:** Fast and accurate vector-based search using ChromaDB.
- **Re-ranking:** Cross-Encoder models (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-rank initial vector search results for higher precision.
- **Parent-Child Chunking:** Intelligent text chunking strategies to preserve context while searching.
- **Document Management:** View uploaded documents, their chunk counts, and processing status.

## Tech Stack

### Backend
- **Framework:** FastAPI (Python)
- **Vector Database:** ChromaDB
- **Metadata Database:** MongoDB (via Motor/PyMongo)
- **ML / Embeddings:** `sentence-transformers`, `torch`
- **Parsing:** `PyMuPDF`, `easyocr`

### Frontend
- **Framework:** React 19 + Vite
- **Styling:** CSS
- **Linting:** ESLint

## Project Structure

```
semantic-search-terminal/
├── backend/                  # FastAPI backend
│   ├── main.py               # API Endpoints (Upload, Search, Delete)
│   ├── config.py             # Configuration and environment variables
│   ├── database.py           # MongoDB and ChromaDB connection
│   ├── embedder.py           # Text chunking and embedding generation
│   ├── parser.py             # Document parsing (PDF, OCR, etc.)
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React frontend
│   ├── src/                  # Components and styles
│   ├── public/               # Static assets
│   ├── index.html            # Entry HTML
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite config
├── documents/                # Sample documents for testing
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **MongoDB** (Ensure MongoDB is running locally or provide a connection string in `backend/config.py`)

## Installation & Setup

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
```

Create a virtual environment and activate it:

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

The backend API will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

### 2. Frontend Setup

Open a new terminal and navigate to the `frontend` directory:

```bash
cd frontend
```

Install the Node dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend application will be available at `http://localhost:5173`.

## API Endpoints Overview

- `GET /health` - System health check.
- `POST /upload` - Upload and process a new document.
- `POST /search` - Perform a semantic search query with cross-encoder re-ranking.
- `GET /documents` - List all uploaded documents.
- `GET /documents/{document_id}/chunks` - Retrieve all chunks for a specific document.
- `DELETE /documents/{document_id}` - Delete a document and its embeddings.

## License
MIT
