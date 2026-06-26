import { useState, useEffect, useRef } from 'react';

const BACKEND_URL = "http://127.0.0.1:8001";

function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(3);
  const [scoreThreshold, setScoreThreshold] = useState(0.35);
  const [documents, setDocuments] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [semanticMatchesFound, setSemanticMatchesFound] = useState(true);
  const [isFallback, setIsFallback] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [uploadText, setUploadText] = useState("");
  const [error, setError] = useState(null);
  const [selectedDocChunks, setSelectedDocChunks] = useState([]);
  const [selectedDocName, setSelectedDocName] = useState("");
  const [showChunksModal, setShowChunksModal] = useState(false);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [selectedDocFilter, setSelectedDocFilter] = useState("");
  const [expandedChunks, setExpandedChunks] = useState({});
  const [hasSearched, setHasSearched] = useState(false);
  
  const fileInputRef = useRef(null);

  // Fetch documents on load
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/documents`);
      if (!res.ok) throw new Error("Failed to fetch documents.");
      const data = await res.json();
      setDocuments(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend database.");
    }
  };

  const viewDocumentChunks = async (docId, filename) => {
    setIsLoadingChunks(true);
    setShowChunksModal(true);
    setSelectedDocName(filename);
    setSelectedDocChunks([]);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/documents/${docId}/chunks`);
      if (!res.ok) throw new Error("Failed to fetch document chunks.");
      const data = await res.json();
      setSelectedDocChunks(data);
    } catch (err) {
      console.error(err);
      setError("Failed to load document text chunks.");
    } finally {
      setIsLoadingChunks(false);
    }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);
    setExpandedChunks({});
    setIsFallback(false);
    try {
      const res = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: parseInt(topK),
          score_threshold: parseFloat(scoreThreshold),
          document_id: selectedDocFilter || null
        })
      });
      if (!res.ok) throw new Error("Search request failed.");
      const data = await res.json();
      setSearchResults(data.results);
      setSemanticMatchesFound(data.semantic_matches_found);
      setIsFallback(data.is_fallback || false);
      setHasSearched(true);
    } catch (err) {
      console.error(err);
      setError("Search query execution failed.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleMultipleFilesUpload = async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setIsUploading(true);
    setError(null);

    const files = Array.from(fileList);
    const totalFiles = files.length;
    let successCount = 0;
    let lastUploadedDoc = null;

    for (let i = 0; i < totalFiles; i++) {
      const file = files[i];
      setUploadText(`Processing file ${i + 1} of ${totalFiles}: ${file.name}...`);
      
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${BACKEND_URL}/upload`, {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const errDetail = await res.json();
          throw new Error(errDetail.detail || `Upload failed for ${file.name}`);
        }
        lastUploadedDoc = await res.json();
        successCount++;
      } catch (err) {
        console.error(err);
        setError(prev => prev ? `${prev}\n• ${err.message}` : `Upload failed:\n• ${err.message}`);
      }
    }

    await fetchDocuments();
    setUploadText("");
    setIsUploading(false);

    // Automatically open the chunks modal if a single file was uploaded successfully
    if (totalFiles === 1 && successCount === 1 && lastUploadedDoc) {
      if (lastUploadedDoc.id) {
        viewDocumentChunks(lastUploadedDoc.id, lastUploadedDoc.filename);
      }
    } else if (totalFiles > 1) {
      setUploadText(`Successfully processed ${successCount} of ${totalFiles} files.`);
      setTimeout(() => setUploadText(""), 4000);
    }
  };

  const handleDelete = async (docId) => {
    if (!confirm("Are you sure you want to delete this document and all its text vectors?")) return;
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/documents/${docId}`, {
        method: "DELETE"
      });
      if (!res.ok) throw new Error("Failed to delete document.");
      
      // Update states
      setDocuments(prev => prev.filter(d => d.id !== docId));
      setSearchResults(prev => prev.filter(r => r.document_id !== docId));
    } catch (err) {
      console.error(err);
      setError("Failed to delete the document.");
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const onFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleMultipleFilesUpload(e.target.files);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleMultipleFilesUpload(e.dataTransfer.files);
    }
  };

  const highlightText = (text, queryStr) => {
    if (!queryStr) return text;
    
    // Set of common English stopwords to exclude from highlighting
    const stopwords = new Set([
      "the", "and", "but", "for", "with", "about", "this", "that", 
      "you", "are", "was", "were", "has", "have", "had", "can", "will",
      "its", "not", "from", "into", "their", "then", "they", "them",
      "his", "her", "she", "him", "who", "whom", "whose", "which"
    ]);

    // Extract alphanumeric tokens of length > 2 and filter out stopwords
    const tokens = queryStr
      .toLowerCase()
      .split(/[^a-zA-Z0-9]+/)
      .filter(t => t.length > 2 && !stopwords.has(t));
      
    if (tokens.length === 0) return text;

    // Create case-insensitive regex for whole word matches or word prefixes
    const escapedTokens = tokens.map(t => t.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
    const pattern = new RegExp(`(\\b(?:${escapedTokens.join('|')})[a-zA-Z0-9]*)`, 'gi');
    const parts = text.split(pattern);

    return parts.map((part, i) => 
      pattern.test(part) ? <mark className="text-highlight" key={i}>{part}</mark> : part
    );
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <div className="logo-icon">🔍</div>
          <div className="logo-text">
            <h1>Semantic Engine</h1>
            <p>Vector Search Dashboard</p>
          </div>
        </div>
        <div>
          {error && (
            <div style={{
              background: 'var(--color-danger-bg)',
              color: 'var(--color-danger)',
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              fontSize: '0.85rem',
              border: '1px solid rgba(239, 68, 68, 0.2)'
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="main-content">
        
        {/* Left column: Upload and Document list */}
        <div className="left-panel">
          
          {/* Upload card */}
          <section className="glass-card upload-container">
            <h3>Add Documents</h3>
            <div 
              className="drag-drop-zone"
              onClick={triggerFileInput}
              onDragOver={onDragOver}
              onDrop={onDrop}
            >
              <div className="upload-icon">📤</div>
              <p style={{ fontWeight: 500, fontSize: '0.9rem' }}>Drag & drop documents here</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Supports TXT, PDF, and Images (PNG/JPG)</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={onFileChange} 
                className="file-input"
                accept=".txt,.pdf,.png,.jpg,.jpeg,.webp"
                multiple
              />
            </div>
            
            {isUploading && (
              <div>
                <div className="uploading-animation">
                  <div className="uploading-bar"></div>
                </div>
                <div className="status-text">{uploadText}</div>
              </div>
            )}
          </section>

          {/* Document list card */}
          <section className="glass-card">
            <div className="doc-list-header">
              <h3>Indexed Documents</h3>
              <span className="badge-count">{documents.length}</span>
            </div>
            
            <div className="doc-list">
              {documents.length === 0 ? (
                <div className="empty-state">
                  No documents indexed yet. Upload a file to get started!
                </div>
              ) : (
                documents.map((doc) => (
                  <div key={doc.id} className="doc-item">
                    <div className="doc-info">
                      <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                      <div className="doc-meta-detail">
                        <span>🧩 {doc.chunk_count} chunks</span>
                        <span>•</span>
                        <span>{new Date(doc.upload_time).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <button 
                        className="view-chunks-btn" 
                        onClick={() => viewDocumentChunks(doc.id, doc.filename)}
                        title="View extracted text chunks"
                        style={{ marginRight: '0.5rem' }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                          <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                      </button>
                      <button 
                        className="delete-btn" 
                        onClick={() => handleDelete(doc.id)}
                        title="Delete document and vectors"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        {/* Right column: Search and results */}
        <div className="right-panel">
          
          {/* Search bar */}
          <section className="glass-card">
            <form onSubmit={handleSearch} className="search-bar-container">
              <div className="search-input-wrapper">
                <span className="search-icon-inline">⚡</span>
                <input 
                  type="text" 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your semantic query (e.g. 'predictive maintenance in EVs')..."
                  className="search-input"
                />
              </div>
              <button type="submit" className="search-btn" disabled={isSearching}>
                {isSearching ? <span className="spinner"></span> : "Search"}
              </button>
            </form>

            {/* Advanced Search Options */}
            <div className="search-options">
              <div className="option-group">
                <label>Top Results (K):</label>
                <input 
                  type="number" 
                  min="1" 
                  max="10" 
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  style={{ width: '55px' }}
                />
              </div>
              <div className="option-group">
                <label>Score Threshold:</label>
                <input 
                  type="range" 
                  min="0.1" 
                  max="0.9" 
                  step="0.05" 
                  value={scoreThreshold} 
                  onChange={(e) => setScoreThreshold(e.target.value)}
                />
                <span style={{ fontFamily: 'monospace', width: '30px' }}>{scoreThreshold}</span>
              </div>
              <div className="option-group">
                <label>Search in:</label>
                <select 
                  value={selectedDocFilter} 
                  onChange={(e) => setSelectedDocFilter(e.target.value)}
                  style={{ 
                    background: 'rgba(0, 0, 0, 0.2)', 
                    border: '1px solid var(--border-color)', 
                    color: 'var(--text-main)', 
                    borderRadius: '4px', 
                    padding: '0.25rem 0.5rem',
                    maxWidth: '150px',
                    outline: 'none'
                  }}
                >
                  <option value="">All Documents</option>
                  {documents.map(d => (
                    <option key={d.id} value={d.id}>{d.filename}</option>
                  ))}
                </select>
              </div>
            </div>
          </section>

          {/* Search Results Display */}
          <section className="glass-card search-results-section">
            <div className="results-header">
              <h3>Search Results</h3>
              {searchResults.length > 0 && (
                <span>
                  Found {searchResults.length} results 
                  {!semanticMatchesFound && " (using lexical fallback)"}
                </span>
              )}
            </div>

            {isFallback && searchResults.length > 0 && (
              <div className="fallback-banner">
                ⚠️ No high-confidence matches found. Showing closest low-confidence results.
              </div>
            )}

            <div className="search-results">
              {isSearching ? (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'var(--text-muted)' }}>
                  <span className="spinner" style={{ width: '30px', height: '30px', marginBottom: '1rem' }}></span>
                  <p>Searching vectors & computing relevance...</p>
                </div>
              ) : searchResults.length === 0 ? (
                hasSearched ? (
                  <div className="empty-state" style={{ padding: '6rem 1.5rem', borderColor: 'rgba(239, 68, 68, 0.2)' }}>
                    <div style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>❌</div>
                    <p style={{ fontWeight: 500, color: 'var(--text-main)' }}>No matches found</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                      No results exceeded the similarity threshold of {scoreThreshold}. Try lowering the threshold or adjusting your query.
                    </p>
                  </div>
                ) : (
                  <div className="empty-state" style={{ padding: '6rem 1.5rem' }}>
                    💡 Enter a query above to run a semantic search across your indexed documents.
                  </div>
                )
              ) : (
                searchResults.map((result) => {
                  const isExpanded = !!expandedChunks[result.chunk_id];
                  return (
                    <div key={result.chunk_id} className={`result-card ${isFallback ? 'fallback-card' : ''}`}>
                      <div className="result-meta">
                        <div className="result-doc-info">
                          <span className="result-doc-icon">📄</span>
                          <span style={{ fontWeight: 500 }}>{result.filename}</span>
                          <span style={{ color: 'var(--text-dark)' }}>|</span>
                          <span>Chunk: {result.chunk_id.split("_chunk_")[1]}</span>
                        </div>
                        <div className="result-score-badges">
                          <span className={`score-badge ${isFallback ? 'fallback-badge' : result.match_type}`}>
                            {result.match_type.toUpperCase()} ({result.score})
                          </span>
                        </div>
                      </div>
                      <p className="result-text">{highlightText(result.text, query)}</p>
                      
                      {result.parent_text && (
                        <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                          <button
                            onClick={() => setExpandedChunks(prev => ({ ...prev, [result.chunk_id]: !isExpanded }))}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: 'var(--color-secondary-light)',
                              fontSize: '0.8rem',
                              fontWeight: 500,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              padding: '0.25rem 0'
                            }}
                          >
                            <span>{isExpanded ? "📖 Hide surrounding context" : "📖 View surrounding context (Parent)"}</span>
                          </button>
                          
                          {isExpanded && (
                            <div style={{ 
                              marginTop: '0.5rem', 
                              padding: '0.75rem 1rem', 
                              background: 'rgba(255, 255, 255, 0.02)', 
                              borderLeft: '3px solid var(--color-secondary)',
                              borderRadius: '4px',
                              fontSize: '0.85rem',
                              lineHeight: 1.5,
                              color: 'var(--text-muted)'
                            }}>
                              <p>{highlightText(result.parent_text, query)}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </div>
      </main>

      {/* Document Chunks Modal */}
      {showChunksModal && (
        <div className="modal-overlay" onClick={() => setShowChunksModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '85%' }}>
                Chunks for: {selectedDocName}
              </h3>
              <button className="close-btn" onClick={() => setShowChunksModal(false)}>×</button>
            </div>
            
            <div className="modal-body">
              {isLoadingChunks ? (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'var(--text-muted)' }}>
                  <span className="spinner" style={{ width: '30px', height: '30px', marginBottom: '1rem' }}></span>
                  <p>Loading document text chunks...</p>
                </div>
              ) : selectedDocChunks.length === 0 ? (
                <div className="empty-state">
                  <p style={{ fontWeight: 500, marginBottom: '0.5rem' }}>No chunks found for this document in the vector store.</p>
                  <p style={{ fontSize: '0.82rem', color: 'var(--color-warning)' }}>
                    ⚠️ The vector database may be out of sync. Please delete this document from the list and upload it again to re-index its chunks.
                  </p>
                </div>
              ) : (
                selectedDocChunks.map((chunk) => (
                  <div key={chunk.chunk_id} className="chunk-card">
                    <div className="chunk-card-header">
                      <span>Chunk {chunk.chunk_index + 1}</span>
                      <span style={{ color: 'var(--text-dark)' }}>ID: {chunk.chunk_id.split("_chunk_")[1]}</span>
                    </div>
                    <p className="chunk-card-text">{chunk.text}</p>
                  </div>
                ))
              )}
            </div>
            
            <div className="modal-footer">
              <button className="close-modal-footer-btn" onClick={() => setShowChunksModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
