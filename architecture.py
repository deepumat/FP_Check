PDFs
  │
  ├── If text-based → PyMuPDF
  └── If scanned   → Tesseract OCR
              │
        Normalized text
              │
         Chunking
              │
        MiniLM Embeddings
              │
     Local Vector Store (NumPy + FAISS-like)
              │
         Semantic Search
              │
           LLM