"""Ingestion script for financial filings."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.chunk import chunk_text


def ingest_filings(
    filings_dir: str = "data/filings",
    index_dir: str = "index",
    chunk_size: int = 512,
    overlap: int = 64,
) -> dict:
    """Ingest all text files from filings directory."""
    filings_path = Path(filings_dir)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    
    chunks = []
    
    # Process all text files
    for filepath in filings_path.glob("*.txt"):
        print(f"Processing {filepath.name}...")
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({
                "source": filepath.name,
                "chunk_id": i,
                "text": chunk,
            })
    
    if not chunks:
        print("No chunks generated. Check your filings directory.")
        return {"status": "error", "chunks": 0, "files": 0}
    
    print(f"Generated {len(chunks)} chunks from {len(list(filings_path.glob('*.txt')))} files")
    
    # Build TF-IDF index
    print("Building TF-IDF index...")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save chunks
    chunks_path = index_path / "chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    
    # Save TF-IDF matrix
    matrix_path = index_path / "tfidf_matrix.npy"
    np.save(matrix_path, tfidf_matrix.toarray())
    
    # Save vectorizer vocabulary (convert numpy int64 to Python int)
    vocab_path = index_path / "vectorizer.json"
    vocab_dict = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    with open(vocab_path, "w") as f:
        json.dump({"vocabulary": vocab_dict}, f)
    
    stats = {
        "status": "success",
        "chunks": len(chunks),
        "files": len(list(filings_path.glob("*.txt"))),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    
    print(f"Index built successfully: {stats}")
    return stats


if __name__ == "__main__":
    ingest_filings()
