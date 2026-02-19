"""RAG retrieval and synthesis logic."""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import Citation

INDEX_DIR = Path(__file__).parent.parent / "index"
CONFIDENCE_THRESHOLD = 0.15
TOP_K = 3


@dataclass
class RAGResponse:
    """Response from RAG engine."""
    answer: str
    confidence: float
    citations: List[Citation]
    abstained: bool
    message: Optional[str] = None


class RAGEngine:
    """RAG engine for querying SEC filings."""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = None
        self._ready = False
        self._load_index()
    
    def _load_index(self):
        """Load index files created by ingestion."""
        chunks_path = INDEX_DIR / "chunks.json"
        matrix_path = INDEX_DIR / "tfidf_matrix.npy"
        vocab_path = INDEX_DIR / "vectorizer.json"
        
        if not all(p.exists() for p in [chunks_path, matrix_path, vocab_path]):
            raise FileNotFoundError("Index not found. Run ingestion first.")
        
        # Load documents (chunks)
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
        
        # Convert to expected format
        self.documents = []
        for chunk in chunks:
            self.documents.append({
                "chunk_id": f"{chunk['source']}_{chunk['chunk_id']}",
                "text": chunk["text"],
                "source_file": chunk["source"],
            })
        
        # Load TF-IDF matrix
        self.tfidf_matrix = np.load(matrix_path)
        
        # Recreate vectorizer from vocabulary
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            vocabulary=vocab_data["vocabulary"],
        )
        # Fit on empty to initialize with vocabulary
        self.vectorizer.fit([""])
        self._ready = True
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self._ready and self.documents is not None
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """Retrieve top-k relevant documents."""
        query_vec = self.vectorizer.transform([query]).toarray()
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append((self.documents[idx], score))
        return results
    
    def ask(self, query: str) -> RAGResponse:
        """Ask a question and get a response object."""
        result = self.synthesize(query)
        return RAGResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            citations=result["citations"],
            abstained=result["abstained"],
            message=result.get("message"),
        )
    
    def synthesize(self, query: str) -> dict:
        """Generate answer with citations (returns dict)."""
        results = self.retrieve(query)
        
        if not results:
            return {
                "answer": "",
                "confidence": 0.0,
                "citations": [],
                "abstained": True,
                "message": "No relevant information found.",
            }
        
        top_score = results[0][1]
        confidence = min(top_score * 2, 1.0)
        
        if top_score < CONFIDENCE_THRESHOLD:
            return {
                "answer": "",
                "confidence": confidence,
                "citations": [],
                "abstained": True,
                "message": f"Confidence too low ({confidence:.2f}).",
            }
        
        citations = []
        for doc, score in results:
            citations.append(Citation(
                chunk_id=doc["chunk_id"],
                text=doc["text"][:500] + ("..." if len(doc["text"]) > 500 else ""),
                score=round(score, 4),
                source_file=doc["source_file"],
            ))
        
        context_texts = [doc["text"] for doc, _ in results]
        answer = self._generate_answer(query, context_texts, citations)
        
        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "citations": citations,
            "abstained": False,
        }
    
    def _generate_answer(self, query: str, contexts: list, citations: list) -> str:
        """Generate answer from context."""
        combined = " ".join(contexts)
        sentences = combined.replace("\n", " ").split(".")
        
        query_terms = set(query.lower().split())
        relevant_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            if overlap > 0:
                relevant_sentences.append((sent, overlap))
        
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        if top_sentences:
            answer = ". ".join(top_sentences) + "."
            sources = list(set(c.source_file for c in citations))
            answer += f" [Sources: {', '.join(sources)}]"
            return answer
        
        return f"Based on the filing: {contexts[0][:300]}..."


_engine = None


def get_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
