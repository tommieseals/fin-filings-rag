"""RAG retrieval and synthesis logic."""
import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .schemas import Citation, AskResponse

INDEX_DIR = Path(__file__).parent.parent / "index"
CONFIDENCE_THRESHOLD = 0.15
TOP_K = 3

class RAGEngine:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = None
        self._load_index()
    
    def _load_index(self):
        vectorizer_path = INDEX_DIR / "vectorizer.pkl"
        matrix_path = INDEX_DIR / "tfidf_matrix.pkl"
        docs_path = INDEX_DIR / "documents.json"
        
        if not all(p.exists() for p in [vectorizer_path, matrix_path, docs_path]):
            raise FileNotFoundError("Index not found. Run ingestion first.")
        
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(docs_path, "r") as f:
            self.documents = json.load(f)
    
    def retrieve(self, query, top_k=TOP_K):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append((self.documents[idx], score))
        return results
    
    def synthesize(self, query):
        results = self.retrieve(query)
        
        if not results:
            return AskResponse(
                answer="",
                confidence=0.0,
                citations=[],
                abstained=True,
                message="No relevant information found."
            )
        
        top_score = results[0][1]
        confidence = min(top_score * 2, 1.0)
        
        if top_score < CONFIDENCE_THRESHOLD:
            return AskResponse(
                answer="",
                confidence=confidence,
                citations=[],
                abstained=True,
                message=f"Confidence too low ({confidence:.2f})."
            )
        
        citations = []
        for doc, score in results:
            citations.append(Citation(
                chunk_id=doc["chunk_id"],
                text=doc["text"][:500] + ("..." if len(doc["text"]) > 500 else ""),
                score=round(score, 4),
                source_file=doc["source_file"]
            ))
        
        context_texts = [doc["text"] for doc, _ in results]
        answer = self._generate_answer(query, context_texts, citations)
        
        return AskResponse(
            answer=answer,
            confidence=round(confidence, 4),
            citations=citations,
            abstained=False
        )
    
    def _generate_answer(self, query, contexts, citations):
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

def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
