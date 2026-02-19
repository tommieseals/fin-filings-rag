"""API tests for /ask endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_valid_question():
    """Test /ask with valid question."""
    response = client.post(
        "/ask",
        json={"question": "What are the risk factors?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert 0 <= data["confidence"] <= 1

def test_ask_short_question():
    """Test /ask rejects too short questions."""
    response = client.post(
        "/ask",
        json={"question": "hi"}
    )
    assert response.status_code == 422

def test_ask_returns_citations():
    """Test that citations include required fields."""
    response = client.post(
        "/ask",
        json={"question": "What financial risks are disclosed?"}
    )
    assert response.status_code == 200
    data = response.json()
    
    if not data.get("abstained") and data["citations"]:
        citation = data["citations"][0]
        assert "chunk_id" in citation
        assert "text" in citation
        assert "score" in citation
        assert "source_file" in citation

def test_stats_endpoint():
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
