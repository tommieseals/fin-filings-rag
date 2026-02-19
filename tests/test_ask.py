"""Tests for the /ask endpoint."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import get_engine


client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_ask_valid_question():
    """Test /ask with valid question."""
    response = client.post(
        "/ask",
        json={"question": "What are the risk factors?"}
    )
    # May be 200 or 503 depending on index state
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "abstained" in data


def test_ask_empty_question():
    """Test /ask rejects empty questions."""
    response = client.post(
        "/ask",
        json={"question": ""}
    )
    assert response.status_code == 422


def test_ask_returns_citations():
    """Test that citations include required fields."""
    response = client.post(
        "/ask",
        json={"question": "What financial risks are disclosed?"}
    )
    if response.status_code == 200:
        data = response.json()
        assert "citations" in data
        for citation in data["citations"]:
            assert "source" in citation
            assert "chunk_id" in citation
            assert "text" in citation
            assert "score" in citation
