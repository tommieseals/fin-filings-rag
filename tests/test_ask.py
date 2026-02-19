"""API tests for /ask endpoint with mocked RAG engine."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_mock_engine():
    """Create a mock engine for tests."""
    mock = MagicMock()
    mock.documents = [{"chunk_id": "test_1", "text": "Sample text", "source_file": "test.txt"}]
    mock.vectorizer = MagicMock()
    mock.synthesize.return_value = {
        "answer": "This is a test answer about financial risks.",
        "citations": [
            {"chunk_id": "chunk_001", "text": "Sample citation text...", "score": 0.85, "source_file": "10k_2023.txt"}
        ],
        "confidence": 0.85,
        "abstained": False,
    }
    return mock


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()


@patch("app.main.get_engine")
def test_ask_valid_question(mock_get_engine):
    """Test /ask with valid question."""
    mock_get_engine.return_value = create_mock_engine()
    
    response = client.post(
        "/ask",
        json={"question": "What are the risk factors?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data


def test_ask_short_question():
    """Test /ask rejects too short questions."""
    response = client.post(
        "/ask",
        json={"question": "hi"}
    )
    assert response.status_code == 422


@patch("app.main.get_engine")
def test_ask_returns_citations(mock_get_engine):
    """Test that citations include required fields."""
    mock_get_engine.return_value = create_mock_engine()
    
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


@patch("app.main.get_engine")
def test_stats_endpoint(mock_get_engine):
    """Test stats endpoint."""
    mock_get_engine.return_value = create_mock_engine()
    
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "documents_indexed" in data
    assert "status" in data
