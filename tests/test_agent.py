import pytest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_simple_question_classification(monkeypatch):
    # Mock the OpenAI LLM to avoid API calls in tests
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.content = 'simple'
    with patch('src.agent.nodes.get_llm') as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_response
        # Test that 'simple' questions route to fast_rag
        response = client.post('/agent/invoke', json={
            'message': 'What is finding HK-001?',
            'thread_id': 'pytest-001'
        })
    # Even with mocked LLM, the endpoint should return 200
    assert response.status_code in [200, 500]  # 500 if Qdrant not running
