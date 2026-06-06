import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_external_services():
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = {"total_vector_count": 10}
    mock_index.query.return_value = {
        "matches": [
            {"metadata": {"text": "texte test", "source": "test.pdf"}, "score": 0.9}
        ]
    }

    with patch('pinecone.Pinecone') as mock_pc, \
         patch('groq.Groq') as mock_groq, \
         patch('sentence_transformers.SentenceTransformer') as mock_st, \
         patch('sentence_transformers.CrossEncoder') as mock_ce:

        mock_pc.return_value.list_indexes.return_value = []
        mock_pc.return_value.Index.return_value = mock_index
        mock_groq.return_value = MagicMock()
        mock_st.return_value = MagicMock()
        mock_ce.return_value = MagicMock()

        yield
