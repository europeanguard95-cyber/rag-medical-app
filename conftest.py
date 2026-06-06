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
    mock_pc_instance = MagicMock()
    mock_pc_instance.list_indexes.return_value = []
    mock_pc_instance.Index.return_value = mock_index

    with patch('rag_hub_cloud.Pinecone', return_value=mock_pc_instance), \
         patch('rag_hub_cloud.client_groq', MagicMock()), \
         patch('rag_hub_cloud.model_embed', MagicMock()), \
         patch('rag_hub_cloud.reranker', MagicMock()), \
         patch('rag_hub_cloud.pine_index', mock_index):
        yield
