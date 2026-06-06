import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_external_services():
    with patch('rag_hub_cloud.Groq') as mock_groq, \
         patch('rag_hub_cloud.SentenceTransformer') as mock_st, \
         patch('rag_hub_cloud.CrossEncoder') as mock_ce, \
         patch('rag_hub_cloud.Pinecone') as mock_pinecone, \
         patch('rag_hub_cloud.compter_docs', return_value=10), \
         patch('streamlit.secrets', {"GROQ_API_KEY": "test", "PINECONE_API_KEY": "test", "PINECONE_INDEX": "test"}):

        mock_groq.return_value = MagicMock()
        mock_st.return_value   = MagicMock()
        mock_ce.return_value   = MagicMock()

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {"total_vector_count": 10}
        mock_pinecone.return_value.list_indexes.return_value = []
        mock_pinecone.return_value.Index.return_value = mock_index

        yield
