import numpy as np
from unittest.mock import MagicMock, patch

import rag_hub_cloud as app

MODULE = "rag_hub_cloud"


def make_embed_mock():
    m = MagicMock()
    def encode(texts, **kw):
        n = len(texts) if isinstance(texts, list) else 1
        return np.random.rand(n, 384)
    m.encode = encode
    return m


def make_reranker_mock():
    m = MagicMock()
    m.predict = lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))]
    return m


def make_pinecone_results(chunks, source="doc.pdf"):
    matches = []
    for chunk in chunks:
        match = MagicMock()
        match.metadata = {"text": chunk, "source": source}
        matches.append(match)
    result = MagicMock()
    result.matches = matches
    return result


def make_bm25_mock(n_docs):
    bm25 = MagicMock()
    bm25.get_scores.return_value = np.zeros(n_docs)
    return bm25


SAMPLE_CHUNKS = [
    "La pre-eclampsie se definit par une hypertension apres 20 SA.",
    "Les criteres de surveillance incluent la tension arterielle.",
    "Le traitement de premiere intention est le labetalol.",
]
SAMPLE_METAS = [{"source": "protocole.pdf"}] * 3


class TestHyDE:

    def test_retourne_une_string(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value.choices[0].message.content = "Reponse hypothetique"
        with patch.object(app, "client_groq", mock_groq):
            result = app.generer_hyde("Question test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_si_groq_echoue(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("API error")
        with patch.object(app, "client_groq", mock_groq):
            result = app.generer_hyde("Ma question originale")
        assert result == "Ma question originale"

    def test_appelle_groq_avec_la_question(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value.choices[0].message.content = "Rep"
        question = "Quels sont les criteres de la pre-eclampsie ?"
        with patch.object(app, "client_groq", mock_groq):
            app.generer_hyde(question)
        appel = mock_groq.chat.completions.create.call_args
        assert question in str(appel)


class TestRerank:

    def test_chunks_vides_retourne_listes_vides(self):
        with patch.object(app, "reranker", make_reranker_mock()):
            chunks, sources = app.rerank("question", [], [], n_final=5)
        assert chunks == []
        assert sources == []

    def test_n_final_respecte(self):
        chunks  = ["chunk_" + str(i) for i in range(10)]
        sources = [{"source": "doc.pdf"}] * 10
        with patch.object(app, "reranker", make_reranker_mock()):
            result_chunks, result_sources = app.rerank("q", chunks, sources, n_final=3)
        assert len(result_chunks)  == 3
        assert len(result_sources) == 3

    def test_trie_par_score_decroissant(self):
        chunks  = ["meilleur", "moyen", "moins_bon"]
        sources = [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}]
        with patch.object(app, "reranker", make_reranker_mock()):
            ranked, _ = app.rerank("question", chunks, sources, n_final=3)
        assert ranked[0] == "meilleur"

    def test_n_final_superieur_a_nb_chunks(self):
        chunks  = ["c1", "c2"]
        sources = [{"source": "x.pdf"}] * 2
        with patch.object(app, "reranker", make_reranker_mock()):
            result, _ = app.rerank("q", chunks, sources, n_final=10)
        assert len(result) == 2


class TestHybridSearch:

    def test_retourne_chunks_et_sources(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 10
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS[:2])
        with patch.object(app, "index",         mock_index),              patch.object(app, "model_embed",   make_embed_mock()),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi", return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            chunks, sources = app.hybrid_search("tension arterielle")
        assert isinstance(chunks, list)
        assert isinstance(sources, list)
        assert len(chunks) == len(sources)

    def test_index_vide_retourne_listes_vides(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 0
        with patch.object(app, "index", mock_index),              patch.object(app, "get_all_chunks", return_value=([], [])):
            chunks, sources = app.hybrid_search("question")
        assert chunks  == []
        assert sources == []

    def test_filtre_source_passe_a_pinecone(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 5
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS[:1])
        with patch.object(app, "index",         mock_index),              patch.object(app, "model_embed",   make_embed_mock()),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi", return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            app.hybrid_search("question", sources_filtres=["protocole.pdf"])
        call_kwargs = mock_index.query.call_args[1]
        assert "filter" in call_kwargs


class TestRagQuery:

    def _make_groq_mock(self, text="Reponse generee."):
        m = MagicMock()
        m.chat.completions.create.return_value.choices[0].message.content = text
        return m

    def test_retourne_les_quatre_valeurs(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 5
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS[:2])
        with patch.object(app, "client_groq",    self._make_groq_mock()),              patch.object(app, "model_embed",    make_embed_mock()),              patch.object(app, "reranker",       make_reranker_mock()),              patch.object(app, "index",          mock_index),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi",        return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            result = app.rag_query("Question", role="soignant", n_chunks=2,
                                   use_hyde=False, use_hybrid=True, use_reranking=True)
        reponse, chunks_f, sources_f, hyde_text = result
        assert isinstance(reponse,   str)
        assert isinstance(chunks_f,  list)
        assert isinstance(sources_f, list)

    def test_reponse_non_vide(self):
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 5
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS)
        with patch.object(app, "client_groq",    self._make_groq_mock("Reponse medicale.")),              patch.object(app, "model_embed",    make_embed_mock()),              patch.object(app, "reranker",       make_reranker_mock()),              patch.object(app, "index",          mock_index),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi",        return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            reponse, _, _, _ = app.rag_query("Question", role="admin", n_chunks=2,
                                             use_hyde=False, use_hybrid=True, use_reranking=True)
        assert len(reponse) > 0

    def test_hyde_active_appelle_groq_deux_fois(self):
        mock_groq = self._make_groq_mock()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 5
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS)
        with patch.object(app, "client_groq",    mock_groq),              patch.object(app, "model_embed",    make_embed_mock()),              patch.object(app, "reranker",       make_reranker_mock()),              patch.object(app, "index",          mock_index),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi",        return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            app.rag_query("Question", role="admin", n_chunks=2,
                          use_hyde=True, use_hybrid=True, use_reranking=False)
        assert mock_groq.chat.completions.create.call_count >= 2

    def test_erreur_llm_retourne_message_erreur(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("Groq unavailable")
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 5
        mock_index.query.return_value = make_pinecone_results(SAMPLE_CHUNKS)
        with patch.object(app, "client_groq",    mock_groq),              patch.object(app, "model_embed",    make_embed_mock()),              patch.object(app, "reranker",       make_reranker_mock()),              patch.object(app, "index",          mock_index),              patch.object(app, "get_all_chunks", return_value=(SAMPLE_CHUNKS, SAMPLE_METAS)),              patch(MODULE + ".BM25Okapi",        return_value=make_bm25_mock(len(SAMPLE_CHUNKS))):
            reponse, _, _, _ = app.rag_query("Question", role="admin", n_chunks=2,
                                             use_hyde=False, use_hybrid=True, use_reranking=False)
        assert "Erreur" in reponse or "indisponible" in reponse.lower()
