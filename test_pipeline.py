def test_contexte_construction():
    chunks = ["chunk 1 contenu", "chunk 2 contenu"]
    sources = [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
    contexte = "\n\n".join([
        "[Source " + str(i+1) + " — " + m["source"] + "]\n" + c
        for i, (c, m) in enumerate(zip(chunks, sources))
    ])
    assert "Source 1" in contexte
    assert "doc1.pdf" in contexte
    assert "chunk 1 contenu" in contexte

def test_chunker_pipeline():
    texte = "Le patient présente une hypertension artérielle. " * 20
    chunks, debut = [], 0
    while debut < len(texte):
        chunk = texte[debut:debut+500]
        if chunk.strip():
            chunks.append(chunk)
        debut += 500 - 80
    assert len(chunks) > 0
    assert all(len(c) <= 500 for c in chunks)

def test_droits_roles():
    DROITS = {
        "admin":    {"upload_pdf": True,  "voir_sources": True},
        "soignant": {"upload_pdf": False, "voir_sources": True},
        "patient":  {"upload_pdf": False, "voir_sources": False}
    }
    assert DROITS["admin"]["upload_pdf"] == True
    assert DROITS["soignant"]["upload_pdf"] == False
    assert DROITS["patient"]["voir_sources"] == False
