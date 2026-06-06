def chunker(texte, taille=500, overlap=80):
    chunks, debut = [], 0
    while debut < len(texte):
        chunk = texte[debut:debut+taille]
        if chunk.strip():
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

def test_chunker_basic():
    texte = "mot " * 200
    chunks = chunker(texte)
    assert len(chunks) > 0

def test_chunker_overlap():
    texte = "a " * 300
    chunks = chunker(texte, taille=100, overlap=20)
    assert len(chunks) > 1

def test_chunker_vide():
    chunks = chunker("")
    assert chunks == []
