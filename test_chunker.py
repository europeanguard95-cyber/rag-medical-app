from rag_hub_cloud import chunker


def test_chunker_retourne_une_liste():
    result = chunker("hello world")
    assert isinstance(result, list)


def test_chunker_texte_court_donne_un_chunk():
    texte = "Texte court"
    chunks = chunker(texte, taille=500, overlap=80)
    assert len(chunks) == 1
    assert chunks[0] == texte


def test_chunker_taille_maximum_respectee():
    texte = "a" * 2000
    chunks = chunker(texte, taille=400, overlap=0)
    for c in chunks:
        assert len(c) <= 400


def test_chunker_nombre_de_chunks_sans_overlap():
    # 1200 chars / taille 500 / step 500 -> 3 chunks (500, 500, 200)
    texte = "x" * 1200
    chunks = chunker(texte, taille=500, overlap=0)
    assert len(chunks) == 3


def test_chunker_overlap_produit_plus_de_chunks():
    # 700 chars, taille 300 :
    #   sans overlap  (step=300) : 3 chunks  (0, 300, 600)
    #   overlap=100   (step=200) : 4 chunks  (0, 200, 400, 600)
    texte = "ab" * 350
    sans  = chunker(texte, taille=300, overlap=0)
    avec  = chunker(texte, taille=300, overlap=100)
    assert len(avec) > len(sans)


def test_chunker_overlap_cree_chevauchement():
    texte = "ABCDEFGHIJ" * 60
    taille, overlap = 100, 20
    chunks = chunker(texte, taille=taille, overlap=overlap)
    if len(chunks) >= 2:
        fin_chunk0   = chunks[0][taille - overlap:]
        debut_chunk1 = chunks[1][:overlap]
        assert fin_chunk0 == debut_chunk1


def test_chunker_ignore_chunks_vides():
    texte = "   " * 200
    chunks = chunker(texte, taille=50, overlap=0)
    assert chunks == []


def test_chunker_texte_vide():
    assert chunker("") == []


def test_chunker_valeurs_defaut():
    # taille=500 overlap=80 -> step=420
    # 1200 chars: debuts 0, 420, 840 -> 3 chunks
    texte = "z" * 1200
    chunks = chunker(texte)
    assert len(chunks) == 3


def test_chunker_contenu_couvre_tout_le_texte():
    texte = "Hello World! " * 100
    chunks = chunker(texte, taille=200, overlap=0)
    reconstruit = "".join(chunks)
    assert reconstruit == texte
