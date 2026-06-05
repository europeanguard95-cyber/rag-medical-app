from rag_hub_cloud import hasher, verifier_login, DROITS


def test_hasher_retourne_string_64_chars():
    h = hasher("password123")
    assert isinstance(h, str)
    assert len(h) == 64


def test_hasher_est_deterministe():
    assert hasher("abc") == hasher("abc")


def test_hasher_entrees_differentes_hashes_differents():
    assert hasher("password1") != hasher("password2")


def test_hasher_chaine_vide():
    h = hasher("")
    assert isinstance(h, str)
    assert len(h) == 64


def test_login_admin_valide():
    ok, role, nom = verifier_login("admin", "admin123")
    assert ok is True
    assert role == "admin"
    assert nom == "Administrateur"


def test_login_soignant_valide():
    ok, role, nom = verifier_login("soignant1", "soignant123")
    assert ok is True
    assert role == "soignant"
    assert nom == "Dr. Dupont"


def test_login_patient_valide():
    ok, role, nom = verifier_login("patient1", "patient123")
    assert ok is True
    assert role == "patient"
    assert nom == "Jean Martin"


def test_login_mauvais_mot_de_passe():
    ok, role, nom = verifier_login("admin", "mauvais_mdp")
    assert ok is False
    assert role is None
    assert nom is None


def test_login_utilisateur_inexistant():
    ok, role, nom = verifier_login("inconnu", "admin123")
    assert ok is False
    assert role is None
    assert nom is None


def test_login_champs_vides():
    ok, role, nom = verifier_login("", "")
    assert ok is False


def test_login_injection_sql_like():
    ok, _, _ = verifier_login("admin' OR '1'='1", "' OR '1'='1")
    assert ok is False


def test_login_casse_sensible():
    ok, _, _ = verifier_login("Admin", "admin123")
    assert ok is False
    ok, _, _ = verifier_login("admin", "Admin123")
    assert ok is False


def test_trois_roles_definis():
    assert "admin" in DROITS
    assert "soignant" in DROITS
    assert "patient" in DROITS


def test_admin_a_tous_les_droits():
    d = DROITS["admin"]
    assert d["upload_pdf"]      is True
    assert d["voir_sources"]    is True
    assert d["generer_rapport"] is True
    assert d["comparer"]        is True


def test_patient_na_aucun_droit():
    d = DROITS["patient"]
    assert d["upload_pdf"]      is False
    assert d["voir_sources"]    is False
    assert d["generer_rapport"] is False
    assert d["comparer"]        is False


def test_soignant_droits_intermediaires():
    d = DROITS["soignant"]
    assert d["upload_pdf"]      is False
    assert d["voir_sources"]    is True
    assert d["generer_rapport"] is True
    assert d["comparer"]        is True


def test_chaque_role_a_les_quatre_cles():
    cles = {"upload_pdf", "voir_sources", "generer_rapport", "comparer"}
    for role, droits in DROITS.items():
        assert set(droits.keys()) == cles


def test_droits_sont_booleens():
    for role, droits in DROITS.items():
        for cle, valeur in droits.items():
            assert isinstance(valeur, bool), f"DROITS['{role}']['{cle}'] n'est pas un bool"
