import hashlib

def hasher(p):
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {
    "admin": {"password": hasher("admin123"), "role": "admin"},
    "soignant1": {"password": hasher("soignant123"), "role": "soignant"},
    "patient1": {"password": hasher("patient123"), "role": "patient"}
}

def verifier_login(u, p):
    if u in USERS and USERS[u]["password"] == hasher(p):
        return True, USERS[u]["role"]
    return False, None

def test_login_admin_valide():
    ok, role = verifier_login("admin", "admin123")
    assert ok and role == "admin"

def test_login_mauvais_mdp():
    ok, role = verifier_login("admin", "mauvais")
    assert not ok

def test_login_user_inexistant():
    ok, role = verifier_login("inconnu", "test")
    assert not ok

def test_hasher_coherent():
    assert hasher("test") == hasher("test")
    assert hasher("a") != hasher("b")
