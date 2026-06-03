import os
import json
import hashlib
import datetime
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from datetime import date
import chromadb

# ── CLÉ API depuis les secrets Streamlit ───────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

st.set_page_config(page_title="RAG Médical", page_icon="🏥", layout="centered")

# ── UTILISATEURS (stockés en mémoire session) ──────────────
USERS_DEFAUT = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "nom": "Administrateur"
    },
    "soignant1": {
        "password": hashlib.sha256("soignant123".encode()).hexdigest(),
        "role": "soignant",
        "nom": "Dr. Dupont"
    },
    "patient1": {
        "password": hashlib.sha256("patient123".encode()).hexdigest(),
        "role": "patient",
        "nom": "Jean Martin"
    }
}

def hasher(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verifier_login(username, password):
    users = USERS_DEFAUT
    if username in users:
        if users[username]["password"] == hasher(password):
            return True, users[username]["role"], users[username]["nom"]
    return False, None, None

DROITS = {
    "admin":    {"upload_pdf": True,  "voir_sources": True,  "generer_rapport": True, "comparer": True,  "niveau_reponse": "professionnel"},
    "soignant": {"upload_pdf": False, "voir_sources": True,  "generer_rapport": True, "comparer": True,  "niveau_reponse": "professionnel"},
    "patient":  {"upload_pdf": False, "voir_sources": False, "generer_rapport": False, "comparer": False, "niveau_reponse": "simplifie"}
}

# ── CHARGEMENT RAG (en mémoire) ────────────────────────────
@st.cache_resource
def charger_modeles():
    client_groq  = Groq(api_key=GROQ_API_KEY)
    model_embed  = SentenceTransformer('all-MiniLM-L6-v2')
    reranker     = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return client_groq, model_embed, reranker

client_groq, model_embed, reranker = charger_modeles()

# VectorDB en mémoire (partagée entre sessions via cache)
if "collection" not in st.session_state:
    client_chroma = chromadb.Client()
    try:
        st.session_state.collection = client_chroma.get_collection("rag_docs")
    except:
        st.session_state.collection = client_chroma.create_collection("rag_docs")

collection = st.session_state.collection

# ── FONCTIONS ──────────────────────────────────────────────
def chunker(texte, taille=500, overlap=80):
    chunks, debut = [], 0
    while debut < len(texte):
        chunk = texte[debut:debut+taille]
        if chunk.strip():
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

def indexer_pdf(fichier_upload, nom_fichier):
    reader = PdfReader(fichier_upload)
    texte  = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    chunks    = chunker(texte)
    existants = collection.count()
    for i in range(0, len(chunks), 50):
        batch      = chunks[i:i+50]
        embeddings = model_embed.encode(batch).tolist()
        metadatas  = [{"source": nom_fichier} for _ in batch]
        ids        = ["chunk_" + str(existants + i + j) for j in range(len(batch))]
        collection.add(documents=batch, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return len(chunks), len(reader.pages)

def hybrid_search(question, sources_filtres=None, n_initial=20):
    if collection.count() == 0:
        return [], []
    q_emb = model_embed.encode([question]).tolist()
    kwargs = {"n_results": min(n_initial, collection.count()), "include": ["documents", "metadatas"]}
    if sources_filtres and len(sources_filtres) == 1:
        kwargs["where"] = {"source": sources_filtres[0]}
    resultats_vect = collection.query(query_embeddings=q_emb, **kwargs)
    chunks_vect  = resultats_vect["documents"][0]
    sources_vect = resultats_vect["metadatas"][0]

    tous  = collection.get(include=["documents", "metadatas"])
    docs, metas = tous["documents"], tous["metadatas"]
    if sources_filtres:
        docs  = [d for d, m in zip(docs, metas) if m["source"] in sources_filtres]
        metas = [m for m in metas if m["source"] in sources_filtres]
    if not docs:
        return chunks_vect, sources_vect

    corpus = [d.lower().split() for d in docs]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(question.lower().split())
    top    = scores.argsort()[-n_initial:][::-1]
    chunks_bm25  = [docs[i] for i in top]
    sources_bm25 = [metas[i] for i in top]

    scores_f, src_map = {}, {}
    for rank, (c, m) in enumerate(zip(chunks_vect, sources_vect)):
        scores_f[c] = scores_f.get(c, 0) + 1/(rank+60)
        src_map[c]  = m
    for rank, (c, m) in enumerate(zip(chunks_bm25, sources_bm25)):
        scores_f[c] = scores_f.get(c, 0) + 1/(rank+60)
        src_map[c]  = m

    tries = sorted(scores_f, key=scores_f.get, reverse=True)
    return tries[:n_initial], [src_map[c] for c in tries[:n_initial]]

def rerank(question, chunks, sources, n_final=5):
    if not chunks:
        return [], []
    scores   = reranker.predict([(question, c) for c in chunks])
    combined = sorted(zip(chunks, sources, scores), key=lambda x: x[2], reverse=True)
    return [c for c, s, _ in combined[:n_final]], [s for _, s, _ in combined[:n_final]]

# ══════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏥 RAG Médical")
    st.caption("Suite complète d'outils RAG pour la santé")
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Connexion")
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter", type="primary", use_container_width=True):
            ok, role, nom = verifier_login(username, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.role      = role
                st.session_state.nom       = nom
                st.session_state.messages  = []
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect")
        st.divider()
        st.caption("admin / admin123 · soignant1 / soignant123 · patient1 / patient123")
    st.stop()

# ══════════════════════════════════════════════════════════
# APP PRINCIPALE
# ══════════════════════════════════════════════════════════
username = st.session_state.username
role     = st.session_state.role
nom      = st.session_state.nom
droits   = DROITS[role]

badge = {"admin": "🔴", "soignant": "🟡", "patient": "🟢"}
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏥 RAG Médical")
    st.caption(badge[role] + " " + nom + " — " + role.upper())
with col2:
    if st.button("🚪 Déconnexion", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.divider()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.header("🧭 Navigation")

    menus_autorises = ["🏠 Accueil", "💬 Assistant Q&A"]
    if droits["generer_rapport"]:
        menus_autorises.append("📋 Formateur")
    if droits["comparer"]:
        menus_autorises.append("⚖️ Comparateur")
    if droits["generer_rapport"]:
        menus_autorises.append("📝 Rapports")

    choix = st.radio("", menus_autorises, label_visibility="collapsed")

    st.divider()

    if droits["upload_pdf"]:
        st.header("📄 Ajouter un PDF")
        pdf_up = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_up:
            nom_f = pdf_up.name
            deja  = False
            if collection.count() > 0:
                test = collection.get(where={"source": nom_f}, limit=1)
                if test["ids"]:
                    deja = True
            if deja:
                st.info("✅ Déjà indexé")
            else:
                if st.button("➕ Indexer"):
                    with st.spinner("Indexation..."):
                        nb_c, nb_p = indexer_pdf(pdf_up, nom_f)
                    st.success(str(nb_p) + " pages — " + str(nb_c) + " chunks")
                    st.rerun()

    st.divider()
    st.header("📚 Documents")
    if collection.count() > 0:
        tous = collection.get(include=["metadatas"])
        srcs = list(set(m["source"] for m in tous["metadatas"]))
        for src in srcs:
            nb = sum(1 for m in tous["metadatas"] if m["source"] == src)
            st.markdown("• " + src + " (" + str(nb) + ")")
        st.metric("Total chunks", collection.count())
    else:
        st.info("Aucun document — l'admin doit uploader un PDF")

# ══════════════════════════════════════════════════════════
# ACCUEIL
# ══════════════════════════════════════════════════════════
if choix == "🏠 Accueil":
    st.subheader("Bienvenue, " + nom + " !")
    st.markdown("Choisis un outil dans le menu à gauche.")
    st.divider()
    cols = st.columns(2)
    outils = [
        ("💬", "Assistant Q&A", "Pose des questions sur tes documents"),
        ("📋", "Formateur", "Fiches procédures étape par étape"),
        ("⚖️", "Comparateur", "Compare deux documents"),
        ("📝", "Rapports", "Génère des synthèses structurées"),
    ]
    for i, (icone, titre_outil, desc) in enumerate(outils):
        with cols[i % 2]:
            st.markdown(
                "<div style='padding:1rem; border:0.5px solid var(--color-border-tertiary); border-radius:8px; margin-bottom:12px;'>"
                "<p style='font-size:24px; margin:0;'>" + icone + "</p>"
                "<p style='font-weight:500; margin:4px 0;'>" + titre_outil + "</p>"
                "<p style='font-size:13px; color:var(--color-text-secondary); margin:0;'>" + desc + "</p>"
                "</div>", unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════
# RAG 1 — ASSISTANT Q&A
# ══════════════════════════════════════════════════════════
elif choix == "💬 Assistant Q&A":
    st.subheader("💬 Assistant Q&A Médical")
    if collection.count() == 0:
        st.info("Aucun document disponible. L'admin doit uploader des PDFs.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Nouveau chat"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Pose ta question...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Recherche..."):
                chunks_c, sources_c = hybrid_search(question)
                chunks_f, sources_f = rerank(question, chunks_c, sources_c)
                contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
                consigne = "Utilise un langage simple et rassurant." if role == "patient" else "Utilise le vocabulaire clinique precis. Cite tes sources."
                prompt   = "Tu es un assistant medical. " + consigne + "\nReponds UNIQUEMENT depuis le contexte.\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nQUESTION : " + question + "\nREPONSE :"
                response = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=500)
                reponse  = response.choices[0].message.content
            st.markdown(reponse)
            if droits["voir_sources"]:
                with st.expander("📚 Sources"):
                    for i, (c, m) in enumerate(zip(chunks_f, sources_f)):
                        st.markdown("**Source " + str(i+1) + " — " + m["source"] + "**")
                        st.text(c[:300] + "..." if len(c) > 300 else c)
                        st.divider()
            if role == "patient":
                st.info("ℹ️ Consultez votre professionnel de santé.")
        st.session_state.messages.append({"role": "assistant", "content": reponse})

# ══════════════════════════════════════════════════════════
# RAG 2 — FORMATEUR
# ══════════════════════════════════════════════════════════
elif choix == "📋 Formateur":
    st.subheader("📋 Formateur de Procédures")
    if collection.count() == 0:
        st.info("Aucun document disponible.")
        st.stop()

    public = st.radio("Public cible", ["Sage-femme / Médecin", "Infirmier(e)", "Patient"], horizontal=True)
    sujet  = st.text_input("Sujet", placeholder="Ex: Surveillance tensionnelle pendant la grossesse")

    if st.button("📋 Générer", type="primary") and sujet.strip():
        with st.spinner("Génération..."):
            chunks_c, sources_c = hybrid_search(sujet)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 6)
            contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
            consigne = "Vocabulaire clinique precis." if "Médecin" in public else ("Langage professionnel accessible." if "Infirmier" in public else "Langage simple.")
            prompt   = "Tu es formateur medical. " + consigne + " Base-toi UNIQUEMENT sur le contexte.\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nSUJET : " + sujet + "\n\nStructure :\n## Objectif\n## Prérequis\n## Étapes\n1.\n2.\n## Points de vigilance\n⚠️\n## Sources"
            response = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=800)
            fiche    = response.choices[0].message.content
        st.markdown(fiche)
        st.download_button("⬇️ Télécharger", data=fiche, file_name="procedure_" + sujet[:20].replace(" ", "_") + ".txt", mime="text/plain")
        etapes = [l for l in fiche.split("\n") if l.strip() and l.strip()[0].isdigit()]
        if etapes:
            st.divider()
            st.subheader("✅ Checklist")
            for i, e in enumerate(etapes):
                st.checkbox(e.strip(), key="chk_" + str(i))

# ══════════════════════════════════════════════════════════
# RAG 3 — COMPARATEUR
# ══════════════════════════════════════════════════════════
elif choix == "⚖️ Comparateur":
    st.subheader("⚖️ Comparateur de Documents")
    if collection.count() == 0:
        st.info("Aucun document disponible.")
        st.stop()

    tous = collection.get(include=["metadatas"])
    srcs = list(set(m["source"] for m in tous["metadatas"]))

    if len(srcs) < 2:
        st.warning("Tu as besoin d'au moins 2 documents pour comparer.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        src1 = st.selectbox("Document 1", srcs, index=0)
    with col2:
        src2 = st.selectbox("Document 2", [s for s in srcs if s != src1], index=0)

    sujet = st.text_input("Sujet de comparaison", placeholder="Ex: Surveillance de la grossesse")
    axe   = st.selectbox("Axe", ["Recommandations cliniques", "Critères de surveillance", "Prise en charge", "Définitions"])

    if st.button("⚖️ Comparer", type="primary") and sujet.strip():
        with st.spinner("Comparaison..."):
            c1, s1 = hybrid_search(sujet, [src1])
            c2, s2 = hybrid_search(sujet, [src2])
            c1f, _ = rerank(sujet, c1, s1, 5)
            c2f, _ = rerank(sujet, c2, s2, 5)
            ctx1 = "\n\n".join(["[" + src1 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c1f)])
            ctx2 = "\n\n".join(["[" + src2 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c2f)])
            prompt = "Compare ces deux documents sur : " + sujet + "\nAxe : " + axe + "\n\n--- DOC 1 : " + src1 + " ---\n" + ctx1 + "\n\n--- DOC 2 : " + src2 + " ---\n" + ctx2 + "\n\nStructure :\n## Synthese\n## Points de convergence\n## Points de divergence\n| Critere | " + src1 + " | " + src2 + " |\n|---|---|---|\n## Recommandation\n## Limites"
            response    = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=900)
            comparaison = response.choices[0].message.content
        st.markdown(comparaison)
        st.download_button("⬇️ Télécharger", data=comparaison, file_name="comparaison_" + sujet[:20].replace(" ", "_") + ".txt", mime="text/plain")

# ══════════════════════════════════════════════════════════
# RAG 4 — RAPPORTS
# ══════════════════════════════════════════════════════════
elif choix == "📝 Rapports":
    st.subheader("📝 Générateur de Rapports")
    if collection.count() == 0:
        st.info("Aucun document disponible.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        titre        = st.text_input("Titre", placeholder="Ex: Protocole sage-femme 2024")
    with col2:
        destinataire = st.text_input("Destinataire", placeholder="Ex: Équipe soignante")

    type_rapport   = st.selectbox("Type", ["Synthèse clinique", "Rapport de formation", "Note de synthèse", "Résumé exécutif", "Compte-rendu"])
    sujet          = st.text_area("Sujet et instructions", height=80)
    contexte_libre = st.text_input("Contexte additionnel (optionnel)")

    tous = collection.get(include=["metadatas"])
    srcs = list(set(m["source"] for m in tous["metadatas"]))
    srcs_sel = st.multiselect("Sources (vide = toutes)", srcs)

    if st.button("📝 Générer", type="primary") and sujet.strip() and titre.strip():
        with st.spinner("Génération..."):
            src_filtres = srcs_sel if srcs_sel else None
            chunks_c, sources_c = hybrid_search(sujet, src_filtres, 25)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 8)
            contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
            today    = date.today().strftime("%d/%m/%Y")
            prompt   = "Tu es expert redacteur medical. Genere un rapport professionnel en francais.\nType : " + type_rapport + "\nTitre : " + titre + "\nDestinataire : " + destinataire + "\nDate : " + today + "\n" + ("Contexte : " + contexte_libre if contexte_libre else "") + "\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nSujet : " + sujet + "\n\nGenere le rapport avec : Contexte, Points cles, Analyse, Recommandations, Sources."
            response = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=1000)
            rapport  = response.choices[0].message.content
        st.markdown(rapport)
        st.download_button("⬇️ Télécharger", data=rapport, file_name=titre[:20].replace(" ", "_") + "_" + date.today().strftime("%Y%m%d") + ".txt", mime="text/plain")
        st.warning("⚠️ Ce rapport doit être relu et validé avant diffusion officielle.")
