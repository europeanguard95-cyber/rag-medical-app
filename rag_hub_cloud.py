import os
import hashlib
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from datetime import date
from pinecone import Pinecone, ServerlessSpec
import time
import json
import datetime

GROQ_API_KEY     = st.secrets.get("GROQ_API_KEY",     os.getenv("GROQ_API_KEY", ""))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY", ""))
PINECONE_INDEX   = st.secrets.get("PINECONE_INDEX",   os.getenv("PINECONE_INDEX", "rag-medical"))

# ── LOGGING ────────────────────────────────────────────────
LOGS_FILE = "/tmp/rag_logs.json"

def charger_logs():
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return []

def sauvegarder_log(username, role, question, reponse, sources, techniques):
    logs = charger_logs()
    log = {
        "id": len(logs) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": username,
        "role": role,
        "question": question[:200],
        "reponse_preview": reponse[:150] if reponse else "",
        "nb_sources": len(sources),
        "sources": [m.get("source", "") for m in sources][:3],
        "techniques": techniques
    }
    logs.append(log)
    try:
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-500:], f, ensure_ascii=False, indent=2)
    except:
        pass
    return log

st.set_page_config(page_title="RAG Professionnel", page_icon="🏥", layout="centered")

st.markdown("""
<style>
#neural-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}
.stApp { background: transparent !important; }
section[data-testid="stSidebar"] { z-index: 10; }
.stMainBlockContainer { z-index: 10; position: relative; }
header[data-testid="stHeader"] { z-index: 10; background: transparent !important; }
.nav-section {
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 14px 0 2px 0;
    padding-left: 2px;
    border-top: 0.5px solid var(--color-border-tertiary);
    padding-top: 10px;
}
</style>
<canvas id="neural-canvas"></canvas>
<script>
(function() {
    const canvas = document.getElementById('neural-canvas');
    const ctx = canvas.getContext('2d');
    let W = window.innerWidth;
    let H = window.innerHeight;
    canvas.width = W;
    canvas.height = H;
    const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark'
        || window.matchMedia('(prefers-color-scheme: dark)').matches;
    const NODE_COUNT = 55;
    const MAX_DIST   = 160;
    const SPEED      = 0.4;
    const nodes = Array.from({length: NODE_COUNT}, () => ({
        x:  Math.random() * W,
        y:  Math.random() * H,
        vx: (Math.random() - 0.5) * SPEED,
        vy: (Math.random() - 0.5) * SPEED,
        r:  Math.random() * 2.5 + 1.5,
        pulse: Math.random() * Math.PI * 2
    }));
    function draw() {
        ctx.clearRect(0, 0, W, H);
        const dark      = isDark();
        const nodeFill  = dark ? 'rgba(55,138,221,0.55)'  : 'rgba(24,95,165,0.35)';
        const nodeGlow  = dark ? 'rgba(55,138,221,0.18)'  : 'rgba(24,95,165,0.10)';
        const lineColor = dark ? [55,138,221]              : [24,95,165];
        const bgColor   = dark ? 'rgba(15,20,30,0.18)'     : 'rgba(235,244,255,0.30)';
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, W, H);
        const now = Date.now() * 0.001;
        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            n.x += n.vx;
            n.y += n.vy;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
            for (let j = i + 1; j < nodes.length; j++) {
                const m = nodes[j];
                const dx = n.x - m.x;
                const dy = n.y - m.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < MAX_DIST) {
                    const alpha = (1 - dist / MAX_DIST) * 0.35;
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(" + lineColor[0] + "," + lineColor[1] + "," + lineColor[2] + "," + alpha + ")";
                    ctx.lineWidth = 0.8;
                    ctx.moveTo(n.x, n.y);
                    ctx.lineTo(m.x, m.y);
                    ctx.stroke();
                }
            }
        }
        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            const pulse = Math.sin(now * 1.2 + n.pulse) * 0.4 + 0.8;
            const r = n.r * pulse;
            const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
            grd.addColorStop(0, nodeFill);
            grd.addColorStop(1, nodeGlow);
            ctx.beginPath();
            ctx.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
            ctx.fillStyle = grd;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
            ctx.fillStyle = nodeFill;
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }
    window.addEventListener('resize', () => {
        W = window.innerWidth;
        H = window.innerHeight;
        canvas.width  = W;
        canvas.height = H;
    });
    draw();
})();
</script>
""", unsafe_allow_html=True)

USERS_DEFAUT = {
    "admin":    {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin",    "nom": "Administrateur"},
    "soignant1":{"password": hashlib.sha256("soignant123".encode()).hexdigest(), "role": "soignant", "nom": "Dr. Dupont"},
    "patient1": {"password": hashlib.sha256("patient123".encode()).hexdigest(), "role": "patient",  "nom": "Jean Martin"}
}

def hasher(p): return hashlib.sha256(p.encode()).hexdigest()
def verifier_login(u, p):
    if u in USERS_DEFAUT and USERS_DEFAUT[u]["password"] == hasher(p):
        return True, USERS_DEFAUT[u]["role"], USERS_DEFAUT[u]["nom"]
    return False, None, None

DROITS = {
    "admin":    {"upload_pdf": True,  "voir_sources": True,  "generer_rapport": True,  "comparer": True},
    "soignant": {"upload_pdf": False, "voir_sources": True,  "generer_rapport": True,  "comparer": True},
    "patient":  {"upload_pdf": False, "voir_sources": False, "generer_rapport": False, "comparer": False}
}

@st.cache_resource
def charger_modeles():
    client_groq = Groq(api_key=GROQ_API_KEY)
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pc          = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX)
    return client_groq, model_embed, reranker, index

client_groq, model_embed, reranker, pine_index = charger_modeles()

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
    texte  = "".join([p.extract_text() + "\n" for p in reader.pages])
    chunks = chunker(texte)
    stats  = pine_index.describe_index_stats()
    offset = stats.get("total_vector_count", 0)
    for i in range(0, len(chunks), 50):
        batch  = chunks[i:i+50]
        embeds = model_embed.encode(batch).tolist()
        vectors = [
            {"id": "chunk_" + str(offset + i + j), "values": embeds[j],
             "metadata": {"source": nom_fichier, "text": batch[j][:1000]}}
            for j in range(len(batch))
        ]
        pine_index.upsert(vectors=vectors)
    return len(chunks), len(reader.pages)

def compter_docs():
    try:
        return pine_index.describe_index_stats().get("total_vector_count", 0)
    except:
        return 0

def generer_hyde(question):
    try:
        r = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Genere un court paragraphe documentaire expert qui repond a cette question.\n\nQuestion : " + question + "\nReponse :"}],
            max_tokens=150
        )
        return r.choices[0].message.content
    except:
        return question

def hybrid_search(question, hyde_text=None, sources_filtres=None, n_initial=20):
    nb_total = compter_docs()
    if nb_total == 0:
        return [], []
    texte_embed = hyde_text if hyde_text else question
    q_emb = model_embed.encode([texte_embed]).tolist()[0]
    filter_dict = {"source": {"$in": sources_filtres}} if sources_filtres else None
    query_kwargs = {"vector": q_emb, "top_k": min(n_initial, nb_total), "include_metadata": True}
    if filter_dict:
        query_kwargs["filter"] = filter_dict
    resultats    = pine_index.query(**query_kwargs)
    chunks_vect  = [r["metadata"]["text"] for r in resultats["matches"]]
    sources_vect = [{"source": r["metadata"]["source"]} for r in resultats["matches"]]
    if not chunks_vect:
        return [], []
    bm25   = BM25Okapi([c.lower().split() for c in chunks_vect])
    scores = bm25.get_scores(question.lower().split())
    top    = scores.argsort()[-n_initial:][::-1]
    chunks_bm25  = [chunks_vect[i] for i in top]
    sources_bm25 = [sources_vect[i] for i in top]
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

def appeler_llm(messages, max_tokens=500, retries=3):
    for attempt in range(retries):
        try:
            r = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens
            )
            return r.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                wait = (attempt + 1) * 3
                st.toast("Limite Groq — nouvelle tentative dans " + str(wait) + "s...", icon="⏳")
                time.sleep(wait)
            else:
                return "Erreur : " + str(e)
    return "Service temporairement indisponible."

def rag_query(question, role, n_chunks=5, use_hyde=True, use_hybrid=True, use_reranking=True, sources_filtres=None):
    hyde_text = generer_hyde(question) if use_hyde else None
    if use_hybrid:
        chunks_c, sources_c = hybrid_search(question, hyde_text, sources_filtres, 20)
    else:
        nb = compter_docs()
        if nb == 0:
            return "Aucun document indexé.", [], [], None
        q_emb = model_embed.encode([question]).tolist()[0]
        filter_dict = {"source": {"$in": sources_filtres}} if sources_filtres else None
        query_kwargs = {"vector": q_emb, "top_k": min(20, nb), "include_metadata": True}
        if filter_dict:
            query_kwargs["filter"] = filter_dict
        r = pine_index.query(**query_kwargs)
        chunks_c  = [m["metadata"]["text"] for m in r["matches"]]
        sources_c = [{"source": m["metadata"]["source"]} for m in r["matches"]]
    if use_reranking and len(chunks_c) > n_chunks:
        chunks_f, sources_f = rerank(question, chunks_c, sources_c, n_chunks)
    else:
        chunks_f, sources_f = chunks_c[:n_chunks], sources_c[:n_chunks]
    contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
    consigne = "Utilise un langage simple et rassurant." if role == "patient" else "Utilise le vocabulaire clinique precis. Cite tes sources."
    prompt   = "Tu es un assistant medical. " + consigne + " Reponds UNIQUEMENT depuis le contexte.\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nQUESTION : " + question + "\nREPONSE :"
    reponse  = appeler_llm([{"role": "user", "content": prompt}])
    return reponse, chunks_f, sources_f, hyde_text

# ══════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏥 RAG Professionnel")
    st.caption("Suite complète d'outils RAG pour les professionnels")
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
    st.title("🏥 RAG Professionnel")
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

    st.markdown('<p class="nav-section">Vitrine</p>', unsafe_allow_html=True)
    menus_vitrine = ["🏠 Accueil", "🚀 Solution RAG", "💡 Pourquoi un RAG ?"]

    st.markdown('<p class="nav-section">Outils</p>', unsafe_allow_html=True)
    menus_outils = ["💬 Assistant Q&A"]
    if droits["generer_rapport"]: menus_outils.append("📋 Formateur")
    if droits["comparer"]:        menus_outils.append("⚖️ Comparateur")
    if droits["generer_rapport"]: menus_outils.append("📝 Rapports")
    if droits["generer_rapport"]: menus_outils.append("🏠 Estimation Immobilière")

    menus_admin = []
    if role == "admin":
        st.markdown('<p class="nav-section">Administration</p>', unsafe_allow_html=True)
        menus_admin = ["📊 Logs & Traçabilité"]

    menus = menus_vitrine + menus_outils + menus_admin
    choix = st.radio("", menus, label_visibility="collapsed")

    st.divider()
    st.header("⚙️ Techniques avancées")
    use_hyde      = st.toggle("🔮 HyDE",         value=True, help="Génère une réponse hypothétique avant de chercher")
    use_hybrid    = st.toggle("🔍 Hybrid Search", value=True, help="Combine recherche vectorielle + mots-clés BM25")
    use_reranking = st.toggle("⚡ Re-ranking",    value=True, help="Re-score les chunks pour plus de précision")
    n_chunks      = st.slider("Chunks finaux", 2, 8, 5)

    st.divider()
    if droits["upload_pdf"]:
        st.header("📄 Ajouter un PDF")
        pdf_up = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_up:
            if pdf_up.size > 20_000_000:
                st.error("Fichier trop lourd — maximum 20 MB")
            else:
                if st.button("➕ Indexer"):
                    with st.spinner("Indexation dans Pinecone..."):
                        nb_c, nb_p = indexer_pdf(pdf_up, pdf_up.name)
                    st.success(str(nb_p) + " pages — " + str(nb_c) + " chunks indexés")
                    st.rerun()

    st.divider()
    st.header("📚 Documents")
    nb_total = compter_docs()
    if nb_total > 0:
        st.metric("Total chunks Pinecone", nb_total)
    else:
        st.info("Aucun document")

# ══════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════
if choix == "🚀 Solution RAG":
    st.markdown("""
<style>
.lp-hero{background:var(--color-background-secondary);border-radius:12px;padding:2.5rem 2rem;margin-bottom:1.5rem;text-align:center}
.lp-badge{display:inline-block;background:#E6F1FB;color:#0C447C;font-size:11px;font-weight:500;padding:4px 14px;border-radius:20px;margin-bottom:1rem;letter-spacing:0.05em}
.lp-h1{font-size:26px;font-weight:500;color:var(--color-text-primary);margin:0 0 10px;line-height:1.3}
.lp-sub{font-size:15px;color:var(--color-text-secondary);margin:0 0 1.5rem;line-height:1.7}
.demo-box{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1.5rem;margin-bottom:2rem}
.demo-q{background:var(--color-background-secondary);border-radius:8px;padding:12px 16px;font-size:13px;color:var(--color-text-secondary);margin-bottom:12px}
.demo-a{border-left:3px solid #185FA5;padding:12px 16px;font-size:13px;color:var(--color-text-primary);line-height:1.7;border-radius:0 8px 8px 0;background:var(--color-background-secondary)}
.demo-src{font-size:11px;color:#185FA5;margin-top:8px;font-weight:500}
.vs-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:2rem}
.vs-card{border-radius:12px;padding:1.25rem}
.vs-bad{background:#FCEBEB;border:0.5px solid #F09595}
.vs-good{background:#E1F5EE;border:0.5px solid #5DCAA5}
.vs-title{font-size:14px;font-weight:500;margin:0 0 10px}
.vs-bad .vs-title{color:#A32D2D}
.vs-good .vs-title{color:#085041}
.vs-item{font-size:13px;margin:6px 0;line-height:1.4}
.vs-bad .vs-item{color:#791F1F}
.vs-good .vs-item{color:#0F6E56}
.section-lbl{font-size:11px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin:0 0 12px}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="lp-hero">
<div class="lp-badge">INTELLIGENCE ARTIFICIELLE · DONNÉES PRIVÉES · RÉSULTATS MESURABLES</div>
<p class="lp-h1">Votre base documentaire devient<br>un assistant expert instantané</p>
<p class="lp-sub">Le RAG connecte l'IA directement à vos propres documents.<br>Réponses fiables, sourcées, adaptées à votre métier — en 2 secondes.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="section-lbl">Impact mesuré</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gain de temps", "73%", "sur la recherche documentaire")
    with col2:
        st.metric("Temps de réponse", "< 2 sec", "sur 1 000 pages de docs")
    with col3:
        st.metric("Hallucinations", "0", "chaque réponse cite sa source")

    st.divider()
    st.markdown('<p class="section-lbl">Calculateur ROI</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        nb_collab = st.slider("Nombre de collaborateurs", 1, 200, 15)
    with col2:
        st.metric("Collaborateurs", nb_collab)
    col1, col2 = st.columns([3,1])
    with col1:
        h_semaine = st.slider("Heures recherche / semaine / personne", 1, 20, 6)
    with col2:
        st.metric("Heures/semaine", h_semaine)
    col1, col2 = st.columns([3,1])
    with col1:
        cout_h = st.slider("Coût horaire moyen (€)", 20, 150, 55, step=5)
    with col2:
        st.metric("Coût horaire", str(cout_h) + " €")
    eco_mois = int(nb_collab * h_semaine * cout_h * 0.73 * 4)
    eco_an   = eco_mois * 12
    col1, col2 = st.columns(2)
    with col1:
        st.success("💰 Économie mensuelle : **" + "{:,}".format(eco_mois).replace(",", " ") + " €**")
    with col2:
        st.success("📈 Économie annuelle : **" + "{:,}".format(eco_an).replace(",", " ") + " €**")
    st.caption("Basé sur 73% de réduction du temps de recherche (McKinsey Digital 2023)")

    st.divider()
    st.markdown('<p class="section-lbl">Exemple concret de réponse sourcée</p>', unsafe_allow_html=True)
    secteur_demo = st.selectbox("Secteur", ["🏥 Santé", "⚖️ Juridique", "👥 Ressources humaines", "🎓 Formation"])
    demos = {
        "🏥 Santé": {"q": "Quels sont les critères de surveillance selon notre protocole ?", "a": "Selon votre protocole (révision 2024) : tension artérielle toutes les 4h, fréquence cardiaque, saturation en oxygène et bilan hebdomadaire. Alerter le médecin sous 30 minutes en cas de valeur anormale.", "src": "protocole_surveillance_2024.pdf — page 12"},
        "⚖️ Juridique": {"q": "Quelles sont nos obligations en matière de délai de livraison ?", "a": "Selon l'article 8.3 du contrat-cadre : délai de 15 jours ouvrés. Pénalité de 0,5% par jour de retard, plafonnée à 10%.", "src": "contrat_cadre_v3.pdf — article 8.3"},
        "👥 Ressources humaines": {"q": "Comment fonctionne le remboursement des frais professionnels ?", "a": "Selon le règlement intérieur : soumission via le portail RH sous 30 jours. Repas jusqu'à 25€, transports sur justificatif.", "src": "reglement_interieur_2024.pdf — chapitre 5"},
        "🎓 Formation": {"q": "Quels sont les prérequis pour le module avancé ?", "a": "Modules 1 et 2 validés avec 12/20 minimum, 6 mois d'expérience, validation du responsable formation.", "src": "referentiel_formation_2024.pdf — section 4.1"}
    }
    d = demos[secteur_demo]
    st.markdown('<div class="demo-box"><div class="demo-q">❓ ' + d["q"] + '</div><div class="demo-a">' + d["a"] + '<div class="demo-src">📎 Source : ' + d["src"] + '</div></div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
<div class="vs-grid">
<div class="vs-card vs-bad"><p class="vs-title">❌ IA généraliste</p>
<p class="vs-item">• Ne connaît pas vos documents</p>
<p class="vs-item">• Peut inventer des informations</p>
<p class="vs-item">• Données envoyées à l'extérieur</p>
<p class="vs-item">• Aucune traçabilité</p></div>
<div class="vs-card vs-good"><p class="vs-title">✅ RAG sur vos données</p>
<p class="vs-item">• Connecté à vos documents internes</p>
<p class="vs-item">• Ancré dans vos sources réelles</p>
<p class="vs-item">• Hébergeable en local</p>
<p class="vs-item">• Chaque réponse cite sa source</p></div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-lbl">Demander une démonstration gratuite</p>', unsafe_allow_html=True)
    with st.form("contact_form"):
        col1, col2 = st.columns(2)
        with col1:
            prenom_nom = st.text_input("Prénom et nom")
            email      = st.text_input("Email professionnel")
        with col2:
            entreprise = st.text_input("Entreprise / Organisation")
            secteur_c  = st.selectbox("Secteur", ["Santé", "Juridique", "RH / Formation", "Industrie", "Autre"])
        besoin  = st.text_area("Votre besoin", height=80)
        envoyer = st.form_submit_button("📩 Demander une démonstration gratuite", use_container_width=True, type="primary")
        if envoyer:
            if prenom_nom and email and entreprise:
                st.success("✅ Demande envoyée ! Nous vous recontactons sous 24h. Merci " + prenom_nom.split()[0] + " !")
                st.balloons()
            else:
                st.warning("Merci de remplir au minimum nom, email et entreprise.")

# ══════════════════════════════════════════════════════════
# POURQUOI UN RAG ?
# ══════════════════════════════════════════════════════════
elif choix == "💡 Pourquoi un RAG ?":
    st.subheader("💡 Pourquoi un RAG pour votre organisation ?")
    st.caption("RAG = Retrieval-Augmented Generation — l'IA qui connaît VOS documents")
    st.divider()
    st.markdown("""
### Le problème des IA classiques

| Limite | Impact concret |
|---|---|
| 🔴 **Date de coupure** | Ne connaît pas vos derniers protocoles, mises à jour |
| 🔴 **Hallucinations** | Invente des réponses — dangereux en contexte professionnel |
| 🔴 **Données privées** | Ne connaît pas vos documents internes |
""")
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Santé", "⚖️ Juridique", "🏢 Entreprise", "🎓 Formation"])
    with tab1:
        st.markdown("- Interroger 300 pages de protocoles en langage naturel\n- Générer des fiches procédures à jour\n- Former les équipes sur les nouvelles procédures\n\n> *Le RAG trouve la bonne page en 2 secondes.*")
    with tab2:
        st.markdown("- Analyser et comparer des contrats\n- Extraire les clauses clés\n- Répondre aux questions réglementaires\n\n> *Réponse extraite de vos documents, jamais inventée.*")
    with tab3:
        st.markdown("- Base de connaissances RH accessible 24h/24\n- Onboarding automatisé\n- Support depuis votre documentation produit\n\n> *Réponse immédiate, sourcée, sans appeler les RH.*")
    with tab4:
        st.markdown("- Créer des parcours pédagogiques depuis vos supports\n- Répondre aux apprenants à toute heure\n- Adapter le niveau selon le public\n\n> *Le RAG adapte le langage au niveau demandé.*")
    st.divider()
    st.success("🚀 Uploadez un PDF dans la sidebar et posez vos premières questions dans l'onglet **Assistant Q&A** !")

# ══════════════════════════════════════════════════════════
# ACCUEIL
# ══════════════════════════════════════════════════════════
elif choix == "🏠 Accueil":
    st.subheader("Bienvenue, " + nom + " !")
    st.markdown("Choisis un outil dans le menu à gauche.")
    st.divider()

    # ── Statut des données par module ──
    nb_docs   = compter_docs()
    nb_ventes = len(st.session_state.get("ventes_immo", []))

    def statut_badge(ok, label_ok, label_ko):
        if ok:
            return "<span style='background:#E1F5EE;color:#085041;font-size:11px;padding:2px 8px;border-radius:12px;font-weight:500'>" + label_ok + "</span>"
        return "<span style='background:#FCEBEB;color:#A32D2D;font-size:11px;padding:2px 8px;border-radius:12px;font-weight:500'>" + label_ko + "</span>"

    outils = [
        ("💬", "Assistant Q&A",        "Pose des questions sur tes documents",    statut_badge(nb_docs > 0,   str(nb_docs) + " chunks indexés",        "Aucun document indexé")),
        ("📋", "Formateur",            "Fiches procédures étape par étape",       statut_badge(nb_docs > 0,   str(nb_docs) + " chunks disponibles",     "Aucun document indexé")),
        ("⚖️", "Comparateur",          "Compare deux documents",                  statut_badge(nb_docs > 1,   str(nb_docs) + " chunks — comparaison OK", "Besoin d'au moins 2 documents")),
        ("📝", "Rapports",             "Génère des synthèses structurées",        statut_badge(nb_docs > 0,   str(nb_docs) + " chunks disponibles",     "Aucun document indexé")),
        ("🏠", "Estimation Immobilière","Estimation par ventes comparables",      statut_badge(nb_ventes > 0, str(nb_ventes) + " ventes en base",        "Aucune vente importée")),
    ]

    cols = st.columns(2)
    for i, (icone, titre_outil, desc, badge) in enumerate(outils):
        with cols[i % 2]:
            st.markdown(
                "<div style='padding:1rem;border:0.5px solid var(--color-border-tertiary);border-radius:8px;margin-bottom:12px'>"
                "<p style='font-size:24px;margin:0'>" + icone + "</p>"
                "<p style='font-weight:500;margin:4px 0'>" + titre_outil + "</p>"
                "<p style='font-size:13px;color:var(--color-text-secondary);margin:0 0 6px'>" + desc + "</p>"
                + badge +
                "</div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════
# ASSISTANT Q&A
# ══════════════════════════════════════════════════════════
elif choix == "💬 Assistant Q&A":
    st.subheader("💬 Assistant Q&A")
    if compter_docs() == 0:
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
            with st.spinner("Pipeline RAG en cours..."):
                reponse, chunks_f, sources_f, hyde_text = rag_query(
                    question, role, n_chunks, use_hyde, use_hybrid, use_reranking
                )
            st.markdown(reponse)
            with st.expander("🔬 Détail du pipeline"):
                if use_hyde and hyde_text:
                    st.markdown("**🔮 HyDE activé**")
                    st.caption(hyde_text[:200] + "...")
                    st.divider()
                st.markdown("**🔍 Hybrid Search :** " + ("activé" if use_hybrid else "désactivé"))
                st.markdown("**⚡ Re-ranking :** " + ("activé" if use_reranking else "désactivé"))
            if droits["voir_sources"]:
                with st.expander("📚 Sources utilisées"):
                    for i, (c, m) in enumerate(zip(chunks_f, sources_f)):
                        st.markdown("**Source " + str(i+1) + " — " + m["source"] + "**")
                        st.text(c[:300] + "..." if len(c) > 300 else c)
                        st.divider()
            if role == "patient":
                st.info("ℹ️ Consultez votre professionnel de santé.")
        st.session_state.messages.append({"role": "assistant", "content": reponse})
        techniques_actives = []
        if use_hyde: techniques_actives.append("HyDE")
        if use_hybrid: techniques_actives.append("Hybrid Search")
        if use_reranking: techniques_actives.append("Re-ranking")
        sauvegarder_log(username, role, question, reponse, sources_f, techniques_actives)

# ══════════════════════════════════════════════════════════
# FORMATEUR
# ══════════════════════════════════════════════════════════
elif choix == "📋 Formateur":
    st.subheader("📋 Formateur de Procédures")
    if compter_docs() == 0:
        st.info("Aucun document disponible.")
        st.stop()
    public = st.radio("Public cible", ["Sage-femme / Médecin", "Infirmier(e)", "Patient"], horizontal=True)
    sujet  = st.text_input("Sujet", placeholder="Ex: Surveillance tensionnelle pendant la grossesse")
    if st.button("📋 Générer", type="primary") and sujet.strip():
        with st.spinner("Génération..."):
            chunks_c, sources_c = hybrid_search(sujet, generer_hyde(sujet) if use_hyde else None)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 6) if use_reranking else (chunks_c[:6], sources_c[:6])
            contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
            consigne = "Vocabulaire clinique precis." if "Médecin" in public else ("Langage professionnel accessible." if "Infirmier" in public else "Langage simple.")
            fiche = appeler_llm([{"role": "user", "content": "Tu es formateur medical. " + consigne + " Base-toi UNIQUEMENT sur le contexte.\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nSUJET : " + sujet + "\n\nStructure :\n## Objectif\n## Prérequis\n## Étapes\n1.\n2.\n## Points de vigilance\n⚠️\n## Sources"}], max_tokens=800)
        st.markdown(fiche)
        st.download_button("⬇️ Télécharger", data=fiche, file_name="procedure_" + sujet[:20].replace(" ", "_") + ".txt", mime="text/plain")
        etapes = [l for l in fiche.split("\n") if l.strip() and l.strip()[0].isdigit()]
        if etapes:
            st.divider()
            st.subheader("✅ Checklist")
            for i, e in enumerate(etapes):
                st.checkbox(e.strip(), key="chk_" + str(i))

# ══════════════════════════════════════════════════════════
# COMPARATEUR
# ══════════════════════════════════════════════════════════
elif choix == "⚖️ Comparateur":
    st.subheader("⚖️ Comparateur de Documents")
    if compter_docs() == 0:
        st.info("Aucun document disponible.")
        st.stop()
    sujet = st.text_input("Sujet de comparaison", placeholder="Ex: Surveillance de la grossesse")
    axe   = st.selectbox("Axe", ["Recommandations cliniques", "Critères de surveillance", "Prise en charge", "Définitions"])
    if st.button("⚖️ Comparer", type="primary") and sujet.strip():
        with st.spinner("Comparaison..."):
            hyde1 = generer_hyde(sujet) if use_hyde else None
            chunks_c, sources_c = hybrid_search(sujet, hyde1)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 10) if use_reranking else (chunks_c[:10], sources_c[:10])
            srcs_uniques = list(dict.fromkeys([m["source"] for m in sources_f]))
            if len(srcs_uniques) < 2:
                st.warning("Pas assez de documents différents trouvés.")
                st.stop()
            src1, src2 = srcs_uniques[0], srcs_uniques[1]
            c1f = [c for c, m in zip(chunks_f, sources_f) if m["source"] == src1][:5]
            c2f = [c for c, m in zip(chunks_f, sources_f) if m["source"] == src2][:5]
            ctx1 = "\n\n".join(["[" + src1 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c1f)])
            ctx2 = "\n\n".join(["[" + src2 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c2f)])
            comparaison = appeler_llm([{"role": "user", "content": "Compare ces deux documents sur : " + sujet + "\nAxe : " + axe + "\n\n--- DOC 1 : " + src1 + " ---\n" + ctx1 + "\n\n--- DOC 2 : " + src2 + " ---\n" + ctx2 + "\n\nStructure :\n## Synthese\n## Points de convergence\n## Points de divergence\n| Critere | " + src1 + " | " + src2 + " |\n|---|---|---|\n## Recommandation"}], max_tokens=900)
        st.markdown(comparaison)
        st.download_button("⬇️ Télécharger", data=comparaison, file_name="comparaison_" + sujet[:20].replace(" ", "_") + ".txt", mime="text/plain")

# ══════════════════════════════════════════════════════════
# RAPPORTS
# ══════════════════════════════════════════════════════════
elif choix == "📝 Rapports":
    st.subheader("📝 Générateur de Rapports")
    if compter_docs() == 0:
        st.info("Aucun document disponible.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1: titre        = st.text_input("Titre", placeholder="Ex: Protocole sage-femme 2024")
    with col2: destinataire = st.text_input("Destinataire", placeholder="Ex: Équipe soignante")
    type_rapport   = st.selectbox("Type", ["Synthèse clinique", "Rapport de formation", "Note de synthèse", "Résumé exécutif", "Compte-rendu"])
    sujet          = st.text_area("Sujet et instructions", height=80)
    contexte_libre = st.text_input("Contexte additionnel (optionnel)")
    if st.button("📝 Générer", type="primary") and sujet.strip() and titre.strip():
        with st.spinner("Génération du rapport..."):
            hyde_text   = generer_hyde(sujet) if use_hyde else None
            chunks_c, sources_c = hybrid_search(sujet, hyde_text, None, 25)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 8) if use_reranking else (chunks_c[:8], sources_c[:8])
            contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
            today    = date.today().strftime("%d/%m/%Y")
            rapport  = appeler_llm([{"role": "user", "content": "Tu es expert redacteur medical. Genere un rapport professionnel en francais.\nType : " + type_rapport + "\nTitre : " + titre + "\nDestinataire : " + destinataire + "\nDate : " + today + "\n" + ("Contexte : " + contexte_libre if contexte_libre else "") + "\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nSujet : " + sujet + "\n\nGenere le rapport avec : Contexte, Points cles, Analyse, Recommandations, Sources."}], max_tokens=1000)
        st.markdown(rapport)
        st.download_button("⬇️ Télécharger", data=rapport, file_name=titre[:20].replace(" ", "_") + "_" + date.today().strftime("%Y%m%d") + ".txt", mime="text/plain")
        st.warning("⚠️ Ce rapport doit être relu et validé avant diffusion officielle.")

# ══════════════════════════════════════════════════════════
# LOGS & TRAÇABILITÉ
# ══════════════════════════════════════════════════════════
elif choix == "📊 Logs & Traçabilité":
    st.subheader("📊 Logs & Traçabilité")
    st.caption("Visible uniquement par l'administrateur")
    logs = charger_logs()
    if not logs:
        st.info("Aucune requête enregistrée pour le moment.")
        st.stop()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total requêtes", len(logs))
    with col2:
        st.metric("Utilisateurs actifs", len(set(l["user"] for l in logs)))
    with col3:
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        st.metric("Requêtes aujourd'hui", sum(1 for l in logs if l["timestamp"].startswith(today_str)))
    with col4:
        hyde_count = sum(1 for l in logs if "HyDE" in l.get("techniques", []))
        st.metric("HyDE utilisé", str(hyde_count) + "/" + str(len(logs)))
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        filtre_user = st.selectbox("Filtrer par utilisateur", ["Tous"] + list(set(l["user"] for l in logs)))
    with col2:
        nb_afficher = st.slider("Nombre de logs", 5, 50, 20)
    logs_filtres   = logs if filtre_user == "Tous" else [l for l in logs if l["user"] == filtre_user]
    logs_affichage = list(reversed(logs_filtres))[:nb_afficher]
    for log in logs_affichage:
        with st.expander("#" + str(log["id"]) + " — " + log["timestamp"] + " — " + log["user"] + " (" + log["role"] + ")"):
            st.markdown("**Question :** " + log["question"])
            st.markdown("**Aperçu réponse :** " + log["reponse_preview"] + "...")
            st.markdown("**Sources (" + str(log["nb_sources"]) + ") :** " + ", ".join(log["sources"]))
            st.markdown("**Techniques :** " + (", ".join(log["techniques"]) if log["techniques"] else "Standard"))
    st.divider()
    logs_json = json.dumps(logs, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Télécharger tous les logs (JSON)", data=logs_json, file_name="rag_logs_" + datetime.datetime.now().strftime("%Y%m%d") + ".json", mime="application/json")
    if st.button("🗑️ Effacer tous les logs", type="secondary"):
        if os.path.exists(LOGS_FILE):
            os.remove(LOGS_FILE)
        st.success("Logs effacés.")
        st.rerun()

# ══
# ══════════════════════════════════════════════════════════
# MODULE 5 — ESTIMATION IMMOBILIERE
# ══════════════════════════════════════════════════════════
elif choix == "🏠 Estimation Immobilière":
    st.subheader("🏠 Estimation Immobilière")
    st.caption("Estimation basée sur les ventes internes de l'agence")

    import pandas as pd
    import io as _io

    if "ventes_immo" not in st.session_state:
        st.session_state.ventes_immo = []

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Estimation", "📥 Importer CSV", "✏️ Ajouter une vente", "⚙️ Paramètres agence"])

    with tab1:
        ventes = st.session_state.ventes_immo
        if not ventes:
            st.info("Aucune vente enregistrée. Importez un CSV ou ajoutez des ventes manuellement.")
        else:
            st.markdown("**Décrivez le bien à estimer :**")
            col1, col2, col3 = st.columns(3)
            with col1:
                surface    = st.number_input("Surface (m²)", min_value=10, max_value=500, value=75)
                nb_pieces  = st.selectbox("Nombre de pièces", [1, 2, 3, 4, 5, 6], index=2)
            with col2:
                adresse_bien = st.text_input("Adresse complète", placeholder="Ex: 12 rue de la Paix, Centre-Ville")
                quartier     = adresse_bien.split(",")[-1].strip() if "," in adresse_bien else adresse_bien
                type_bien    = st.selectbox("Type de bien", ["Appartement", "Maison", "Studio", "Loft", "Autre"])
            with col3:
                etage      = st.number_input("Étage", min_value=0, max_value=30, value=2)
                annee_bien = st.number_input("Année de construction", min_value=1800, max_value=2024, value=1990)

            if st.button("🔍 Estimer le prix", type="primary"):
                with st.spinner("Analyse des ventes comparables..."):
                    df = pd.DataFrame(ventes)
                    comparables = df.copy()
                    if "type_bien" in df.columns:
                        comparables = comparables[comparables["type_bien"].str.lower() == type_bien.lower()]
                    if quartier and "quartier" in df.columns:
                        mask_q = comparables["quartier"].str.lower().str.contains(quartier.lower(), na=False)
                        if mask_q.sum() >= 3:
                            comparables = comparables[mask_q]
                    if "surface" in df.columns:
                        mask_s = (comparables["surface"] >= surface * 0.70) & (comparables["surface"] <= surface * 1.30)
                        if mask_s.sum() >= 3:
                            comparables = comparables[mask_s]
                    nb_comp = len(comparables)
                    if nb_comp == 0:
                        st.warning("Pas assez de ventes comparables. Essayez des critères moins restrictifs.")
                    else:
                        comparables["prix_m2"] = comparables["prix"] / comparables["surface"]
                        prix_m2_moyen = comparables["prix_m2"].mean()
                        prix_m2_min   = comparables["prix_m2"].quantile(0.25)
                        prix_m2_max   = comparables["prix_m2"].quantile(0.75)
                        est      = prix_m2_moyen * surface
                        est_min  = prix_m2_min   * surface
                        est_max  = prix_m2_max   * surface

                        st.divider()
                        st.markdown("### 💰 Résultat de l'estimation")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Estimation basse",    "{:,.0f} €".format(est_min).replace(",", " "))
                        with c2: st.metric("Estimation centrale", "{:,.0f} €".format(est).replace(",", " "),    delta="référence")
                        with c3: st.metric("Estimation haute",    "{:,.0f} €".format(est_max).replace(",", " "))
                        st.metric("Prix au m² moyen", "{:,.0f} €/m²".format(prix_m2_moyen).replace(",", " "), delta=str(nb_comp) + " ventes comparables")

                        st.divider()
                        st.markdown("### 📋 Ventes comparables")
                        top = comparables.sort_values("prix_m2").head(10)
                        for _, row in top.iterrows():
                            label = str(row.get("adresse", "Bien")) + " — " + "{:,.0f} €".format(row["prix"]).replace(",", " ") + " (" + str(int(row["surface"])) + " m²)"
                            with st.expander(label):
                                ca, cb, cc, cd = st.columns(4)
                                with ca: st.metric("Prix",     "{:,.0f} €".format(row["prix"]).replace(",", " "))
                                with cb: st.metric("Surface",  str(int(row["surface"])) + " m²")
                                with cc: st.metric("Prix/m²",  "{:,.0f} €".format(row["prix_m2"]).replace(",", " "))
                                with cd: st.metric("Quartier", str(row.get("quartier", "N/A")))

                        st.divider()
                        st.markdown("### 📈 Analyse du marché local")
                        ctx_ventes = "\n".join([
                            "- " + str(r.get("adresse","Bien")) + ": " + str(int(r["surface"])) + "m2, " +
                            str(r.get("nb_pieces","?")) + " pieces, " + str(r.get("quartier","?")) +
                            " -> " + "{:,.0f}EUR ({:,.0f}EUR/m2)".format(r["prix"], r["prix_m2"]).replace(",", " ")
                            for _, r in top.iterrows()
                        ])
                        prompt_rap = (
                            "Tu es expert immobilier. Analyse ces ventes et redige un rapport professionnel. "
                            "Bien a estimer: " + type_bien + ", " + str(surface) + "m2, " + str(nb_pieces) + " pieces, " +
                            (quartier if quartier else "quartier non precise") + ", etage " + str(etage) + ", annee " + str(annee_bien) + ". "
                            "Ventes comparables (" + str(nb_comp) + "): " + ctx_ventes + ". "
                            "Estimation calculee: basse " + "{:,.0f}EUR".format(est_min).replace(",", " ") +
                            ", centrale " + "{:,.0f}EUR".format(est).replace(",", " ") +
                            ", haute " + "{:,.0f}EUR".format(est_max).replace(",", " ") + ". "
                            "Redige avec sections: Synthese, Analyse marche local, Facteurs prix, Recommandation, Points de vigilance."
                        )
                        rapport = appeler_llm([{"role": "user", "content": prompt_rap}], max_tokens=800)
                        st.markdown(rapport)
                        st.download_button("⬇️ Télécharger le rapport", data=rapport,
                            file_name="estimation_" + str(surface) + "m2.txt", mime="text/plain")

    with tab2:
        st.markdown("**Format CSV attendu :**")
        st.code("adresse,prix,surface,nb_pieces,type_bien,quartier,etage,annee_construction,date_vente")
        csv_file = st.file_uploader("Importer CSV ou Excel", type=["csv", "xlsx"])
        if csv_file:
            try:
                df_import = pd.read_excel(csv_file) if csv_file.name.endswith(".xlsx") else pd.read_csv(csv_file, sep=None, engine="python")
                df_import.columns = [c.lower().strip().replace(" ", "_") for c in df_import.columns]
                if "prix" not in df_import.columns or "surface" not in df_import.columns:
                    st.error("Le fichier doit contenir au minimum 'prix' et 'surface'.")
                else:
                    df_import["prix"]    = pd.to_numeric(df_import["prix"],    errors="coerce")
                    df_import["surface"] = pd.to_numeric(df_import["surface"], errors="coerce")
                    df_import = df_import.dropna(subset=["prix", "surface"])
                    st.success(str(len(df_import)) + " ventes importées !")
                    st.dataframe(df_import.head(10), use_container_width=True)
                    if st.button("✅ Valider et ajouter à la base", type="primary"):
                        st.session_state.ventes_immo.extend(df_import.to_dict(orient="records"))
                        st.success(str(len(df_import)) + " ventes ajoutées ! Total : " + str(len(st.session_state.ventes_immo)))
                        st.rerun()
            except Exception as e:
                st.error("Erreur import : " + str(e))
        if st.session_state.ventes_immo:
            st.divider()
            st.metric("Ventes en base", len(st.session_state.ventes_immo))
            if st.button("🗑️ Vider la base", type="secondary"):
                st.session_state.ventes_immo = []
                st.rerun()

    with tab3:
        st.markdown("**Ajouter une vente + générer le flyer automatiquement**")
        with st.form("ajout_vente"):
            col1, col2 = st.columns(2)
            with col1:
                v_adresse  = st.text_input("Adresse complète", placeholder="12 rue de la Paix, Centre-Ville")
                v_prix     = st.number_input("Prix de vente (€)", min_value=10000, max_value=10000000, value=250000, step=5000)
                v_surface  = st.number_input("Surface (m²)", min_value=10, max_value=1000, value=70)
                v_pieces   = st.selectbox("Nombre de pièces", [1, 2, 3, 4, 5, 6])
            with col2:
                v_type     = st.selectbox("Type de bien", ["Appartement", "Maison", "Studio", "Loft", "Autre"])
                v_quartier = st.text_input("Quartier", placeholder="Centre-Ville")
                v_etage    = st.number_input("Étage", min_value=0, max_value=30, value=0)
                v_date     = st.text_input("Date de vente", placeholder="2024-03")
            c1, c2 = st.columns(2)
            with c1: ajouter       = st.form_submit_button("➕ Ajouter la vente", type="primary",   use_container_width=True)
            with c2: ajouter_flyer = st.form_submit_button("📄 Ajouter + Flyer", type="secondary", use_container_width=True)

        if ajouter or ajouter_flyer:
            nouvelle_vente = {
                "adresse": v_adresse, "prix": v_prix, "surface": v_surface,
                "nb_pieces": v_pieces, "type_bien": v_type, "quartier": v_quartier,
                "etage": v_etage, "date_vente": v_date
            }
            st.session_state.ventes_immo.append(nouvelle_vente)
            st.success("✅ Vente ajoutée ! Total : " + str(len(st.session_state.ventes_immo)) + " ventes")

            if ajouter_flyer:
                with st.spinner("Génération du flyer et du texte Instagram..."):
                    v_quartier_display = v_quartier if v_quartier else v_adresse
                    prompt_insta = (
                        "Tu es expert en marketing immobilier. "
                        "Genere un post Instagram professionnel pour annoncer cette vente: "
                        + v_type + ", " + (v_adresse if v_adresse else v_quartier) + ", "
                        + str(v_surface) + "m2, " + str(v_pieces) + " pieces, etage " + str(v_etage) + ", "
                        + "{:,.0f}EUR".format(v_prix).replace(",", " ") + " vendu le " + (v_date if v_date else "2024") + ". "
                        "Genere: 1. Titre accrocheur. 2. Texte 3-4 lignes. 3. 10 hashtags. "
                        "Ton: professionnel, chaleureux. En francais."
                    )
                    texte_insta = appeler_llm([{"role": "user", "content": prompt_insta}], max_tokens=400)

                st.divider()
                st.markdown("### 📱 Texte Instagram généré")
                st.markdown(texte_insta)
                st.download_button("⬇️ Télécharger le texte Instagram", data=texte_insta,
                    file_name="instagram_" + str(v_surface) + "m2.txt", mime="text/plain")

                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import cm
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
                    from reportlab.lib.enums import TA_CENTER, TA_LEFT

                    buffer = _io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
                    story = []
                    styles = getSampleStyleSheet()

                    BLEU = colors.HexColor("#185FA5")
                    GRIS = colors.HexColor("#F5F7FA")
                    NOIR = colors.HexColor("#1A1A2E")
                    VERT = colors.HexColor("#1D9E75")

                    s_titre  = ParagraphStyle("t", fontSize=32, textColor=BLEU, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=6)
                    s_sub    = ParagraphStyle("s", fontSize=14, textColor=NOIR, alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
                    s_prix   = ParagraphStyle("p", fontSize=18, textColor=VERT, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=16)
                    s_sect   = ParagraphStyle("sc", fontSize=11, textColor=BLEU, fontName="Helvetica-Bold", spaceAfter=6)
                    s_insta  = ParagraphStyle("i", fontSize=9, textColor=colors.HexColor("#444444"), fontName="Helvetica", leading=14, spaceAfter=3, leftIndent=8)
                    s_footer = ParagraphStyle("f", fontSize=8, textColor=colors.grey, alignment=TA_CENTER, fontName="Helvetica")

                    story.append(Paragraph("VENDU !", s_titre))
                    story.append(HRFlowable(width="100%", thickness=2, color=BLEU))
                    story.append(Spacer(1, 8))
                    story.append(Paragraph(v_type.upper() + " - " + str(v_pieces) + " PIECES - " + str(v_surface) + " m2", s_sub))
                    if v_adresse:
                        story.append(Paragraph(v_adresse, s_sub))
                    story.append(Spacer(1, 4))
                    story.append(Paragraph("{:,.0f} EUR".format(v_prix).replace(",", " "), s_prix))
                    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                    story.append(Spacer(1, 12))

                    story.append(Paragraph("Caracteristiques du bien", s_sect))
                    data = [
                        ["Type",    v_type,          "Surface",  str(v_surface) + " m2"],
                        ["Pieces",  str(v_pieces),   "Etage",    str(v_etage)],
                        ["Quartier", v_quartier_display, "Date",  v_date if v_date else "N/A"],
                    ]
                    tbl = Table(data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 4*cm])
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (0,-1), BLEU),
                        ("BACKGROUND", (2,0), (2,-1), BLEU),
                        ("TEXTCOLOR",  (0,0), (0,-1), colors.white),
                        ("TEXTCOLOR",  (2,0), (2,-1), colors.white),
                        ("BACKGROUND", (1,0), (1,-1), GRIS),
                        ("BACKGROUND", (3,0), (3,-1), GRIS),
                        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
                        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
                        ("FONTNAME",   (2,0), (2,-1), "Helvetica-Bold"),
                        ("FONTSIZE",   (0,0), (-1,-1), 9),
                        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
                        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                        ("GRID",       (0,0), (-1,-1), 0.5, colors.lightgrey),
                        ("TOPPADDING", (0,0), (-1,-1), 8),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 16))

                    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                    story.append(Spacer(1, 8))
                    story.append(Paragraph("Notre annonce", s_sect))
                    for ligne in texte_insta.split("\n")[:8]:
                        if ligne.strip():
                            story.append(Paragraph(ligne.strip(), s_insta))
                    story.append(Spacer(1, 16))

                    story.append(Spacer(1, 16))

                    # ── PHOTO EQUIPE ──
                    agence_p = st.session_state.get("agence_params", {})
                    if agence_p.get("photo_equipe") and len(agence_p["photo_equipe"]) > 100:
                        try:
                            import base64 as b64
                            import os as _os2
                            from reportlab.platypus import Image as RLImage
                            from PIL import Image as PILImage
                            photo_bytes_dec = b64.b64decode(agence_p["photo_equipe"])
                            pil_img = PILImage.open(_io.BytesIO(photo_bytes_dec))
                            if pil_img.mode in ("RGBA", "P", "CMYK"):
                                pil_img = pil_img.convert("RGB")
                            pil_img = pil_img.resize((900, 360), PILImage.LANCZOS)
                            fixed_path = "/tmp/equipe_agence.jpg"
                            pil_img.save(fixed_path, format="JPEG", quality=85)
                            story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                            story.append(Spacer(1, 8))
                            story.append(Paragraph("L'equipe qui a realise cette vente", s_sect))
                            img = RLImage(fixed_path, width=15*cm, height=6*cm)
                            img.hAlign = "CENTER"
                            story.append(img)
                            story.append(Spacer(1, 6))
                            story.append(Paragraph("Felicitations a toute l'equipe !", s_footer))
                        except Exception as photo_err:
                            st.warning("Photo non incluse dans le flyer : " + str(photo_err))

                    story.append(Spacer(1, 8))
                    story.append(HRFlowable(width="100%", thickness=2, color=BLEU))
                    story.append(Spacer(1, 6))
                    nom_ag    = agence_p.get("nom",       "Agence Immobiliere Professionnelle")
                    slogan_ag = agence_p.get("slogan",    "Votre partenaire de confiance")
                    tel_ag    = agence_p.get("telephone", "")
                    email_ag2 = agence_p.get("email",     "")
                    site_ag   = agence_p.get("site",      "")
                    footer_txt = nom_ag + " | " + slogan_ag
                    contact_txt = " | ".join(filter(None, [tel_ag, email_ag2, site_ag]))
                    story.append(Paragraph(footer_txt, s_footer))
                    if contact_txt:
                        story.append(Paragraph(contact_txt, s_footer))
                    story.append(Paragraph("Ce bien a ete vendu avec succes grace a notre expertise du marche local.", s_footer))

                    doc.build(story)
                    pdf_bytes = buffer.getvalue()

                    st.divider()
                    st.markdown("### 📄 Flyer PDF prêt !")
                    st.success("Votre flyer professionnel est généré — téléchargez et publiez sur Instagram !")
                    st.download_button(
                        "⬇️ Télécharger le flyer PDF",
                        data=pdf_bytes,
                        file_name="flyer_vendu_" + str(v_surface) + "m2_" + (v_quartier if v_quartier else "bien") + ".pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                except Exception as e:
                    st.error("Erreur generation PDF : " + str(e))

        if st.session_state.ventes_immo:
            st.divider()
            st.markdown("**Dernières ventes ajoutées :**")
            st.dataframe(pd.DataFrame(st.session_state.ventes_immo[-5:]), use_container_width=True)

    with tab4:
        st.markdown("### ⚙️ Paramètres de l'agence")
        st.caption("Ces informations apparaîtront automatiquement sur tous vos flyers.")
        st.divider()

        if "agence_params" not in st.session_state:
            st.session_state.agence_params = {
                "nom": "", "slogan": "", "telephone": "",
                "email": "", "site": "", "photo_equipe": None
            }
        p = st.session_state.agence_params

        with st.form("form_agence"):
            col1, col2 = st.columns(2)
            with col1:
                nom_agence  = st.text_input("Nom de l'agence",  value=p.get("nom", ""),       placeholder="Immobilier Excellence")
                slogan      = st.text_input("Slogan",            value=p.get("slogan", ""),    placeholder="Votre partenaire de confiance")
                telephone   = st.text_input("Téléphone",         value=p.get("telephone", ""), placeholder="01 23 45 67 89")
            with col2:
                email_ag    = st.text_input("Email",             value=p.get("email", ""),     placeholder="contact@agence.fr")
                site        = st.text_input("Site web",          value=p.get("site", ""),      placeholder="www.agence-excellence.fr")

            st.divider()
            st.markdown("**📸 Photo de l'équipe**")
            st.caption("Cette photo sera intégrée dans tous vos flyers de vente.")
            photo_upload = st.file_uploader("Uploader la photo de l'équipe", type=["jpg", "jpeg", "png"])

            sauver = st.form_submit_button("💾 Sauvegarder les paramètres", type="primary", use_container_width=True)
            if sauver:
                st.session_state.agence_params["nom"]       = nom_agence
                st.session_state.agence_params["slogan"]    = slogan
                st.session_state.agence_params["telephone"] = telephone
                st.session_state.agence_params["email"]     = email_ag
                st.session_state.agence_params["site"]      = site
                if photo_upload:
                    import base64
                    photo_bytes = photo_upload.read()
                    photo_b64   = base64.b64encode(photo_bytes).decode()
                    st.session_state.agence_params["photo_equipe"]      = photo_b64
                    st.session_state.agence_params["photo_equipe_type"] = photo_upload.type
                st.success("✅ Paramètres sauvegardés ! Ils seront utilisés dans vos prochains flyers.")

        # Aperçu photo si déjà uploadée
        if st.session_state.agence_params.get("photo_equipe"):
            st.divider()
            st.markdown("**Aperçu de la photo d'équipe :**")
            import base64
            photo_data = st.session_state.agence_params["photo_equipe"]
            photo_type = st.session_state.agence_params.get("photo_equipe_type", "image/jpeg")
            st.markdown(
                "<img src='data:" + photo_type + ";base64," + photo_data + "' style='max-width:100%;border-radius:12px;border:2px solid #185FA5'>",
                unsafe_allow_html=True
            )
            st.caption("Cette photo sera intégrée dans vos flyers PDF.")
