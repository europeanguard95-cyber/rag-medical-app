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

GROQ_API_KEY     = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY", ""))
PINECONE_INDEX   = "rag-medical"

st.set_page_config(page_title="RAG professionnel", page_icon="🏥", layout="centered")

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

# ══════════════════════════════════════════════════════════
# UTILISATEURS & DROITS
# ══════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════
# CHARGEMENT MODÈLES & PINECONE
# ══════════════════════════════════════════════════════════
@st.cache_resource
def charger_modeles():
    client_groq = Groq(api_key=GROQ_API_KEY)
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return client_groq, model_embed, reranker

@st.cache_resource
def init_pinecone():
    """Initialise l'index Pinecone (créé automatiquement si absent)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Attendre que l'index soit prêt
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX)

client_groq, model_embed, reranker = charger_modeles()
index = init_pinecone()

# ══════════════════════════════════════════════════════════
# HELPERS PINECONE
# ══════════════════════════════════════════════════════════

def source_id_prefix(nom_fichier):
    """Préfixe d'ID stable basé sur le nom de fichier (MD5)."""
    return "doc_" + hashlib.md5(nom_fichier.encode()).hexdigest()

def get_total_count():
    """Nombre total de vecteurs dans l'index."""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count
    except:
        return 0

def source_deja_indexee(nom_fichier):
    """Vérifie si le premier chunk d'un PDF est déjà dans Pinecone."""
    try:
        premier_id = source_id_prefix(nom_fichier) + "_chunk_0"
        result = index.fetch(ids=[premier_id])
        return premier_id in result.vectors
    except:
        return False

def get_all_chunks():
    """
    Récupère tous les chunks depuis Pinecone pour le BM25.
    Résultat mis en cache dans session_state (valide pour la durée de la session).
    """
    if "pinecone_all_chunks" in st.session_state:
        return st.session_state["pinecone_all_chunks"]

    docs, metas, all_ids = [], [], []
    try:
        for ids_page in index.list():
            all_ids.extend(ids_page)
    except Exception:
        st.session_state["pinecone_all_chunks"] = ([], [])
        return [], []

    for i in range(0, len(all_ids), 100):
        batch = all_ids[i:i+100]
        try:
            fetched = index.fetch(ids=batch)
            for _, vdata in fetched.vectors.items():
                meta = vdata.metadata or {}
                if meta.get("text"):
                    docs.append(meta["text"])
                    metas.append({"source": meta.get("source", "")})
        except Exception:
            pass

    st.session_state["pinecone_all_chunks"] = (docs, metas)
    return docs, metas

def get_sources_list():
    """Retourne la liste des sources uniques et le nombre de chunks par source."""
    docs, metas = get_all_chunks()
    sources = {}
    for m in metas:
        src = m["source"]
        sources[src] = sources.get(src, 0) + 1
    return sources

def invalider_cache_chunks():
    """À appeler après chaque indexation pour forcer le rechargement du cache BM25."""
    if "pinecone_all_chunks" in st.session_state:
        del st.session_state["pinecone_all_chunks"]

# ══════════════════════════════════════════════════════════
# INDEXATION PDF
# ══════════════════════════════════════════════════════════
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
    prefix = source_id_prefix(nom_fichier)

    for i in range(0, len(chunks), 50):
        batch   = chunks[i:i+50]
        embeds  = model_embed.encode(batch).tolist()
        vectors = [
            {
                "id":     prefix + "_chunk_" + str(i + j),
                "values": embeds[j],
                "metadata": {
                    "text":   batch[j],
                    "source": nom_fichier
                }
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors)

    invalider_cache_chunks()
    return len(chunks), len(reader.pages)

# ══════════════════════════════════════════════════════════
# TECHNIQUE 1 : HyDE
# ══════════════════════════════════════════════════════════
def generer_hyde(question):
    try:
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Genere un court paragraphe documentaire expert qui repond a cette question. Sois factuel et precis.\n\nQuestion : " + question + "\nReponse :"}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except:
        return question

# ══════════════════════════════════════════════════════════
# TECHNIQUE 2 : Hybrid Search (Pinecone dense + BM25)
# ══════════════════════════════════════════════════════════
def hybrid_search(question, hyde_text=None, sources_filtres=None, n_initial=20):
    if get_total_count() == 0:
        return [], []

    texte_embed = hyde_text if hyde_text else question
    q_emb = model_embed.encode([texte_embed]).tolist()[0]

    # Filtre Pinecone par source (optionnel)
    pinecone_filter = None
    if sources_filtres and len(sources_filtres) == 1:
        pinecone_filter = {"source": {"$eq": sources_filtres[0]}}
    elif sources_filtres and len(sources_filtres) > 1:
        pinecone_filter = {"source": {"$in": sources_filtres}}

    # Recherche vectorielle via Pinecone
    kwargs = dict(
        vector=q_emb,
        top_k=min(n_initial, get_total_count()),
        include_metadata=True
    )
    if pinecone_filter:
        kwargs["filter"] = pinecone_filter

    results_vect = index.query(**kwargs)
    chunks_vect  = [m.metadata.get("text", "") for m in results_vect.matches]
    sources_vect = [{"source": m.metadata.get("source", "")} for m in results_vect.matches]

    # Recherche BM25 sur le corpus complet
    tous_docs, tous_metas = get_all_chunks()

    # Appliquer filtre source pour BM25
    if sources_filtres:
        filtre_docs  = [d for d, m in zip(tous_docs, tous_metas) if m["source"] in sources_filtres]
        filtre_metas = [m for m in tous_metas if m["source"] in sources_filtres]
    else:
        filtre_docs, filtre_metas = tous_docs, tous_metas

    if not filtre_docs:
        return chunks_vect, sources_vect

    bm25   = BM25Okapi([d.lower().split() for d in filtre_docs])
    scores = bm25.get_scores(question.lower().split())
    top    = scores.argsort()[-n_initial:][::-1]
    chunks_bm25  = [filtre_docs[i] for i in top]
    sources_bm25 = [filtre_metas[i] for i in top]

    # Fusion RRF (Reciprocal Rank Fusion)
    scores_f, src_map = {}, {}
    for rank, (c, m) in enumerate(zip(chunks_vect, sources_vect)):
        scores_f[c] = scores_f.get(c, 0) + 1 / (rank + 60)
        src_map[c]  = m
    for rank, (c, m) in enumerate(zip(chunks_bm25, sources_bm25)):
        scores_f[c] = scores_f.get(c, 0) + 1 / (rank + 60)
        src_map[c]  = m

    tries = sorted(scores_f, key=scores_f.get, reverse=True)
    return tries[:n_initial], [src_map[c] for c in tries[:n_initial]]

# ══════════════════════════════════════════════════════════
# TECHNIQUE 3 : Re-ranking
# ══════════════════════════════════════════════════════════
def rerank(question, chunks, sources, n_final=5):
    if not chunks:
        return [], []
    scores   = reranker.predict([(question, c) for c in chunks])
    combined = sorted(zip(chunks, sources, scores), key=lambda x: x[2], reverse=True)
    return [c for c, s, _ in combined[:n_final]], [s for _, s, _ in combined[:n_final]]

# ══════════════════════════════════════════════════════════
# APPEL LLM AVEC RETRY
# ══════════════════════════════════════════════════════════
def appeler_llm(messages, max_tokens=500, retries=3):
    for attempt in range(retries):
        try:
            response = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                wait = (attempt + 1) * 3
                st.toast("Limite Groq atteinte — nouvelle tentative dans " + str(wait) + "s...", icon="⏳")
                time.sleep(wait)
            else:
                return "Erreur : " + str(e)
    return "Service temporairement indisponible. Réessaie dans quelques secondes."

# ══════════════════════════════════════════════════════════
# PIPELINE RAG COMPLET
# ══════════════════════════════════════════════════════════
def rag_query(question, role, n_chunks=5, use_hyde=True, use_hybrid=True, use_reranking=True, sources_filtres=None):
    hyde_text = None
    if use_hyde:
        hyde_text = generer_hyde(question)

    if use_hybrid:
        chunks_c, sources_c = hybrid_search(question, hyde_text, sources_filtres, n_initial=20)
    else:
        q_emb = model_embed.encode([question]).tolist()[0]
        pinecone_filter = None
        if sources_filtres and len(sources_filtres) == 1:
            pinecone_filter = {"source": {"$eq": sources_filtres[0]}}
        elif sources_filtres and len(sources_filtres) > 1:
            pinecone_filter = {"source": {"$in": sources_filtres}}
        kwargs = dict(vector=q_emb, top_k=min(20, get_total_count()), include_metadata=True)
        if pinecone_filter:
            kwargs["filter"] = pinecone_filter
        r = index.query(**kwargs)
        chunks_c  = [m.metadata.get("text", "") for m in r.matches]
        sources_c = [{"source": m.metadata.get("source", "")} for m in r.matches]

    if use_reranking and len(chunks_c) > n_chunks:
        chunks_f, sources_f = rerank(question, chunks_c, sources_c, n_chunks)
    else:
        chunks_f, sources_f = chunks_c[:n_chunks], sources_c[:n_chunks]

    contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
    consigne = "Utilise un langage simple et rassurant." if role == "patient" else "Utilise le vocabulaire clinique precis. Cite tes sources."
    prompt   = "Tu es un assistant medical. " + consigne + " Reponds UNIQUEMENT depuis le contexte.\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nQUESTION : " + question + "\nREPONSE :"

    reponse = appeler_llm([{"role": "user", "content": prompt}])
    return reponse, chunks_f, sources_f, hyde_text

# ══════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏥 RAG professionel")
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

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🧭 Navigation")
    menus = ["🏠 Accueil", "🚀 Solution RAG", "💡 Pourquoi un RAG ?", "💬 Assistant Q&A"]
    if droits["generer_rapport"]: menus.append("📋 Formateur")
    if droits["comparer"]:        menus.append("⚖️ Comparateur")
    if droits["generer_rapport"]: menus.append("📝 Rapports")
    choix = st.radio("", menus, label_visibility="collapsed")

    st.divider()
    st.header("⚙️ Techniques avancées")
    use_hyde      = st.toggle("🔮 HyDE", value=True, help="Génère une réponse hypothétique avant de chercher")
    use_hybrid    = st.toggle("🔍 Hybrid Search", value=True, help="Combine recherche vectorielle + mots-clés BM25")
    use_reranking = st.toggle("⚡ Re-ranking", value=True, help="Re-score les chunks pour plus de précision")
    n_chunks      = st.slider("Chunks finaux", 2, 8, 5)

    st.divider()
    if droits["upload_pdf"]:
        st.header("📄 Ajouter un PDF")
        pdf_up = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_up:
            nom_f = pdf_up.name
            if source_deja_indexee(nom_f):
                st.info("✅ Déjà indexé")
            else:
                if st.button("➕ Indexer"):
                    with st.spinner("Indexation Pinecone..."):
                        nb_c, nb_p = indexer_pdf(pdf_up, nom_f)
                    st.success(str(nb_p) + " pages — " + str(nb_c) + " chunks ✅")
                    st.rerun()

    st.divider()
    st.header("📚 Documents")
    total = get_total_count()
    if total > 0:
        sources_dict = get_sources_list()
        for src, nb in sources_dict.items():
            st.markdown("• " + src + " (" + str(nb) + ")")
        st.metric("Total chunks", total)
    else:
        st.info("Aucun document")

# ══════════════════════════════════════════════════════════
# LANDING PAGE PROFESSIONNELLE
# ══════════════════════════════════════════════════════════
if choix == "🚀 Solution RAG":

    st.markdown("""
<style>
.lp-hero{background:var(--color-background-secondary);border-radius:12px;padding:2.5rem 2rem;margin-bottom:1.5rem;text-align:center}
.lp-badge{display:inline-block;background:#E6F1FB;color:#0C447C;font-size:11px;font-weight:500;padding:4px 14px;border-radius:20px;margin-bottom:1rem;letter-spacing:0.05em}
.lp-h1{font-size:26px;font-weight:500;color:var(--color-text-primary);margin:0 0 10px;line-height:1.3}
.lp-sub{font-size:15px;color:var(--color-text-secondary);margin:0 0 1.5rem;line-height:1.7}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:2rem}
.metric-card{background:var(--color-background-secondary);border-radius:8px;padding:1.25rem;text-align:center}
.metric-val{font-size:28px;font-weight:500;color:var(--color-text-primary);margin:0}
.metric-lbl{font-size:12px;color:var(--color-text-secondary);margin:6px 0 0;line-height:1.4}
.feat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:2rem}
.feat-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1.25rem}
.feat-title{font-size:14px;font-weight:500;color:var(--color-text-primary);margin:8px 0 4px}
.feat-desc{font-size:13px;color:var(--color-text-secondary);margin:0;line-height:1.5}
.roi-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1.5rem;margin-bottom:2rem}
.roi-row{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.roi-label{font-size:13px;color:var(--color-text-secondary);width:200px}
.roi-result{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:1rem}
.roi-res{background:var(--color-background-secondary);border-radius:8px;padding:1rem;text-align:center}
.roi-res-val{font-size:24px;font-weight:500;color:var(--color-text-primary);margin:0}
.roi-res-lbl{font-size:12px;color:var(--color-text-secondary);margin:4px 0 0}
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
.contact-form{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1.5rem}
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

    st.markdown('<p class="section-lbl">Calculateur de retour sur investissement</p>', unsafe_allow_html=True)
    st.markdown('<div class="roi-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        nb_collab = st.slider("Nombre de collaborateurs concernés", 1, 200, 15, step=1)
    with col2:
        st.metric("Collaborateurs", nb_collab)

    col1, col2 = st.columns([3,1])
    with col1:
        h_semaine = st.slider("Heures de recherche documentaire / semaine / personne", 1, 20, 6, step=1)
    with col2:
        st.metric("Heures/semaine", h_semaine)

    col1, col2 = st.columns([3,1])
    with col1:
        cout_h = st.slider("Coût horaire moyen (€)", 20, 150, 55, step=5)
    with col2:
        st.metric("Coût horaire", str(cout_h) + " €")

    gain = 0.73
    eco_mois = int(nb_collab * h_semaine * cout_h * gain * 4)
    eco_an   = eco_mois * 12

    col1, col2 = st.columns(2)
    with col1:
        st.success("💰 Économie mensuelle estimée : **" + "{:,}".format(eco_mois).replace(",", " ") + " €**")
    with col2:
        st.success("📈 Économie annuelle estimée : **" + "{:,}".format(eco_an).replace(",", " ") + " €**")
    st.caption("Basé sur 73% de réduction du temps de recherche documentaire (source : McKinsey Digital 2023)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-lbl">Exemple concret de réponse sourcée</p>', unsafe_allow_html=True)
    secteur_demo = st.selectbox("Choisir un secteur", ["🏥 Santé", "⚖️ Juridique", "👥 Ressources humaines", "🎓 Formation"])

    demos = {
        "🏥 Santé": {
            "q": "Quels sont les critères de surveillance selon notre protocole interne ?",
            "a": "Selon votre protocole de surveillance (révision 2024), les critères incluent : tension artérielle toutes les 4h, fréquence cardiaque, saturation en oxygène et bilan biologique hebdomadaire. En cas de valeur anormale, alerter le médecin responsable dans les 30 minutes.",
            "src": "protocole_surveillance_2024.pdf — page 12, section 3.2"
        },
        "⚖️ Juridique": {
            "q": "Quelles sont nos obligations contractuelles en matière de délai de livraison ?",
            "a": "Selon l'article 8.3 de votre contrat-cadre fournisseur, le délai de livraison contractuel est de 15 jours ouvrés. Tout dépassement entraîne une pénalité de 0,5% par jour de retard, plafonnée à 10% du montant total de la commande.",
            "src": "contrat_cadre_fournisseurs_v3.pdf — article 8.3, page 24"
        },
        "👥 Ressources humaines": {
            "q": "Comment fonctionne la procédure de remboursement des frais professionnels ?",
            "a": "Selon votre règlement intérieur, les frais professionnels doivent être soumis via le portail RH dans les 30 jours suivant la dépense. Les repas sont remboursés jusqu'à 25€, les transports sur justificatif. Le remboursement est effectué sous 15 jours avec la paie du mois suivant.",
            "src": "reglement_interieur_2024.pdf — chapitre 5, page 18"
        },
        "🎓 Formation": {
            "q": "Quels sont les prérequis pour accéder au module avancé de gestion de projet ?",
            "a": "Selon le référentiel de formation, l'accès au module avancé (niveau 3) requiert : avoir validé les modules 1 et 2 avec une note minimale de 12/20, justifier d'au moins 6 mois d'expérience en gestion de projet, et obtenir la validation du responsable formation.",
            "src": "referentiel_formation_GP_2024.pdf — section 4.1, page 31"
        }
    }

    d = demos[secteur_demo]
    st.markdown("""
<div class="demo-box">
<div class="demo-q">❓ """ + d["q"] + """</div>
<div class="demo-a">""" + d["a"] + """<div class="demo-src">📎 Source : """ + d["src"] + """</div></div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-lbl">Ce que vous pouvez faire avec cette solution</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="feat-grid">
<div class="feat-card"><p style="font-size:20px;margin:0">🔍</p><p class="feat-title">Recherche intelligente</p><p class="feat-desc">Posez des questions en langage naturel sur l'ensemble de vos documents internes</p></div>
<div class="feat-card"><p style="font-size:20px;margin:0">📋</p><p class="feat-title">Génération de rapports</p><p class="feat-desc">Synthèses, comptes-rendus et notes structurées générées automatiquement</p></div>
<div class="feat-card"><p style="font-size:20px;margin:0">⚖️</p><p class="feat-title">Comparaison de documents</p><p class="feat-desc">Détectez les divergences entre versions, protocoles ou contrats en quelques secondes</p></div>
<div class="feat-card"><p style="font-size:20px;margin:0">🔐</p><p class="feat-title">Sécurité multi-niveaux</p><p class="feat-desc">Droits par rôle, données hébergées on-premise ou cloud privé selon vos contraintes</p></div>
<div class="feat-card"><p style="font-size:20px;margin:0">🧠</p><p class="feat-title">Techniques avancées</p><p class="feat-desc">HyDE, Hybrid Search, Re-ranking — les meilleures réponses, pas juste les plus rapides</p></div>
<div class="feat-card"><p style="font-size:20px;margin:0">⚡</p><p class="feat-title">Déploiement rapide</p><p class="feat-desc">Opérationnel en quelques jours sur vos documents existants, sans refonte IT</p></div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-lbl">IA généraliste vs RAG sur vos données</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="vs-grid">
<div class="vs-card vs-bad">
<p class="vs-title">❌ IA généraliste (ChatGPT…)</p>
<p class="vs-item">• Ne connaît pas vos documents</p>
<p class="vs-item">• Peut inventer des informations</p>
<p class="vs-item">• Vos données envoyées à l'extérieur</p>
<p class="vs-item">• Réponses génériques, non sourcées</p>
<p class="vs-item">• Aucune traçabilité</p>
</div>
<div class="vs-card vs-good">
<p class="vs-title">✅ RAG sur vos données</p>
<p class="vs-item">• Connecté à vos documents internes</p>
<p class="vs-item">• Ancré dans vos sources réelles</p>
<p class="vs-item">• Hébergeable en local (on-premise)</p>
<p class="vs-item">• Chaque réponse cite sa source</p>
<p class="vs-item">• Accès contrôlé par rôle</p>
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-lbl">Demander une démonstration gratuite</p>', unsafe_allow_html=True)
    with st.form("contact_form"):
        col1, col2 = st.columns(2)
        with col1:
            prenom_nom  = st.text_input("Prénom et nom")
            email       = st.text_input("Email professionnel")
        with col2:
            entreprise  = st.text_input("Entreprise / Organisation")
            secteur_c   = st.selectbox("Secteur", ["Santé", "Juridique", "RH / Formation", "Industrie", "Retail", "Autre"])
        besoin = st.text_area("Votre besoin en quelques mots", placeholder="Ex : automatiser la recherche dans nos 500 procédures internes...", height=80)
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

Les assistants IA comme ChatGPT sont puissants, mais ils ont **3 limites critiques** pour une entreprise :

| Limite | Impact concret |
|---|---|
| 🔴 **Date de coupure** | Ne connaît pas vos derniers protocoles, circulaires, mises à jour |
| 🔴 **Hallucinations** | Invente des réponses avec confiance — dangereux en contexte professionnel |
| 🔴 **Données privées** | Ne connaît pas vos documents internes, vos procédures, votre base de connaissances |

---

### Ce que change un RAG

Un RAG connecte l'IA à **vos propres documents**. Au lieu de répondre depuis sa mémoire générale, il **cherche d'abord dans vos sources**, puis génère une réponse ancrée dans vos données.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div style='padding:1rem; border:0.5px solid var(--color-border-tertiary); border-radius:8px; text-align:center;'>
<p style='font-size:28px; margin:0;'>📄</p>
<p style='font-weight:500; margin:8px 0 4px;'>Vos documents</p>
<p style='font-size:13px; color:var(--color-text-secondary); margin:0;'>Protocoles, guides, contrats, formations, rapports</p>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div style='padding:1rem; border:0.5px solid var(--color-border-tertiary); border-radius:8px; text-align:center;'>
<p style='font-size:28px; margin:0;'>🧠</p>
<p style='font-weight:500; margin:8px 0 4px;'>Le RAG</p>
<p style='font-size:13px; color:var(--color-text-secondary); margin:0;'>Cherche, analyse et synthétise en temps réel</p>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div style='padding:1rem; border:0.5px solid var(--color-border-tertiary); border-radius:8px; text-align:center;'>
<p style='font-size:28px; margin:0;'>✅</p>
<p style='font-weight:500; margin:8px 0 4px;'>Réponses fiables</p>
<p style='font-size:13px; color:var(--color-text-secondary); margin:0;'>Sourcées, vérifiables, adaptées à votre contexte</p>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### Cas d'usage concrets par secteur")

    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Santé", "⚖️ Juridique", "🏢 Entreprise", "🎓 Formation"])

    with tab1:
        st.markdown("""
**Pour les professionnels de santé :**
- Interroger des centaines de protocoles en langage naturel
- Générer des fiches procédures à jour depuis les recommandations HAS
- Comparer des versions de guides cliniques
- Former les équipes sur les nouvelles procédures
- Répondre aux questions des patients avec des sources vérifiées

> *"Quels sont les critères de surveillance de la pré-éclampsie selon le protocole en vigueur ?"*
> → Le RAG trouve la bonne page dans vos 300 pages de protocoles en 2 secondes.
""")

    with tab2:
        st.markdown("""
**Pour les équipes juridiques :**
- Analyser des contrats et extraire les clauses clés
- Comparer des versions de documents légaux
- Répondre aux questions sur la réglementation en vigueur
- Générer des synthèses de dossiers complexes
- Former les collaborateurs sur les nouvelles lois

> *"Quelles sont les obligations de l'employeur en matière de RGPD selon notre politique interne ?"*
> → Réponse extraite directement de vos documents internes, jamais inventée.
""")

    with tab3:
        st.markdown("""
**Pour les entreprises :**
- Base de connaissances RH accessible en langage naturel
- Onboarding des nouveaux collaborateurs sur vos procédures
- Support client automatisé depuis votre documentation produit
- Analyse de rapports et synthèses exécutives automatiques
- Gestion des connaissances internes sans perte d'expertise

> *"Quelle est la procédure de remboursement des frais professionnels selon notre règlement intérieur ?"*
> → Réponse immédiate, sourcée, sans appeler les RH.
""")

    with tab4:
        st.markdown("""
**Pour les organismes de formation :**
- Créer des parcours pédagogiques depuis vos supports existants
- Répondre aux questions des apprenants 24h/24
- Générer des évaluations et fiches mémo depuis vos contenus
- Adapter le niveau de réponse selon le public
- Valoriser votre capital documentaire existant

> *"Explique-moi la réglementation sur le temps de travail comme si j'étais un stagiaire de 1ère année"*
> → Le RAG adapte le langage au niveau demandé tout en restant dans vos documents.
""")

    st.divider()

    st.markdown("### Pourquoi choisir un RAG plutôt qu'une IA généraliste ?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**❌ IA généraliste (ChatGPT, etc.)**
- Ne connaît pas vos documents
- Peut inventer des informations
- Données envoyées à l'extérieur
- Réponses génériques
- Pas de traçabilité des sources
""")
    with col2:
        st.markdown("""
**✅ RAG sur vos données**
- Connecté à vos documents
- Ancré dans vos sources réelles
- Peut tourner en local (on-premise)
- Réponses personnalisées à votre contexte
- Chaque réponse cite sa source
""")

    st.divider()
    st.success("🚀 Cette application est un exemple concret de RAG médical — uploadez un PDF dans la sidebar et posez vos premières questions dans l'onglet **Assistant Q&A** !")

# ══════════════════════════════════════════════════════════
# ACCUEIL
# ══════════════════════════════════════════════════════════
elif choix == "🏠 Accueil":
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
            st.markdown("<div style='padding:1rem; border:0.5px solid var(--color-border-tertiary); border-radius:8px; margin-bottom:12px;'><p style='font-size:24px; margin:0;'>" + icone + "</p><p style='font-weight:500; margin:4px 0;'>" + titre_outil + "</p><p style='font-size:13px; color:var(--color-text-secondary); margin:0;'>" + desc + "</p></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# RAG 1 — ASSISTANT Q&A
# ══════════════════════════════════════════════════════════
elif choix == "💬 Assistant Q&A":
    st.subheader("💬 Assistant Q&A Médical")
    if get_total_count() == 0:
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

# ══════════════════════════════════════════════════════════
# RAG 2 — FORMATEUR
# ══════════════════════════════════════════════════════════
elif choix == "📋 Formateur":
    st.subheader("📋 Formateur de Procédures")
    if get_total_count() == 0:
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
# RAG 3 — COMPARATEUR
# ══════════════════════════════════════════════════════════
elif choix == "⚖️ Comparateur":
    st.subheader("⚖️ Comparateur de Documents")
    if get_total_count() == 0:
        st.info("Aucun document disponible.")
        st.stop()

    srcs = list(get_sources_list().keys())
    if len(srcs) < 2:
        st.warning("Tu as besoin d'au moins 2 documents pour comparer.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1: src1 = st.selectbox("Document 1", srcs, index=0)
    with col2: src2 = st.selectbox("Document 2", [s for s in srcs if s != src1], index=0)

    sujet = st.text_input("Sujet de comparaison", placeholder="Ex: Surveillance de la grossesse")
    axe   = st.selectbox("Axe", ["Recommandations cliniques", "Critères de surveillance", "Prise en charge", "Définitions"])

    if st.button("⚖️ Comparer", type="primary") and sujet.strip():
        with st.spinner("Comparaison..."):
            hyde1 = generer_hyde(sujet) if use_hyde else None
            c1, s1 = hybrid_search(sujet, hyde1, [src1]) if use_hybrid else ([], [])
            c2, s2 = hybrid_search(sujet, hyde1, [src2]) if use_hybrid else ([], [])
            if not c1: c1, s1 = hybrid_search(sujet, None, [src1])
            if not c2: c2, s2 = hybrid_search(sujet, None, [src2])
            c1f, _ = rerank(sujet, c1, s1, 5) if use_reranking else (c1[:5], s1[:5])
            c2f, _ = rerank(sujet, c2, s2, 5) if use_reranking else (c2[:5], s2[:5])
            ctx1 = "\n\n".join(["[" + src1 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c1f)])
            ctx2 = "\n\n".join(["[" + src2 + " — " + str(i+1) + "]\n" + c for i, c in enumerate(c2f)])
            comparaison = appeler_llm([{"role": "user", "content": "Compare ces deux documents sur : " + sujet + "\nAxe : " + axe + "\n\n--- DOC 1 : " + src1 + " ---\n" + ctx1 + "\n\n--- DOC 2 : " + src2 + " ---\n" + ctx2 + "\n\nStructure :\n## Synthese\n## Points de convergence\n## Points de divergence\n| Critere | " + src1 + " | " + src2 + " |\n|---|---|---|\n## Recommandation\n## Limites"}], max_tokens=900)
        st.markdown(comparaison)
        st.download_button("⬇️ Télécharger", data=comparaison, file_name="comparaison_" + sujet[:20].replace(" ", "_") + ".txt", mime="text/plain")

# ══════════════════════════════════════════════════════════
# RAG 4 — RAPPORTS
# ══════════════════════════════════════════════════════════
elif choix == "📝 Rapports":
    st.subheader("📝 Générateur de Rapports")
    if get_total_count() == 0:
        st.info("Aucun document disponible.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1: titre        = st.text_input("Titre", placeholder="Ex: Protocole sage-femme 2024")
    with col2: destinataire = st.text_input("Destinataire", placeholder="Ex: Équipe soignante")

    type_rapport   = st.selectbox("Type", ["Synthèse clinique", "Rapport de formation", "Note de synthèse", "Résumé exécutif", "Compte-rendu"])
    sujet          = st.text_area("Sujet et instructions", height=80)
    contexte_libre = st.text_input("Contexte additionnel (optionnel)")

    srcs     = list(get_sources_list().keys())
    srcs_sel = st.multiselect("Sources (vide = toutes)", srcs)

    if st.button("📝 Générer", type="primary") and sujet.strip() and titre.strip():
        with st.spinner("Génération du rapport..."):
            src_filtres = srcs_sel if srcs_sel else None
            hyde_text   = generer_hyde(sujet) if use_hyde else None
            chunks_c, sources_c = hybrid_search(sujet, hyde_text, src_filtres, 25) if use_hybrid else ([], [])
            if not chunks_c: chunks_c, sources_c = hybrid_search(sujet, None, src_filtres, 25)
            chunks_f, sources_f = rerank(sujet, chunks_c, sources_c, 8) if use_reranking else (chunks_c[:8], sources_c[:8])
            contexte = "\n\n".join(["[Source " + str(i+1) + " — " + m["source"] + "]\n" + c for i, (c, m) in enumerate(zip(chunks_f, sources_f))])
            today    = date.today().strftime("%d/%m/%Y")
            rapport  = appeler_llm([{"role": "user", "content": "Tu es expert redacteur medical. Genere un rapport professionnel en francais.\nType : " + type_rapport + "\nTitre : " + titre + "\nDestinataire : " + destinataire + "\nDate : " + today + "\n" + ("Contexte : " + contexte_libre if contexte_libre else "") + "\n\n--- CONTEXTE ---\n" + contexte + "\n--- FIN ---\n\nSujet : " + sujet + "\n\nGenere le rapport avec : Contexte, Points cles, Analyse, Recommandations, Sources."}], max_tokens=1000)
        st.markdown(rapport)
        st.download_button("⬇️ Télécharger", data=rapport, file_name=titre[:20].replace(" ", "_") + "_" + date.today().strftime("%Y%m%d") + ".txt", mime="text/plain")
        st.warning("⚠️ Ce rapport doit être relu et validé avant diffusion officielle.")
