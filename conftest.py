import sys
import os
from unittest.mock import MagicMock


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return MagicMock()
    def __setattr__(self, key, value):
        self[key] = value


_ss = _SessionState({
    "logged_in": True,
    "username":  "admin",
    "role":      "admin",
    "nom":       "Administrateur",
    "messages":  [],
})

_pc_index = MagicMock()
_pc_index.describe_index_stats.return_value.total_vector_count = 0
_pc_index.list.return_value = []
_pc_index.query.return_value.matches = []

_pc_client = MagicMock()
_pc_client.list_indexes.return_value = []
_pc_client.Index.return_value = _pc_index

_pinecone_mod = MagicMock()
_pinecone_mod.Pinecone.return_value = _pc_client


def _columns(spec):
    n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [MagicMock() for _ in range(n)]


def _tabs(labels):
    return [MagicMock() for _ in labels]


_st = MagicMock()
_st.session_state  = _ss
_st.secrets        = {}
_st.cache_resource = lambda f: f
_st.stop           = MagicMock()
_st.columns        = _columns
_st.tabs           = _tabs
_st.button.return_value             = False
_st.form_submit_button.return_value = False
_st.file_uploader.return_value      = None
_st.chat_input.return_value         = None
_st.toggle.return_value             = False
_st.checkbox.return_value           = False
_st.radio.return_value              = "_NONE_"
_st.selectbox.return_value          = "Sante"
_st.text_input.return_value         = ""
_st.text_area.return_value          = ""
_st.slider.return_value             = 5
_st.multiselect.return_value        = []

sys.modules["streamlit"]             = _st
sys.modules["pinecone"]              = _pinecone_mod
sys.modules["groq"]                  = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pypdf"]                 = MagicMock()
sys.modules["rank_bm25"]             = MagicMock()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
