# app.py - Streamlit Chat Interface

import streamlit as st
import streamlit.components.v1 as components
import os
import logging
import time
import json
import re
from datetime import datetime
from query import answer, compare_documents, generate_followups, detect_coverage_gaps
from dotenv import load_dotenv
import base64
import chromadb

load_dotenv()

# Load chatbot icon as base64 for embedding in HTML
def load_icon_b64(path='chatbot_icon.png'):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ''

ICON_B64 = load_icon_b64()
ICON_DATA_URI = f'data:image/png;base64,{ICON_B64}' if ICON_B64 else ''
ASSISTANT_AVATAR = 'chatbot_icon.png' if os.path.exists('chatbot_icon.png') else None

logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title='MUM Policy Assistant',
    page_icon='chatbot_icon.png',
    layout='wide',
    initial_sidebar_state='expanded'
)

# loading stylesheet
# encoding='utf-8' prevents UnicodeDecodeError on Windows
# style.css defines the LIGHT theme in :root by default. The DARK theme is injected dynamically below based on session state.
with open('style.css', encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# session state initialization
if 'messages'      not in st.session_state: st.session_state.messages      = []
if 'query_count'   not in st.session_state: st.session_state.query_count   = 0
if 'avg_score_all' not in st.session_state: st.session_state.avg_score_all = []
if 'total_time'    not in st.session_state: st.session_state.total_time    = 0.0
if 'answer_length' not in st.session_state: st.session_state.answer_length = 'Standard'
if 'language'      not in st.session_state: st.session_state.language      = 'English'
if 'dark_mode'     not in st.session_state: st.session_state.dark_mode     = False
if 'tts_text'      not in st.session_state: st.session_state.tts_text      = None
if 'tts_counter'   not in st.session_state: st.session_state.tts_counter   = 0
if 'copy_text'     not in st.session_state: st.session_state.copy_text     = None

# Early widget-state sync — Streamlit persists widget values by key from the
# previous run.  Reading them here BEFORE any t() calls ensures that every
# translated label on the page uses the language / settings the user last
# selected, not the stale value from two runs ago.
if 'lang_radio'    in st.session_state: st.session_state.language      = st.session_state.lang_radio
if 'length_slider' in st.session_state: st.session_state.answer_length = st.session_state.length_slider
if 'theme_toggle'  in st.session_state: st.session_state.dark_mode     = st.session_state.theme_toggle

# ═══════════════════════════════════════════════════════════════
# TRANSLATIONS — every UI string in three languages
# ═══════════════════════════════════════════════════════════════
TR = {
    # Sidebar
    'response_length':    {'English': 'Response Length',       'French': 'Longueur de réponse'},
    'detail_level':       {'English': 'Detail level:',         'French': 'Niveau de détail :'},
    'brief':              {'English': '⚡ Key facts only, ~80 words',   'French': '⚡ Faits clés, ~80 mots'},
    'standard':           {'English': '📄 Balanced answer, ~200 words', 'French': '📄 Réponse équilibrée, ~200 mots'},
    'detailed':           {'English': '📚 Full explanation, 300+ words','French': '📚 Explication complète, 300+ mots'},
    'language':           {'English': 'Language',              'French': 'Langue'},
    'quick_search':       {'English': 'Policy Quick Search',   'French': 'Recherche Rapide'},
    'search_label':       {'English': 'Search indexed documents:','French': 'Rechercher les documents :'},
    'search_placeholder': {'English': 'e.g. plagiarism, warning points...','French': 'ex. plagiat, points d\'avertissement...'},
    'no_match':           {'English': 'No matching content found.','French': 'Aucun contenu correspondant.'},
    'doc_index':          {'English': 'Document Index',        'French': 'Index des Documents'},
    'chunks':             {'English': 'Chunks',                'French': 'Fragments'},
    'documents':          {'English': 'Documents',             'French': 'Documents'},
    'no_docs':            {'English': 'No documents indexed. Run `python ingest.py` first.','French': 'Aucun document indexé. Exécutez `python ingest.py`.'},
    'view_all_docs':      {'English': 'View all {n} documents','French': 'Voir les {n} documents'},
    'summariser':         {'English': 'Document Summariser',   'French': 'Résumeur de Documents'},
    'choose_doc':         {'English': 'Choose a document:',    'French': 'Choisir un document :'},
    'gen_summary':        {'English': '📝  Generate Summary',  'French': '📝  Générer un résumé'},
    'index_first':        {'English': 'Index documents first to use the summariser.','French': 'Indexez les documents d\'abord.'},
    'session_stats':      {'English': 'Session Statistics',    'French': 'Statistiques de Session'},
    'queries':            {'English': 'Queries',               'French': 'Requêtes'},
    'avg_conf':           {'English': 'Avg Conf.',             'French': 'Conf. Moy.'},
    'avg_resp_time':      {'English': '⏱ Avg response time:',  'French': '⏱ Temps moyen :'},
    'rated_responses':    {'English': 'out of {n} rated responses','French': 'sur {n} réponses évaluées'},
    'export_chat':        {'English': '💾 Export Chat History', 'French': '💾 Exporter l\'historique'},
    'quick_questions':    {'English': 'Quick Questions',       'French': 'Questions Rapides'},
    'clear':              {'English': '🗑️ Clear',              'French': '🗑️ Effacer'},
    'refresh':            {'English': '🔄 Refresh',            'French': '🔄 Actualiser'},
    # Quick questions
    'qq1': {'English': 'What are the types of student misconduct?',          'French': 'Quels sont les types de faute étudiante ?'},
    'qq2': {'English': 'How many warning points lead to expulsion?',         'French': 'Combien de points mènent à l\'expulsion ?'},
    'qq3': {'English': 'What is the appeal process for disciplinary decisions?','French': 'Quel est le processus d\'appel disciplinaire ?'},
    'qq4': {'English': 'What support is available for students in disciplinary matters?','French': 'Quel soutien est disponible pour les étudiants ?'},
    'qq5': {'English': 'What is the acceptable use policy?',                 'French': 'Quelle est la politique d\'utilisation acceptable ?'},
    'qq6': {'English': 'What is the data protection policy?',                'French': 'Quelle est la politique de protection des données ?'},
    'qq7': {'English': 'What are the safeguarding responsibilities?',        'French': 'Quelles sont les responsabilités de protection ?'},
    # Tabs
    'tab_chat':           {'English': '💬 Chat',               'French': '💬 Discussion'},
    'tab_compare':        {'English': '🔀 Compare Documents',  'French': '🔀 Comparer Documents'},
    # Chat
    'chat_placeholder':   {'English': 'Ask about MUM policies — e.g. What are the types of student misconduct?','French': 'Posez une question — ex. Quels sont les types de faute étudiante ?'},
    'searching':          {'English': 'Searching policy documents...','French': 'Recherche en cours...'},
    'followup':           {'English': '💡 You may also want to ask:','French': '💡 Vous pourriez aussi demander :'},
    'helpful':            {'English': 'This answer was helpful','French': 'Cette réponse était utile'},
    'needs_improvement':  {'English': 'This answer needs improvement','French': 'Cette réponse nécessite amélioration'},
    'thanks_up':          {'English': 'Thanks for your feedback! 👍','French': 'Merci pour votre avis ! 👍'},
    'thanks_down':        {'English': 'Thanks — we will use this to improve. 👎','French': 'Merci — nous utiliserons ceci pour améliorer. 👎'},
    'view_sources':       {'English': 'View {n} source reference(s)','French': 'Voir {n} référence(s)'},
    # Compare tab
    'compare_title':      {'English': '### 🔀 Compare Two Policy Documents','French': '### 🔀 Comparer Deux Documents'},
    'compare_desc':       {'English': 'Select any two documents to generate a structured side-by-side comparison.','French': 'Sélectionnez deux documents pour une comparaison structurée.'},
    'need_2_docs':        {'English': 'You need at least 2 indexed documents to use this feature.','French': 'Il faut au moins 2 documents indexés.'},
    'first_doc':          {'English': '📄 First document:',    'French': '📄 Premier document :'},
    'second_doc':         {'English': '📄 Second document:',   'French': '📄 Deuxième document :'},
    'same_doc_warn':      {'English': 'Please select two different documents to compare.','French': 'Veuillez sélectionner deux documents différents.'},
    'gen_comparison':     {'English': '🔀  Generate Comparison','French': '🔀  Générer la comparaison'},
    'compare_done':       {'English': 'Comparison complete! Also saved to your Chat tab.','French': 'Comparaison terminée ! Aussi sauvegardé dans l\'onglet Discussion.'},
    'comparing':          {'English': 'Comparing {a} and {b}...','French': 'Comparaison de {a} et {b}...'},
    # Footer
    'footer':             {'English': 'Chatbot Assistant &nbsp;·&nbsp; Middlesex University Mauritius &nbsp;·&nbsp; Powered by Google Gemini + ChromaDB',
                           'French':  'Assistant Chatbot &nbsp;·&nbsp; Middlesex University Mauritius &nbsp;·&nbsp; Propulsé par Google Gemini + ChromaDB'},
    # Errors
    'api_quota':          {'English': '⚠️ **API quota reached.** Please wait a few minutes and try again.','French': '⚠️ **Quota API atteint.** Veuillez patienter et réessayer.'},
    'model_404':          {'English': '⚠️ **Model not found.** Please check your API configuration.','French': '⚠️ **Modèle introuvable.** Vérifiez votre configuration API.'},
    'key_missing':        {'English': '⚠️ **API key missing.** Please check your .env file.','French': '⚠️ **Clé API manquante.** Vérifiez votre fichier .env.'},
    'gen_error':          {'English': '⚠️ **An error occurred:** ','French': '⚠️ **Une erreur est survenue :** '},
    'gen_summary_spin':   {'English': 'Generating summary...', 'French': 'Génération du résumé...'},
    # Theme toggle
    'dark':               {'English': 'Dark',    'French': 'Sombre'},
    'light':              {'English': 'Light',   'French': 'Clair'},
    # Slider display values
    'sl_brief':           {'English': 'Brief',     'French': 'Bref'},
    'sl_standard':        {'English': 'Standard',  'French': 'Standard'},
    'sl_detailed':        {'English': 'Detailed',  'French': 'Détaillé'},
    # Topic buttons
    'topic_misconduct':   {'English': 'Misconduct types',  'French': 'Types de fautes'},
    'topic_appeal':       {'English': 'Appeal process',    'French': 'Procédure d\'appel'},
    'topic_data':         {'English': 'Data protection',   'French': 'Protection des données'},
    'topic_deferrals':    {'English': 'Deferrals',         'French': 'Reports'},
    # Source cards
    'file_label':         {'English': 'FILE',    'French': 'FICHIER'},
    'relevance_label':    {'English': 'RELEVANCE','French': 'PERTINENCE'},
    'confidence':         {'English': 'Confidence:','French': 'Confiance :'},
    'type_label':         {'English': 'Type:',   'French': 'Type :'},
    # Compare
    'comparing_label':    {'English': 'Comparing:','French': 'Comparaison :'},
    # Summarise user message
    'summarise_label':    {'English': '📋 Summarise:','French': '📋 Résumer :'},
    'compare_label':      {'English': '🔀 Compare:','French': '🔀 Comparer :'},
    # TTS
    'tts_speak':          {'English': 'Read aloud','French': 'Lire à voix haute'},
    'tts_stop':           {'English': 'Stop reading','French': 'Arrêter la lecture'},
    'copy':               {'English': 'Copy answer','French': 'Copier la réponse'},
    'copied':             {'English': 'Copied to clipboard! 📋','French': 'Copié dans le presse-papiers ! 📋'},
    # Confidence warnings
    'conf_low':           {'English': '⚠️ **Low confidence** — This answer may not be fully reliable. Please verify with the relevant department or email careandconcern@mdx.ac.mu',
                           'French':  '⚠️ **Confiance faible** — Cette réponse pourrait ne pas être entièrement fiable. Veuillez vérifier auprès du département concerné ou écrire à careandconcern@mdx.ac.mu'},
    'conf_med':           {'English': '⚡ **Moderate confidence** — The answer is likely correct but may be incomplete. Check the source references below for full details.',
                           'French':  '⚡ **Confiance modérée** — La réponse est probablement correcte mais peut être incomplète. Consultez les références ci-dessous.'},
    # Coverage gaps
    'coverage_gap':       {'English': '📋 **Possible coverage gaps** — The indexed documents may not fully cover:',
                           'French':  '📋 **Lacunes possibles** — Les documents indexés pourraient ne pas couvrir entièrement :'},
}

def t(key, **kwargs):
    """Get translated string for current language. Supports {var} formatting."""
    lang = st.session_state.get('language', 'English')
    text = TR.get(key, {}).get(lang, TR.get(key, {}).get('English', key))
    if kwargs:
        text = text.format(**kwargs)
    return text

#injecting theme styles — BOTH light and dark use !important to override Streamlit's base theme.
# This is necessary because Streamlit may apply its own dark mode (via config or system preference)
# which would override our default :root variables in style.css.
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    :root {
        --bg-base:       #0e1117 !important;
        --bg-panel:      #13181f !important;
        --bg-card:       #161c26 !important;
        --bg-hover:      #1c2535 !important;
        --navy:          #1a3a6e !important;
        --blue:          #2563eb !important;
        --blue-light:    #3b82f6 !important;
        --blue-glow:     #60a5fa !important;
        --teal:          #0ea5e9 !important;
        --gold:          #d4a843 !important;
        --gold-dim:      #a07830 !important;
        --green:         #22c55e !important;
        --green-dim:     #16a34a !important;
        --amber:         #f59e0b !important;
        --red:           #ef4444 !important;
        --border:        rgba(59,130,246,0.15) !important;
        --border-mid:    rgba(59,130,246,0.3)  !important;
        --border-bright: rgba(59,130,246,0.55) !important;
        --text-main:     #e8edf5 !important;
        --text-sub:      #a8b8cc !important;
        --text-dim:      #6b83a8 !important;
        --header-bg1:    #0d1b35 !important;
        --header-bg2:    #112244 !important;
        --header-text:   #e8f0ff !important;
        --header-sub:    rgba(255,255,255,0.45) !important;
        --pill-color:    #60a5fa !important;
        --pill-bg:       rgba(37,99,235,0.12) !important;
        --pill-border:   rgba(59,130,246,0.3)  !important;
        --badge-high:    #4ade80 !important;
        --badge-med:     #fbbf24 !important;
        --badge-low:     #f87171 !important;
        --dl-btn-color:  #4ade80 !important;
        --dot-color:     rgba(59,130,246,0.08) !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light mode — also injected with !important to override Streamlit's own dark theme
    st.markdown("""
    <style>
    :root {
        --bg-base:       #f0f4ff !important;
        --bg-panel:      #e4ebf8 !important;
        --bg-card:       #ffffff !important;
        --bg-hover:      #dce6f7 !important;
        --navy:          #1a3a6e !important;
        --blue:          #2563eb !important;
        --blue-light:    #3b82f6 !important;
        --blue-glow:     #1d4ed8 !important;
        --teal:          #0284c7 !important;
        --gold:          #b45309 !important;
        --gold-dim:      #92400e !important;
        --green:         #16a34a !important;
        --green-dim:     #15803d !important;
        --amber:         #d97706 !important;
        --red:           #dc2626 !important;
        --border:        rgba(37,99,235,0.18) !important;
        --border-mid:    rgba(37,99,235,0.35) !important;
        --border-bright: rgba(37,99,235,0.6)  !important;
        --text-main:     #111827 !important;
        --text-sub:      #1f2937 !important;
        --text-dim:      #4b5563 !important;
        --header-bg1:    #1e3a8a !important;
        --header-bg2:    #1d4ed8 !important;
        --header-text:   #ffffff !important;
        --header-sub:    rgba(255,255,255,0.75) !important;
        --pill-color:    #bfdbfe !important;
        --pill-bg:       rgba(255,255,255,0.18) !important;
        --pill-border:   rgba(255,255,255,0.35) !important;
        --badge-high:    #15803d !important;
        --badge-med:     #b45309 !important;
        --badge-low:     #b91c1c !important;
        --dl-btn-color:  #15803d !important;
        --dot-color:     rgba(37,99,235,0.06) !important;
    }
    </style>
    """, unsafe_allow_html=True)

FEEDBACK_LOG = './feedback_log.json'

#feedback logging functions
def load_feedback_log():
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, 'r') as f:
            return json.load(f)
    return []

def save_feedback(msg_index, rating, query, response_preview):
    log = load_feedback_log()
    log.append({
        'timestamp':        datetime.now().isoformat(),
        'message_index':    msg_index,
        'rating':           rating,
        'query':            query,
        'response_preview': response_preview[:200],
    })
    with open(FEEDBACK_LOG, 'w') as f:
        json.dump(log, f, indent=2)

# chromadb collection access with caching to avoid reconnecting on every query.
#  The collection is stored in './chroma_store' and named 'mum_policy_docs'.
@st.cache_resource
def get_chroma_collection():
    chroma = chromadb.PersistentClient(path='./chroma_store')
    return chroma.get_or_create_collection('mum_policy_docs')

def get_indexed_info():
    try:
        col     = get_chroma_collection()
        count   = col.count()
        results = col.get(include=['metadatas']) if count > 0 else {'metadatas': []}
        sources = sorted(set(m.get('source', '') for m in results['metadatas'] if m.get('source')))
        titles  = {m.get('source', ''): m.get('doc_title', m.get('source', '')) for m in results['metadatas']}
        dates   = {}
        for m in results['metadatas']:
            src = m.get('source', '')
            if src and m.get('indexed_at'):
                dates[src] = m['indexed_at'][:10]
        return count, sources, titles, dates
    except Exception:
        return 0, [], {}, {}

def load_index_log():
    try:
        if os.path.exists('./index_log.json'):
            with open('./index_log.json', 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

# Policy search function that takes a user query, generates an embedding, 
# and retrieves the most relevant document chunks from ChromaDB. 
# It returns a list of hits with text, title, file, and confidence score.
def policy_search(query, top_k=5):
    try:
        from query import get_embedding, compute_confidence
        chroma  = chromadb.PersistentClient(path='./chroma_store')
        col     = chroma.get_or_create_collection('mum_policy_docs')
        count   = col.count()
        if count == 0:
            return []
        emb     = get_embedding(query)
        results = col.query(
            query_embeddings=[emb],
            n_results=min(top_k, count),
            include=['documents', 'metadatas', 'distances']
        )
        hits = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            hits.append({
                'text':  doc,
                'title': meta.get('doc_title', meta.get('source', 'Unknown')),
                'file':  meta.get('source', 'Unknown'),
                'score': compute_confidence(dist),
            })
        return hits
    except Exception as e:
        st.error(f'Search error: {e}')
        return []

# Typing animation generator — yields words with a small delay
# to create a ChatGPT-like streaming effect for new responses.
# Only used for live responses; chat history replays instantly.
def stream_text(text, delay=0.03):
    """Yield words one at a time with a small delay for typing effect."""
    for word in text.split(' '):
        yield word + ' '
        time.sleep(delay)

# Text-to-speech — uses the browser's Web Speech API via an injected HTML component.
# Language is mapped to the closest available TTS locale.
TTS_LANG_MAP = {'English': 'en-US', 'French': 'fr-FR'}

def render_tts(text, lang='English'):
    """Inject a mini audio player with play/pause/resume/stop using Web Speech API."""
    locale = TTS_LANG_MAP.get(lang, 'en-US')
    # Clean text for JS
    clean = text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', ' ').replace('\r', '')
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
    clean = re.sub(r'\[Source:[^\]]*\]', '', clean)
    clean = re.sub(r'#{1,3}\s*', '', clean)
    clean = re.sub(r'💡.*', '', clean)
    clean = re.sub(r'You may also want to ask.*', '', clean)
    clean = re.sub(r'Vous pourriez aussi.*', '', clean)
    clean = re.sub(r'Ou kapav osi.*', '', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    components.html(f"""
    <div id="tts-player" style="
        display:flex; align-items:center; justify-content:center; gap:12px;
        padding:8px 16px; border-radius:8px;
        background:rgba(37,99,235,0.08); border:1px solid rgba(59,130,246,0.25);
        font-family:'Inter',sans-serif; font-size:13px; color:#94a3b8;
        margin:4px 0;
    ">
        <button id="tts-play" onclick="ttsPlay()" style="all:unset;cursor:pointer;font-size:20px;" title="Play from start">▶️</button>
        <button id="tts-pause" onclick="ttsPause()" style="all:unset;cursor:pointer;font-size:20px;display:none;" title="Pause">⏸️</button>
        <button id="tts-resume" onclick="ttsResume()" style="all:unset;cursor:pointer;font-size:16px;display:none;padding:2px 10px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4);border-radius:6px;" title="Resume">▶ Resume</button>
        <button id="tts-stop" onclick="ttsStop()" style="all:unset;cursor:pointer;font-size:20px;" title="Stop">⏹️</button>
        <span id="tts-status" style="font-size:11px;letter-spacing:0.5px;text-transform:uppercase;">Ready</span>
    </div>
    <script>
        var synth = window.speechSynthesis;
        var utterance = null;
        var ttsText = '{clean}';
        var ttsLang = '{locale}';

        function ttsPlay() {{
            synth.cancel();
            utterance = new SpeechSynthesisUtterance(ttsText);
            utterance.lang = ttsLang;
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.onstart = function() {{
                document.getElementById('tts-play').style.display = 'none';
                document.getElementById('tts-pause').style.display = 'inline';
                document.getElementById('tts-resume').style.display = 'none';
                document.getElementById('tts-status').textContent = '🔊 Speaking...';
            }};
            utterance.onpause = function() {{
                document.getElementById('tts-pause').style.display = 'none';
                document.getElementById('tts-resume').style.display = 'inline';
                document.getElementById('tts-status').textContent = '⏸ Paused';
            }};
            utterance.onresume = function() {{
                document.getElementById('tts-pause').style.display = 'inline';
                document.getElementById('tts-resume').style.display = 'none';
                document.getElementById('tts-status').textContent = '🔊 Speaking...';
            }};
            utterance.onend = function() {{
                document.getElementById('tts-play').style.display = 'inline';
                document.getElementById('tts-pause').style.display = 'none';
                document.getElementById('tts-resume').style.display = 'none';
                document.getElementById('tts-status').textContent = '✔ Done';
            }};
            synth.speak(utterance);
        }}

        function ttsPause() {{
            if (synth.speaking && !synth.paused) {{
                synth.pause();
                // Force UI update in case onpause doesn't fire (Chrome bug)
                setTimeout(function() {{
                    if (synth.paused) {{
                        document.getElementById('tts-pause').style.display = 'none';
                        document.getElementById('tts-resume').style.display = 'inline';
                        document.getElementById('tts-status').textContent = '⏸ Paused';
                    }}
                }}, 100);
            }}
        }}

        function ttsResume() {{
            if (synth.paused) {{
                synth.resume();
                // Force UI update in case onresume doesn't fire
                setTimeout(function() {{
                    if (synth.speaking && !synth.paused) {{
                        document.getElementById('tts-pause').style.display = 'inline';
                        document.getElementById('tts-resume').style.display = 'none';
                        document.getElementById('tts-status').textContent = '🔊 Speaking...';
                    }}
                }}, 100);
            }}
        }}

        function ttsStop() {{
            synth.cancel();
            document.getElementById('tts-play').style.display = 'inline';
            document.getElementById('tts-pause').style.display = 'none';
            document.getElementById('tts-resume').style.display = 'none';
            document.getElementById('tts-status').textContent = 'Ready';
        }}

        // Auto-play on load
        ttsPlay();
    </script>
    """, height=50)

# Copy to clipboard — uses the browser's Clipboard API via injected JS.
def render_copy(text):
    """Inject a hidden JS component that copies text to the clipboard."""
    clean = text.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    components.html(f"""
    <script>
    (function() {{
        var text = `{clean}`;
        if (window.parent && window.parent.navigator && window.parent.navigator.clipboard) {{
            window.parent.navigator.clipboard.writeText(text).catch(function() {{}});
        }}
    }})();
    </script>
    """, height=0)

# follow-up question parser that looks for specific markers in the AI response to identify suggested follow-up questions.
def parse_followups(response):
    """Split AI response into (main_text, [follow_up_questions])."""
    markers = [
        '💡 You may also want to ask:',
        '💡 You may also want to ask',
        'You may also want to ask:',
    ]
    split_index = -1
    used_marker = ''
    for marker in markers:
        idx = response.find(marker)
        if idx != -1:
            split_index = idx
            used_marker = marker
            break

    if split_index == -1:
        return response, []

    main_text     = response[:split_index].strip()
    followup_text = response[split_index + len(used_marker):].strip()

    questions = []
    for line in followup_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*\d\.\)]+\s*', '', line).strip()
        line = line.strip('"\'')
        if len(line) > 10:
            questions.append(line)

    return main_text, questions[:3]

# badge generators for confidence score and query type, 
# which return HTML snippets with appropriate styling based on the score or type.
def confidence_badge(score):
    pct = score * 100
    lang = st.session_state.get('language', 'English')
    levels = {
        'high':   {'English': 'High',   'French': 'Élevé'},
        'medium': {'English': 'Medium', 'French': 'Moyen'},
        'low':    {'English': 'Low',    'French': 'Faible'},
    }
    if pct >= 65:
        lbl = levels['high'].get(lang, 'High')
        return f'<span class="badge badge-high">● {pct:.0f}% {lbl}</span>'
    elif pct >= 40:
        lbl = levels['medium'].get(lang, 'Medium')
        return f'<span class="badge badge-med">● {pct:.0f}% {lbl}</span>'
    else:
        lbl = levels['low'].get(lang, 'Low')
        return f'<span class="badge badge-low">● {pct:.0f}% {lbl}</span>'

def render_confidence_warning(score):
    """Show a contextual warning banner when confidence is below the high threshold."""
    pct = score * 100
    if pct < 40:
        st.warning(t('conf_low'))
    elif pct < 65:
        st.info(t('conf_med'))
    # High confidence (≥65%) — no warning needed

QTYPE_ICONS = {
    'summary': '📋', 'disciplinary': '⚖️', 'procedural': '📝',
    'comparison': '🔀', 'definition': '📖', 'general': '💬', 'error': '⚠️',
}
QTYPE_TR = {
    'summary':      {'English': 'Summary',      'French': 'Résumé'},
    'disciplinary': {'English': 'Disciplinary',  'French': 'Disciplinaire'},
    'procedural':   {'English': 'Procedural',    'French': 'Procédural'},
    'comparison':   {'English': 'Comparison',    'French': 'Comparaison'},
    'definition':   {'English': 'Definition',    'French': 'Définition'},
    'general':      {'English': 'General',       'French': 'Général'},
    'error':        {'English': 'Error',         'French': 'Erreur'},
}

def query_type_badge(qtype):
    lang = st.session_state.get('language', 'English')
    icon = QTYPE_ICONS.get(qtype, '💬')
    label = QTYPE_TR.get(qtype, {}).get(lang, QTYPE_TR.get(qtype, {}).get('English', 'General'))
    return f'<span class="badge badge-type">{icon} {label}</span>'

def render_source_cards(sources):
    for s in sources:
        st.markdown(f"""
        <div class="src-card">
            <div class="src-title">📄 {s['title']}</div>
            <div class="src-meta">{t('file_label')}: {s['file']}  ·  {t('relevance_label')}: {s['score']*100:.0f}%</div>
            <div class="src-preview">{s['preview']}</div>
        </div>
        """, unsafe_allow_html=True)

from contextlib import contextmanager

@contextmanager
def custom_spinner(text=''):
    """Custom HTML spinner that works in both light and dark themes."""
    ph = st.empty()
    ph.markdown(f"""
    <div class="mum-spinner">
        <div class="mum-spinner-ring"></div>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)
    try:
        yield
    finally:
        ph.empty()

def export_chat_history():
    if not st.session_state.messages:
        return None
    lines = [f'{t("export_chat")} — MUM Policy Assistant\n{"="*50}\n',
             f'Exported: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n']
    for msg in st.session_state.messages:
        role = 'Student' if msg['role'] == 'user' else 'Assistant'  # keep English in exports for consistency
        lines.append(f'[{role}]\n{msg["content"]}\n')
        if msg.get('score', 0) > 0:
            lines.append(f'Confidence: {msg["score"]*100:.0f}%  |  Response time: {msg.get("elapsed",0)}s\n')
        lines.append('\n')
    return ''.join(lines)

# sidebar content with branding, theme toggle, indexed document info, 
# policy search, summariser, and session statistics.
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <img src="{ICON_DATA_URI}" class="brand-icon-img"/>
        <span class="brand-name">MUM Policy Assistant</span>
        <span class="brand-sub">CST3990 · Middlesex University Mauritius</span>
    </div>
    """, unsafe_allow_html=True)

    # theme toggle — single pill switch
    is_dark = st.session_state.dark_mode
    st.markdown(f"""
    <div class="theme-toggle-row">
        <span class="theme-icon">{'🌙' if is_dark else '☀️'}</span>
        <span class="theme-toggle-label">{t('dark') if is_dark else t('light')}</span>
    </div>
    """, unsafe_allow_html=True)
    toggled = st.toggle('Dark mode', value=st.session_state.dark_mode, key='theme_toggle', label_visibility='collapsed')
    if toggled != st.session_state.dark_mode:
        st.session_state.dark_mode = toggled
        st.rerun()

    chunk_count, sources, titles, dates = get_indexed_info()
    index_log = load_index_log()

    # an option slider for response length that updates session state 
    # and provides descriptions for each level of detail.
    st.markdown(f'<span class="sec-head">{t("response_length")}</span>', unsafe_allow_html=True)
    slider_display = {'Brief': t('sl_brief'), 'Standard': t('sl_standard'), 'Detailed': t('sl_detailed')}
    length_choice = st.select_slider(
        label=t('detail_level'),
        options=['Brief', 'Standard', 'Detailed'],
        format_func=lambda x: slider_display[x],
        value=st.session_state.answer_length,
        key='length_slider',
        label_visibility='collapsed'
    )
    st.session_state.answer_length = length_choice

    # Inject dynamic track color based on current slider value
    # This colors the entire slider track line to match the selection.
    slider_colors = {
        'Brief':    ('#ef4444', 'rgba(239,68,68,0.25)'),
        'Standard': ('#f59e0b', 'rgba(245,158,11,0.25)'),
        'Detailed': ('#22c55e', 'rgba(34,197,94,0.25)'),
    }
    track_color, track_glow = slider_colors[length_choice]
    st.markdown(f"""
    <style>
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
        background: {track_color} !important;
        background-color: {track_color} !important;
    }}
    div[data-testid="stSlider"] [role="slider"] {{
        background: {track_color} !important;
        border-color: {track_color} !important;
        box-shadow: 0 0 0 3px {track_glow} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.caption(t(length_choice.lower()))

    # Language selector — respond in English or French
    st.markdown(f'<span class="sec-head">{t("language")}</span>', unsafe_allow_html=True)
    lang_options = {
        'English':  '🇬🇧 English',
        'French':   '🇫🇷 Français',
    }
    selected_lang = st.radio(
        label='Response language:',
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language),
        key='lang_radio',
        horizontal=True,
        label_visibility='collapsed'
    )
    st.session_state.language = selected_lang

    # Policy quick search that allows users to enter a query 
    # and retrieves relevant document chunks with confidence scores, displayed in expanders.
    st.markdown(f'<span class="sec-head">{t("quick_search")}</span>', unsafe_allow_html=True)
    search_query = st.text_input(
        label=t('search_label'),
        placeholder=t('search_placeholder'),
        key='policy_search_input'
    )
    if search_query and len(search_query) > 2:
        with custom_spinner(t('searching')):
            hits = policy_search(search_query, top_k=4)
        if hits:
            for hit in hits:
                with st.expander(f"📄 {hit['title']} — {hit['score']*100:.0f}% match", expanded=False):
                    st.caption(f"**{t('file_label')}:** {hit['file']}")
                    st.markdown(f"_{hit['text'][:300]}{'...' if len(hit['text']) > 300 else ''}_")
        else:
            st.info(t('no_match'))

    # ══════════════════════════════════════════════════════
    # DOCUMENT INDEX + SUMMARISER — merged into one section
    # ══════════════════════════════════════════════════════
    st.markdown(f'<span class="sec-head">{t("doc_index")}</span>', unsafe_allow_html=True)

    if chunk_count == 0:
        st.warning(t('no_docs'))
    else:
        # Stats + doc list + summariser in one compact block
        st.markdown(f"""
        <div class="sidebar-stats-strip">
            <span class="strip-stat"><strong>{chunk_count}</strong> {t('chunks')}</span>
            <span class="strip-dot">·</span>
            <span class="strip-stat"><strong>{len(sources)}</strong> {t('documents')}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📚 {t('view_all_docs', n=len(sources))}", expanded=False):
            doc_rows = []
            for src in sources:
                title    = titles.get(src, src)
                chunks_n = index_log.get(src, {}).get('chunks', '?')
                date_str = dates.get(src, '')
                date_tag = f' · {date_str}' if date_str else ''
                doc_rows.append(
                    f'<div class="doc-row">'
                    f'<span class="doc-row-title">📄 {title}</span>'
                    f'<span class="doc-row-meta">{chunks_n} {t("chunks").lower()}{date_tag}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="doc-list">{"".join(doc_rows)}</div>',
                unsafe_allow_html=True
            )

        # Summariser — inline under doc index
        if sources:
            display_titles = [titles.get(s, s) for s in sources]
            sel_idx        = st.selectbox(t('choose_doc'), range(len(sources)),
                                          format_func=lambda i: display_titles[i], key='sum_select',
                                          label_visibility='collapsed')
            selected_doc   = sources[sel_idx]
            selected_title = display_titles[sel_idx]

            if st.button(t('gen_summary'), use_container_width=True, type='primary'):
                with custom_spinner(t('gen_summary_spin')):
                    try:
                        t0 = time.time()
                        summary, srcs, score, qtype = answer(
                            f'Please provide a detailed structured summary of all key points in {selected_doc}',
                            history=None,
                            length=st.session_state.answer_length,
                            language=st.session_state.language
                        )
                        elapsed = round(time.time() - t0, 2)
                        st.session_state.messages.append({
                            'role': 'user', 'content': f'{t("summarise_label")} **{selected_title}**',
                            'sources': [], 'score': 0, 'elapsed': 0, 'qtype': 'summary'
                        })
                        st.session_state.messages.append({
                            'role': 'assistant', 'content': summary,
                            'sources': srcs, 'score': score, 'elapsed': elapsed, 'qtype': qtype
                        })
                        st.session_state.query_count   += 1
                        st.session_state.avg_score_all.append(score)
                        st.session_state.total_time    += elapsed
                        st.rerun()
                    except Exception as e:
                        err = str(e)
                        if '429' in err:
                            st.error(t('api_quota'))
                        else:
                            st.error(f'Error: {err}')

    # Export + Clear + Refresh — compact icon row
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.session_state.messages:
            export_text = export_chat_history()
            if export_text:
                st.download_button(
                    label='💾',
                    data=export_text,
                    file_name=f'mum_chat_{datetime.now().strftime("%Y%m%d_%H%M")}.txt',
                    mime='text/plain',
                    use_container_width=True,
                    help=t('export_chat')
                )
    with col_b:
        if st.button('🗑️', key='btn_clear', use_container_width=True, help=t('clear')):
            st.session_state.messages      = []
            st.session_state.query_count   = 0
            st.session_state.avg_score_all = []
            st.session_state.total_time    = 0.0
            st.rerun()
    with col_c:
        if st.button('🔄', key='btn_refresh', use_container_width=True, help=t('refresh')):
            st.cache_resource.clear()
            st.rerun()

    # ══════════════════════════════════════════════════════
    # QUICK QUESTIONS — inside a collapsible expander
    # ══════════════════════════════════════════════════════
    with st.expander(f"💬 {t('quick_questions')}", expanded=False):
        quick_qs = [t('qq1'), t('qq2'), t('qq3'), t('qq4'), t('qq5'), t('qq6'), t('qq7')]
        for qi, q in enumerate(quick_qs):
            if st.button(q, key=f'qq_{qi}', use_container_width=True):
                st.session_state['pending_question'] = q
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT — two modes: landing (empty) vs chat (active)
# ═══════════════════════════════════════════════════════════════

has_messages = len(st.session_state.messages) > 0
has_pending = 'pending_question' in st.session_state

# Toggle page scroll: locked on landing, unlocked when chatting
# Uses JS to find every scrollable element inside .main and disable scrolling.
if not has_messages and not has_pending:
    components.html("""
    <script>
    (function() {
        function lockScroll() {
            // Find all scrollable elements inside the main area
            var main = window.parent.document.querySelector('section.main');
            if (!main) return;
            var els = main.querySelectorAll('*');
            for (var i = 0; i < els.length; i++) {
                var el = els[i];
                if (el.scrollHeight > el.clientHeight) {
                    el.style.overflow = 'hidden';
                    el.dataset.scrollLocked = 'true';
                }
            }
            // Also lock the main section itself
            main.style.overflow = 'hidden';
            // And the stAppViewContainer
            var appView = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) appView.style.overflow = 'hidden';
        }
        // Run after a short delay to let Streamlit finish rendering
        setTimeout(lockScroll, 200);
        setTimeout(lockScroll, 600);
    })();
    </script>
    """, height=0)

if not has_messages and not has_pending:
    # ── LANDING MODE ──────────────────────────────────────────
    # Single centered hero: icon + greeting + description + topic pills
    # Greeting adapts to time of day AND selected language.
    hour = datetime.now().hour
    lang = st.session_state.language

    greetings = {
        'English':  ('Good morning', 'Good afternoon', 'Good evening'),
        'French':   ('Bonjour', 'Bon après-midi', 'Bonsoir'),
    }
    descs = {
        'English': 'Ask me anything about university policies — misconduct rules, appeal processes, data protection, deferrals, and more. Every answer is grounded in official MUM documents with source citations.',
        'French':  'Posez-moi vos questions sur les politiques universitaires — règles de conduite, procédures d\'appel, protection des données, reports et plus encore. Chaque réponse est fondée sur les documents officiels de MUM avec citations.',
    }
    topic_labels = {
        'English': 'Try asking about:',
        'French':  'Essayez de demander :',
    }
    morning, afternoon, evening = greetings.get(lang, greetings['English'])
    if hour < 12:
        greeting, greeting_icon = morning, '🌅'
    elif hour < 17:
        greeting, greeting_icon = afternoon, '☀️'
    else:
        greeting, greeting_icon = evening, '🌙'

    desc_text = descs.get(lang, descs['English'])

    st.markdown(f"""
    <div class="landing-hero">
        <img src="{ICON_DATA_URI}" class="landing-icon"/>
        <div class="landing-greeting">{greeting_icon} {greeting}!</div>
        <div class="landing-title">MUM Policy & Academic Assistant</div>
        <div class="landing-desc">{desc_text}</div>
        <div class="landing-pills">
            <span class="pill">RAG</span>
            <span class="pill">Gemini 2.5</span>
            <span class="pill">ChromaDB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Topic quick-start buttons — clicking one sends it as a query
    st.markdown(f'<div class="landing-topics-label">{topic_labels.get(lang, topic_labels["English"])}</div>', unsafe_allow_html=True)
    topic_cols = st.columns(4)
    topic_questions = [
        ('📋', t('topic_misconduct'), t('qq1')),
        ('⚖️', t('topic_appeal'), t('qq3')),
        ('🔒', t('topic_data'), t('qq6')),
        ('📝', t('topic_deferrals'), t('qq7')),
    ]
    for ti, (col, (icon, label, question)) in enumerate(zip(topic_cols, topic_questions)):
        with col:
            if st.button(f'{icon} {label}', key=f'topic_{ti}', use_container_width=True):
                st.session_state['pending_question'] = question
                st.rerun()

# ── Tabs (always visible) ─────────────────────────────────────
if has_messages:
    # Compact header bar when in chat mode
    st.markdown(f"""
    <div class="chat-topbar">
        <img src="{ICON_DATA_URI}" class="topbar-icon"/>
        <span class="topbar-title">MUM Policy Assistant</span>
        <span class="topbar-pills">
            <span class="pill-sm">RAG</span>
            <span class="pill-sm">Gemini 2.5</span>
            <span class="pill-sm">ChromaDB</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

tab_chat, tab_compare = st.tabs([t('tab_chat'), t('tab_compare')])

# Tab 1 — Chat interface that displays the conversation history with user and assistant messages, 
# confidence badges, source references, and follow-up suggestions. It also includes a chat input for new questions.
with tab_chat:
    for i, msg in enumerate(st.session_state.messages):
        avatar = ASSISTANT_AVATAR if msg['role'] == 'assistant' else None
        with st.chat_message(msg['role'], avatar=avatar):

            if msg['role'] == 'assistant':
                main_text = msg['content']
                # Use stored follow-ups if available; fall back to parsing for old messages
                followups = msg.get('followups', [])
                if not followups:
                    main_text, followups = parse_followups(msg['content'])
                st.markdown(main_text)
                # Show confidence warning for low/medium confidence answers
                if msg.get('score', 0) > 0:
                    render_confidence_warning(msg['score'])
                # Show coverage gaps if they were detected
                gaps = msg.get('coverage_gaps', [])
                if gaps:
                    gap_list = '\n'.join(f'- {g}' for g in gaps)
                    st.warning(f'{t("coverage_gap")}\n{gap_list}')
            else:
                st.markdown(msg['content'])

            if msg['role'] == 'assistant' and msg.get('score', 0) > 0:
                meta_cols = st.columns([2, 2, 1, 1, 1, 1, 1])
                with meta_cols[0]:
                    st.markdown(f'**{t("confidence")}** {confidence_badge(msg["score"])}', unsafe_allow_html=True)
                with meta_cols[1]:
                    st.markdown(f'**{t("type_label")}** {query_type_badge(msg.get("qtype","general"))}', unsafe_allow_html=True)
                with meta_cols[2]:
                    if msg.get('elapsed', 0) > 0:
                        st.caption(f'⏱ {msg["elapsed"]}s')
                with meta_cols[3]:
                    if st.button('👍', key=f'up_{i}', help=t('helpful')):
                        prev_user = next(
                            (m['content'] for m in reversed(st.session_state.messages[:i])
                             if m['role'] == 'user'), ''
                        )
                        save_feedback(i, 'up', prev_user, msg['content'])
                        st.toast(t('thanks_up'))
                with meta_cols[4]:
                    if st.button('👎', key=f'down_{i}', help=t('needs_improvement')):
                        prev_user = next(
                            (m['content'] for m in reversed(st.session_state.messages[:i])
                             if m['role'] == 'user'), ''
                        )
                        save_feedback(i, 'down', prev_user, msg['content'])
                        st.toast(t('thanks_down'))
                with meta_cols[5]:
                    if st.button('🔊', key=f'tts_{i}', help=t('tts_speak')):
                        st.session_state.tts_text = main_text
                        st.session_state.tts_counter += 1
                with meta_cols[6]:
                    if st.button('📋', key=f'copy_{i}', help=t('copy')):
                        st.session_state.copy_text = main_text
                        st.toast(t('copied'))

                if msg.get('sources'):
                    with st.expander(t('view_sources', n=len(msg['sources'])), expanded=False):
                        render_source_cards(msg['sources'])

                if followups:
                    st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
                    st.caption(t('followup'))
                    for q in followups:
                        if st.button(f'→ {q}', key=f'fq_{i}_{q[:30]}', use_container_width=True):
                            st.session_state['pending_question'] = q
                            st.rerun()

    # chat input area that captures user questions, appends them to the conversation history,
    # and triggers the answer generation process with appropriate error handling and feedback logging.
    pending    = st.session_state.pop('pending_question', None)
    user_input = st.chat_input(t('chat_placeholder')) or pending

    if user_input:
        st.session_state.messages.append({
            'role': 'user', 'content': user_input,
            'sources': [], 'score': 0, 'elapsed': 0, 'qtype': ''
        })

        with st.chat_message('user'):
            st.markdown(user_input)

        history = [
            ('human' if m['role'] == 'user' else 'assistant', m['content'])
            for m in st.session_state.messages[:-1]
            if m['role'] in ('user', 'assistant')
        ]

        with st.chat_message('assistant', avatar=ASSISTANT_AVATAR):
            with custom_spinner(t('searching')):
                t0 = time.time()
                try:
                    response, srcs, score, qtype = answer(
                        user_input,
                        history=history,
                        length=st.session_state.answer_length,
                        language=st.session_state.language
                    )
                    elapsed   = round(time.time() - t0, 2)
                    main_text = response

                    # Typing animation — stream words one at a time
                    st.write_stream(stream_text(main_text))

                    # Show confidence warning for low/medium confidence answers
                    if score > 0:
                        render_confidence_warning(score)

                    # Generate context-aware follow-up suggestions via a dedicated Gemini call
                    followups = generate_followups(
                        user_input, response,
                        history=history,
                        language=st.session_state.language
                    )

                    # Detect if the retrieved documents had coverage gaps
                    coverage_gaps = detect_coverage_gaps(
                        user_input, response, srcs,
                        language=st.session_state.language
                    )
                    if coverage_gaps:
                        gap_list = '\n'.join(f'- {g}' for g in coverage_gaps)
                        st.warning(f'{t("coverage_gap")}\n{gap_list}')

                    if score > 0:
                        meta_cols = st.columns([2, 2, 1, 1, 1, 1, 1])
                        with meta_cols[0]:
                            st.markdown(f'**{t("confidence")}** {confidence_badge(score)}', unsafe_allow_html=True)
                        with meta_cols[1]:
                            st.markdown(f'**{t("type_label")}** {query_type_badge(qtype)}', unsafe_allow_html=True)
                        with meta_cols[2]:
                            st.caption(f'⏱ {elapsed}s')
                        with meta_cols[3]:
                            st.caption('👍')
                        with meta_cols[4]:
                            st.caption('👎')
                        with meta_cols[5]:
                            st.caption('🔊')
                        with meta_cols[6]:
                            st.caption('📋')

                    if srcs:
                        with st.expander(t('view_sources', n=len(srcs)), expanded=False):
                            render_source_cards(srcs)

                    if followups:
                        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
                        st.caption(t('followup'))
                        for q in followups:
                            if st.button(f'→ {q}', key=f'nfq_{q[:30]}', use_container_width=True):
                                st.session_state['pending_question'] = q
                                st.rerun()

                    st.session_state.query_count   += 1
                    st.session_state.avg_score_all.append(score)
                    st.session_state.total_time    += elapsed

                except Exception as e:
                    elapsed   = round(time.time() - t0, 2)
                    err       = str(e)
                    srcs      = []
                    score     = 0.0
                    qtype     = 'error'
                    followups = []
                    coverage_gaps = []
                    if '429' in err:
                        response = t('api_quota')
                    elif '404' in err:
                        response = t('model_404')
                    elif 'GOOGLE_API_KEY' in err:
                        response = t('key_missing')
                    else:
                        response = t('gen_error') + err
                    st.error(response)

        st.session_state.messages.append({
            'role': 'assistant', 'content': response,
            'sources': srcs, 'score': score, 'elapsed': elapsed, 'qtype': qtype,
            'followups': followups, 'coverage_gaps': coverage_gaps
        })
        st.rerun()

# Tab 2 — Document comparison interface that allows users to select two indexed documents 
# and generate a structured side-by-side comparison,
with tab_compare:
    st.markdown(t('compare_title'))
    st.markdown(t('compare_desc'))

    if len(sources) < 2:
        st.warning(t('need_2_docs'))
    else:
        display_titles = [titles.get(s, s) for s in sources]

        col_a, col_b = st.columns(2)
        with col_a:
            idx_a = st.selectbox(
                t('first_doc'),
                range(len(sources)),
                format_func=lambda i: display_titles[i],
                key='compare_a'
            )
        with col_b:
            default_b = 1 if idx_a == 0 else 0
            idx_b = st.selectbox(
                t('second_doc'),
                range(len(sources)),
                format_func=lambda i: display_titles[i],
                index=default_b,
                key='compare_b'
            )

        doc_a   = sources[idx_a]
        doc_b   = sources[idx_b]
        title_a = display_titles[idx_a]
        title_b = display_titles[idx_b]

        if idx_a == idx_b:
            st.warning(t('same_doc_warn'))
        else:
            st.markdown(f'**{t("comparing_label")}** {title_a} &nbsp;↔&nbsp; {title_b}')

            if st.button(t('gen_comparison'), use_container_width=True, type='primary'):
                with custom_spinner(t('comparing', a=title_a, b=title_b)):
                    try:
                        t0 = time.time()
                        response, srcs, score, qtype = compare_documents(
                            doc_a, doc_b, title_a, title_b,
                            length=st.session_state.answer_length,
                            language=st.session_state.language
                        )
                        elapsed = round(time.time() - t0, 2)

                        st.session_state.messages.append({
                            'role':    'user',
                            'content': f'{t("compare_label")} **{title_a}** vs **{title_b}**',
                            'sources': [], 'score': 0, 'elapsed': 0, 'qtype': 'comparison'
                        })

                        # Generate context-aware follow-ups for the comparison
                        compare_query = f'Compare {title_a} vs {title_b}'
                        followups = generate_followups(
                            compare_query, response,
                            language=st.session_state.language
                        )

                        st.session_state.messages.append({
                            'role':    'assistant',
                            'content': response,
                            'sources': srcs,
                            'score':   score,
                            'elapsed': elapsed,
                            'qtype':   qtype,
                            'followups': followups
                        })
                        st.session_state.query_count   += 1
                        st.session_state.avg_score_all.append(score)
                        st.session_state.total_time    += elapsed

                        st.markdown(response)

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(
                                f'**{t("type_label")}** {query_type_badge(qtype)}  &nbsp; ⏱ {elapsed}s',
                                unsafe_allow_html=True
                            )

                        if srcs:
                            with st.expander(t('view_sources', n=len(srcs)), expanded=False):
                                render_source_cards(srcs)

                        if followups:
                            st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
                            st.caption(t('followup'))
                            for q in followups:
                                if st.button(f'→ {q}', key=f'cfq_{q[:30]}', use_container_width=True):
                                    st.session_state['pending_question'] = q
                                    st.rerun()

                        st.success(t('compare_done'))

                    except Exception as e:
                        err = str(e)
                        if '429' in err:
                            st.error(t('api_quota'))
                        else:
                            st.error(f'Error: {err}')

# ── Text-to-Speech renderer ──────────────────────────
# If a TTS button was clicked, render the player.
if st.session_state.tts_text:
    render_tts(st.session_state.tts_text, st.session_state.language)
    st.session_state.tts_text = None

# ── Copy to clipboard renderer ───────────────────────
# If a copy button was clicked, inject JS to copy.
if st.session_state.copy_text:
    render_copy(st.session_state.copy_text)
    st.session_state.copy_text = None

# Footer
st.markdown(f"""
<div class="mum-footer">
    {t('footer')}
</div>
""", unsafe_allow_html=True)