import streamlit as st
import pandas as pd
import pickle
import re
import hashlib
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx2txt


# ============================================================
# SMARTHIRE — ENTERPRISE RECRUITMENT & ATS PLATFORM
# ============================================================

st.set_page_config(
    page_title="SmartHire — Intelligent Recruitment & ATS Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PREMIUM STYLING & UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #071016;
        --panel: #0d171e;
        --panel2: #101d25;
        --border: rgba(148,163,184,.14);
        --text: #f4f7fb;
        --muted: #91a0ad;
        --accent: #2dd4bf;
        --accent2: #38bdf8;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 5%, rgba(45,212,191,.08), transparent 28%),
            radial-gradient(circle at 35% 0%, rgba(56,189,248,.06), transparent 25%),
            #071016;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(7,16,22,.86);
    }

    [data-testid="stSidebar"] {
        background: #091117;
        border-right: 1px solid var(--border);
    }

    .brand {
        padding: 8px 6px 24px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: linear-gradient(135deg, #2dd4bf, #38bdf8);
        box-shadow: 0 8px 28px rgba(45,212,191,.18);
    }

    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -.5px;
    }

    .brand-sub {
        color: #71808c;
        font-size: .73rem;
        letter-spacing: 1.5px;
        margin-top: 5px;
        text-transform: uppercase;
    }

    .status-pill {
        margin-top: 22px;
        border: 1px solid rgba(45,212,191,.22);
        background: rgba(45,212,191,.07);
        color: #6ee7d4;
        border-radius: 999px;
        padding: 9px 12px;
        font-size: .78rem;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #2dd4bf;
        border-radius: 50%;
        box-shadow: 0 0 12px #2dd4bf;
    }

    [data-testid="stSidebar"] label {
        border-radius: 10px;
        padding: 9px 10px !important;
        color: #aeb9c3 !important;
        transition: .2s ease;
    }

    [data-testid="stSidebar"] label:hover {
        background: rgba(255,255,255,.045);
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(56,189,248,.16);
        border-radius: 28px;
        padding: 42px 46px;
        min-height: 285px;
        background: linear-gradient(135deg, rgba(13,42,57,.96), rgba(8,31,33,.96));
        box-shadow: 0 25px 80px rgba(0,0,0,.22);
    }

    .kpi {
        background: linear-gradient(145deg, #0f1a22, #0b141b);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        min-height: 122px;
    }

    .kpi-label {
        color: #84939f;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        font-weight: 700;
    }

    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #f4f7fb;
        font-size: 2rem;
        margin-top: 10px;
        letter-spacing: -1px;
    }

    .kpi-note {
        color: #5eead4;
        font-size: .72rem;
        margin-top: 3px;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f4f7fb;
        margin: 26px 0 13px;
    }

    .page-header {
        margin-bottom: 24px;
    }

    .page-kicker {
        color: #5eead4;
        font-size: .73rem;
        letter-spacing: 1.5px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -1.3px;
        margin-top: 7px;
    }

    .page-description {
        color: #84939e;
        font-size: .92rem;
        margin-top: 6px;
    }

    .analysis-box {
        border: 1px solid var(--border);
        background: #0c151c;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .analysis-label {
        color: #74838e;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: .68rem;
        font-weight: 800;
    }

    .analysis-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e9eef2;
        margin-top: 7px;
    }

    .tag {
        display: inline-block;
        padding: 7px 10px;
        margin: 4px 4px 0 0;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.15);
        background: rgba(255,255,255,.035);
        color: #cbd5df;
        font-size: .74rem;
    }

    .tag.primary {
        border-color: rgba(45,212,191,.23);
        background: rgba(45,212,191,.08);
        color: #78eadb;
    }

    .tag.missing {
        border-color: rgba(248,113,113,.23);
        background: rgba(248,113,113,.08);
        color: #fca5a5;
    }

    .footer {
        text-align: center;
        color: #566570;
        font-size: .72rem;
        margin-top: 55px;
        padding-top: 22px;
        border-top: 1px solid rgba(148,163,184,.08);
    }

    textarea, input {
        color: #e8eef2 !important;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(45,212,191,.28);
        background: linear-gradient(135deg, #123d3a, #0f3541);
        color: #dffcf7;
        font-weight: 700;
        min-height: 44px;
        padding: 0 18px;
        transition: .2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(45,212,191,.55);
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(45,212,191,.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "pipeline_stages" not in st.session_state:
    st.session_state["pipeline_stages"] = {}

if "uploaded_history" not in st.session_state:
    st.session_state["uploaded_history"] = []


# ============================================================
# LOAD ML RESOURCES WITH DYNAMIC VECTOR GENERATION
# ============================================================

@st.cache_resource(show_spinner="Loading SmartHire enterprise engine...")
def load_resources():
    try:
        data_path = BASE_DIR / "data" / "cleaned_resumes.csv"
        df = pd.read_csv(data_path)
        df["cleaned_resume"] = df["cleaned_resume"].fillna("")

        with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # Dynamically compute vectors on startup to avoid large GitHub file limits
        resume_vectors = vectorizer.transform(df["cleaned_resume"])

        with open(BASE_DIR / "classifier_model.pkl", "rb") as f:
            classifier_model = pickle.load(f)

        with open(BASE_DIR / "classifier_vectorizer.pkl", "rb") as f:
            classifier_vectorizer = pickle.load(f)

        return df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer, True, ""
    except Exception as exc:
        return pd.DataFrame(), None, None, None, None, False, str(exc)


df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer, ENGINE_READY, ENGINE_ERROR = load_resources()


# ============================================================
# NLTK SETUP
# ============================================================

@st.cache_resource
def load_nlp_tools():
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        sw = set()

    try:
        nltk.data.find("corpora/wordnet")
        lemma = WordNetLemmatizer()
    except LookupError:
        lemma = None

    return sw, lemma


stop_words, lemmatizer = load_nlp_tools()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    if stop_words:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    else:
        words = [w for w in words if len(w) > 2]

    if lemmatizer:
        words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# ============================================================
# SKILL TAXONOMY
# ============================================================

SKILLS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript",
    "Artificial Intelligence", "Machine Learning", "Deep Learning",
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Classification", "Regression", "Clustering", "Feature Engineering",
    "Feature Extraction", "Model Evaluation", "Predictive Modeling",
    "Neural Networks", "CNN", "RNN", "LSTM", "Transformers",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
    "Matplotlib", "Seaborn", "Data Science", "Data Analysis",
    "Data Preprocessing", "NLP", "Text Classification", "Sentiment Analysis",
    "Text Mining", "Large Language Models", "Generative AI", "RAG",
    "Computer Vision", "Image Classification", "Object Detection",
    "OpenCV", "YOLO", "SQL", "MySQL", "PostgreSQL", "MongoDB",
    "Git", "GitHub", "Docker", "AWS", "Azure", "GCP",
    "React", "Node.js", "Express", "Flask", "Django",
    "FastAPI", "Streamlit", "MLOps"
]


def normalize_skill_text(text):
    text = str(text).lower()
    text = text.replace("scikit learn", "scikit-learn")
    text = text.replace("node js", "node.js")
    text = text.replace("large language model", "large language models")
    return re.sub(r"\s+", " ", text)


def skill_pattern(skill):
    escaped = re.escape(skill.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"