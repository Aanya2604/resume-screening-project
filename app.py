import streamlit as st
import pandas as pd
import pickle
import re
import time
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx2txt


# ============================================================
# TALENTAI — AI-POWERED RESUME SCREENING & JOB ANALYSIS
# ============================================================

st.set_page_config(
    page_title="TalentAI — Intelligent Recruitment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PREMIUM UI
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

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
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

    /* Sidebar radio */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 5px;
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

    /* Hide the actual radio circles */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none;
    }

    /* Hero */
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(56,189,248,.16);
        border-radius: 28px;
        padding: 42px 46px;
        min-height: 285px;
        background:
            linear-gradient(135deg, rgba(13,42,57,.96), rgba(8,31,33,.96));
        box-shadow: 0 25px 80px rgba(0,0,0,.22);
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        right: -100px;
        top: -170px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(45,212,191,.18), transparent 68%);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: 170px;
        bottom: -170px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56,189,248,.12), transparent 68%);
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 760px;
    }

    .eyebrow {
        display: inline-flex;
        border: 1px solid rgba(45,212,191,.24);
        background: rgba(45,212,191,.08);
        color: #75ead9;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.1rem, 4vw, 3.7rem);
        line-height: 1.02;
        letter-spacing: -2px;
        margin: 19px 0 14px 0;
        color: #f8fafc;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #f8fafc, #6ee7d4, #7dd3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #aebdc8;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 700px;
        margin: 0;
    }

    .hero-mini {
        margin-top: 24px;
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
    }

    .hero-mini span {
        padding: 8px 11px;
        border-radius: 9px;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.07);
        color: #c6d1d9;
        font-size: .78rem;
    }

    /* KPI cards */
    .kpi {
        background: linear-gradient(145deg, #0f1a22, #0b141b);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        min-height: 122px;
        box-shadow: 0 14px 40px rgba(0,0,0,.13);
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

    .section-sub {
        color: #7f8d98;
        margin-top: -7px;
        margin-bottom: 18px;
        font-size: .88rem;
    }

    .feature-card {
        border: 1px solid var(--border);
        background: linear-gradient(145deg, #0f181f, #0b1319);
        border-radius: 18px;
        padding: 23px;
        min-height: 185px;
    }

    .feature-number {
        color: #2dd4bf;
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .feature-title {
        color: #f4f7fb;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.04rem;
        margin-top: 10px;
    }

    .feature-text {
        color: #8998a3;
        font-size: .83rem;
        line-height: 1.65;
        margin-top: 10px;
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

    .domain-card {
        border: 1px solid rgba(45,212,191,.18);
        background: linear-gradient(135deg, rgba(45,212,191,.09), rgba(56,189,248,.05));
        border-radius: 18px;
        padding: 22px;
    }

    .domain-label {
        color: #6ee7d4;
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 800;
    }

    .domain-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .role-name {
        color: #9fadb8;
        margin-top: 5px;
        font-size: .9rem;
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

    .empty-state {
        border: 1px dashed rgba(148,163,184,.18);
        background: rgba(255,255,255,.018);
        border-radius: 18px;
        padding: 38px;
        text-align: center;
        color: #82909b;
    }

    .footer {
        text-align: center;
        color: #566570;
        font-size: .72rem;
        margin-top: 55px;
        padding-top: 22px;
        border-top: 1px solid rgba(148,163,184,.08);
    }

    /* Streamlit inputs */
    textarea, input {
        color: #e8eef2 !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stFileUploaderDropzone"] {
        background: #0d171e !important;
        border: 1px solid rgba(148,163,184,.16) !important;
        border-radius: 14px !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(45,212,191,.45) !important;
        box-shadow: 0 0 0 1px rgba(45,212,191,.2) !important;
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

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    .small-muted {
        color: #71808b;
        font-size: .76rem;
    }

    @media (max-width: 900px) {
        .hero {
            padding: 30px 25px;
        }
        .hero h1 {
            font-size: 2.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA + MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading TalentAI engine...")
def load_resources():
    data_path = BASE_DIR / "data" / "cleaned_resumes.csv"

    with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(BASE_DIR / "resume_vectors.pkl", "rb") as f:
        resume_vectors = pickle.load(f)

    with open(BASE_DIR / "classifier_model.pkl", "rb") as f:
        classifier_model = pickle.load(f)

    with open(BASE_DIR / "classifier_vectorizer.pkl", "rb") as f:
        classifier_vectorizer = pickle.load(f)

    df = pd.read_csv(data_path)
    df["cleaned_resume"] = df["cleaned_resume"].fillna("")

    return df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer


try:
    df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer = load_resources()
    ENGINE_READY = True
    ENGINE_ERROR = ""
except Exception as exc:
    ENGINE_READY = False
    ENGINE_ERROR = str(exc)
    df = pd.DataFrame()
    vectorizer = None
    resume_vectors = None
    classifier_model = None
    classifier_vectorizer = None


# ============================================================
# NLTK — SAFE, NON-BLOCKING SETUP
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
# TECHNICAL SKILL ENGINE
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


def normalize_for_skill_matching(text):
    text = str(text).lower()
    text = text.replace("scikit learn", "scikit-learn")
    text = text.replace("scikit_learn", "scikit-learn")
    text = text.replace("node js", "node.js")
    text = text.replace("large language model", "large language models")
    text = re.sub(r"\s+", " ", text)
    return text


def skill_pattern(skill):
    escaped = re.escape(skill.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"


def extract_skills(text):
    normalized = normalize_for_skill_matching(text)
    found = []

    for skill in SKILLS:
        if re.search(skill_pattern(skill), normalized):
            found.append(skill)

    return found


@st.cache_data(show_spinner=False)
def precompute_resume_skills(text_series):
    # Cache skill extraction for the whole dataset.
    return [extract_skills(text) for text in text_series]


if ENGINE_READY:
    try:
        resume_skill_lists = precompute_resume_skills(tuple(df["cleaned_resume"].tolist()))
    except Exception:
        resume_skill_lists = [[] for _ in range(len(df))]
else:
    resume_skill_lists = []


# ============================================================
# JOB DOMAIN / ROLE ANALYZER
# ============================================================

DOMAIN_RULES = {
    "Artificial Intelligence & Machine Learning": [
        "artificial intelligence", "machine learning", "deep learning",
        "nlp", "natural language processing", "computer vision",
        "tensorflow", "pytorch", "scikit-learn", "neural network",
        "generative ai", "llm", "large language model", "rag"
    ],
    "Data Science & Analytics": [
        "data science", "data analyst", "data analysis", "analytics",
        "statistics", "pandas", "numpy", "power bi", "tableau",
        "sql", "data visualization", "business intelligence"
    ],
    "Software & Web Development": [
        "software engineer", "software developer", "web developer",
        "frontend", "backend", "full stack", "react", "node.js",
        "javascript", "typescript", "api", "django", "flask",
        "fastapi", "express"
    ],
    "Cloud & DevOps": [
        "devops", "cloud", "aws", "azure", "gcp", "docker",
        "kubernetes", "ci/cd", "deployment", "infrastructure"
    ],
    "Cybersecurity": [
        "cybersecurity", "cyber security", "penetration testing",
        "ethical hacking", "network security", "vulnerability",
        "security analyst", "siem"
    ],
    "Human Resources": [
        "human resources", "hr", "recruitment", "talent acquisition",
        "employee relations", "onboarding", "payroll"
    ],
    "Finance & Banking": [
        "finance", "financial analyst", "banking", "investment",
        "accounting", "credit", "risk management", "financial modeling"
    ],
    "Marketing & Sales": [
        "marketing", "sales", "business development", "seo",
        "social media", "digital marketing", "lead generation"
    ],
}


def analyze_job_description(jd):
    text = normalize_for_skill_matching(jd)

    scores = {}
    for domain, keywords in DOMAIN_RULES.items():
        score = 0
        for keyword in keywords:
            if re.search(skill_pattern(keyword), text):
                score += 1
        scores[domain] = score

    best_domain = max(scores, key=scores.get)

    if scores[best_domain] == 0:
        best_domain = "General / Multi-Disciplinary"

    skills = extract_skills(jd)

    role = "General Professional Role"

    role_patterns = [
        (r"\bai\s*/?\s*ml\b|\bai/ml\b|\bmachine learning engineer\b", "AI/ML Engineer"),
        (r"\bdata scientist\b", "Data Scientist"),
        (r"\bdata analyst\b", "Data Analyst"),
        (r"\bml engineer\b", "Machine Learning Engineer"),
        (r"\bsoftware engineer\b", "Software Engineer"),
        (r"\bsoftware developer\b", "Software Developer"),
        (r"\bweb developer\b", "Web Developer"),
        (r"\bfull stack\b", "Full Stack Developer"),
        (r"\bfrontend\b|\bfront-end\b", "Frontend Developer"),
        (r"\bbackend\b|\bback-end\b", "Backend Developer"),
        (r"\bdevops\b", "DevOps Engineer"),
        (r"\bcloud\b", "Cloud Engineer"),
        (r"\bcybersecurity\b|\bcyber security\b", "Cybersecurity Engineer"),
        (r"\bhr\b|\bhuman resources\b", "HR / Recruitment"),
        (r"\bfinancial analyst\b", "Financial Analyst"),
        (r"\bmarketing\b", "Marketing Specialist"),
    ]

    for pattern, role_name in role_patterns:
        if re.search(pattern, text):
            role = role_name
            break

    if "intern" in text:
        level = "Internship / Entry Level"
    elif "senior" in text or "lead" in text:
        level = "Senior / Lead"
    elif "junior" in text:
        level = "Junior"
    else:
        level = "Professional / Entry-to-Mid Level"

    education = "Related bachelor's degree or equivalent practical experience"
    if any(x in text for x in ["computer science", "artificial intelligence", "machine learning", "data science"]):
        education = "Computer Science / AI / ML / Data Science or related degree"

    return {
        "domain": best_domain,
        "role": role,
        "level": level,
        "education": education,
        "skills": skills,
    }


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(uploaded_file):
    uploaded_file.seek(0)
    return docx2txt.process(uploaded_file)


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    return uploaded_file.read().decode("utf-8", errors="ignore")


def predict_category(cleaned):
    vec = classifier_vectorizer.transform([cleaned])
    return classifier_model.predict(vec)[0]


# ============================================================
# HYBRID MATCHING
# ============================================================

def calculate_hybrid_score(cosine_score, matched_count, required_count):
    skill_coverage = (
        matched_count / required_count
        if required_count > 0
        else 0.0
    )

    hybrid = (0.55 * cosine_score) + (0.45 * skill_coverage)
    return hybrid, skill_coverage


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-row">
                <div class="brand-mark">🎯</div>
                <div class="brand-name">TalentAI</div>
            </div>
            <div class="brand-sub">Intelligent Recruitment Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "🏠  Dashboard",
        "🔍  Analyze Job",
        "👥  Find Candidates",
        "📤  Upload Resumes",
        "📁  Resume Database",
        "📊  Analytics",
        "ℹ️  About",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="status-pill">
            <span class="status-dot"></span>
            AI Screening Engine Online
        </div>
        <div class="small-muted" style="margin-top:12px;">
            TF-IDF · Skill Matching · ML Classification
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENGINE ERROR
# ============================================================

if not ENGINE_READY:
    st.error("TalentAI could not load its saved ML resources.")
    st.code(ENGINE_ERROR)
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if selected_page.startswith("🏠"):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">AI-POWERED RECRUITMENT</div>
                <h1>Find the right talent <span>faster.</span></h1>
                <p>
                    TalentAI analyzes job descriptions, identifies the role and
                    required skills, then matches resumes using TF-IDF similarity
                    and technical skill coverage.
                </p>
                <div class="hero-mini">
                    <span>⚡ Fast JD Analysis</span>
                    <span>🧠 ML Matching</span>
                    <span>🎯 Skill Intelligence</span>
                    <span>📄 PDF / DOCX Support</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">📄 Resume Database</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-note">Candidates indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">🏷️ Categories</div>
                <div class="kpi-value">{df["Category"].nunique()}</div>
                <div class="kpi-note">Resume domains</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">🧠 Skills Tracked</div>
                <div class="kpi-value">{len(SKILLS)}</div>
                <div class="kpi-note">Technical signals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="kpi">
                <div class="kpi-label">⚡ Screening Engine</div>
                <div class="kpi-value">Online</div>
                <div class="kpi-note">Ready to analyze</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">How TalentAI works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A simple three-stage workflow from job description to shortlist.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-number">01 — UNDERSTAND</div>
                <div class="feature-title">Analyze the Job</div>
                <div class="feature-text">
                    TalentAI reads the job description and identifies the likely
                    domain, role, experience level and technical skills.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-number">02 — ANALYZE</div>
                <div class="feature-title">Understand Resumes</div>
                <div class="feature-text">
                    Resume text is transformed through the existing TF-IDF
                    pipeline and technical skills are extracted for matching.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-number">03 — RANK</div>
                <div class="feature-title">Rank Candidates</div>
                <div class="feature-text">
                    Candidates receive a hybrid score combining textual
                    similarity with technical skill coverage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            TalentAI · AI-Based Resume Screening System · AIML-451 Minor Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYZE JOB
# ============================================================

elif selected_page.startswith("🔍"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">JOB INTELLIGENCE</div>
            <div class="page-title">Analyze a Job Description</div>
            <div class="page-description">
                Understand what the role is looking for before matching candidates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd = st.text_area(
        "Job Description",
        height=280,
        placeholder="Paste the complete job description here...",
        key="analyze_jd",
    )

    if st.button("✨ Analyze Job Description", use_container_width=False):
        if not jd.strip():
            st.warning("Please paste a job description first.")
        else:
            analysis = analyze_job_description(jd)
            st.session_state["active_jd"] = jd
            st.session_state["job_analysis"] = analysis

    if "job_analysis" in st.session_state:
        analysis = st.session_state["job_analysis"]

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1.1, 1])

        with left:
            st.markdown(
                f"""
                <div class="domain-card">
                    <div class="domain-label">Primary Job Domain</div>
                    <div class="domain-name">{analysis["domain"]}</div>
                    <div class="role-name">Recommended role · {analysis["role"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f"""
                <div class="analysis-box">
                    <div class="analysis-label">Experience Level</div>
                    <div class="analysis-value">{analysis["level"]}</div>
                </div>
                <div class="analysis-box">
                    <div class="analysis-label">Education Profile</div>
                    <div class="analysis-value">{analysis["education"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Detected Technical Skills</div>', unsafe_allow_html=True)

        if analysis["skills"]:
            tags = "".join(
                f'<span class="tag primary">{skill}</span>'
                for skill in analysis["skills"]
            )
            st.markdown(
                f'<div style="line-height:2.4">{tags}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No tracked technical skills were detected in this JD.")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("👥 Continue to Candidate Matching"):
            st.session_state["active_jd"] = jd
            st.success("Job description saved. Open 'Find Candidates' from the sidebar.")


# ============================================================
# FIND CANDIDATES
# ============================================================

elif selected_page.startswith("👥"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">CANDIDATE INTELLIGENCE</div>
            <div class="page-title">Find Best Candidates</div>
            <div class="page-description">
                Match the selected job against the existing resume database.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_jd = st.session_state.get("active_jd", "")

    jd = st.text_area(
        "Job Description",
        value=default_jd,
        height=230,
        placeholder="Paste a job description or analyze one first...",
        key="candidate_jd",
    )

    top_n = st.slider(
        "Number of top matches",
        min_value=1,
        max_value=20,
        value=5,
    )

    if st.button("🎯 Find Best Matches", use_container_width=False):
        if not jd.strip():
            st.warning("Please paste a job description first.")
        else:
            start = time.time()

            with st.spinner("TalentAI is ranking the resume database..."):
                cleaned_jd = clean_text(jd)
                jd_vector = vectorizer.transform([cleaned_jd])

                cosine_scores = cosine_similarity(
                    jd_vector,
                    resume_vectors
                ).flatten()

                job_skills = extract_skills(jd)
                required_count = len(job_skills)

                hybrid_scores = []
                coverages = []
                matched_lists = []

                for i, cosine_score in enumerate(cosine_scores):
                    candidate_skills = set(resume_skill_lists[i])
                    matched = [s for s in job_skills if s in candidate_skills]

                    hybrid, coverage = calculate_hybrid_score(
                        float(cosine_score),
                        len(matched),
                        required_count,
                    )

                    hybrid_scores.append(hybrid)
                    coverages.append(coverage)
                    matched_lists.append(matched)

                results = df[["ID", "Category"]].copy()
                results["Match Score"] = hybrid_scores
                results["Cosine Similarity"] = cosine_scores
                results["Skill Coverage"] = coverages
                results["Matched Skills"] = [
                    ", ".join(x) if x else "—"
                    for x in matched_lists
                ]

                results = (
                    results
                    .sort_values("Match Score", ascending=False)
                    .head(top_n)
                    .reset_index(drop=True)
                )

            elapsed = time.time() - start

            st.success(f"Screening complete in {elapsed:.2f} seconds")

            analysis = analyze_job_description(jd)
            st.session_state["active_jd"] = jd
            st.session_state["job_analysis"] = analysis

            if job_skills:
                tags = "".join(
                    f'<span class="tag primary">{s}</span>'
                    for s in job_skills
                )
                st.markdown(
                    f"""
                    <div class="analysis-box">
                        <div class="analysis-label">Detected Job Skills</div>
                        <div style="margin-top:8px;line-height:2.3">{tags}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            display = results.copy()
            display["Match Score"] = display["Match Score"].map(lambda x: f"{x:.1%}")
            display["Cosine Similarity"] = display["Cosine Similarity"].map(lambda x: f"{x:.3f}")
            display["Skill Coverage"] = display["Skill Coverage"].map(lambda x: f"{x:.1%}")

            st.markdown('<div class="section-title">Top Matches</div>', unsafe_allow_html=True)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Match Score = 55% TF-IDF textual similarity + 45% technical skill coverage. "
                "The Category column is the original dataset category."
            )


# ============================================================
# UPLOAD RESUMES
# ============================================================

elif selected_page.startswith("📤"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">DOCUMENT SCREENING</div>
            <div class="page-title">Upload Resumes</div>
            <div class="page-description">
                Upload PDF or DOCX resumes and screen them against a job description.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_jd = st.session_state.get("active_jd", "")

    jd = st.text_area(
        "Job Description",
        value=default_jd,
        height=210,
        placeholder="Paste the job description...",
        key="upload_jd",
    )

    files = st.file_uploader(
        "Upload candidate resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="PDF and DOCX files are supported.",
    )

    if st.button("🚀 Screen Uploaded Resumes"):
        if not jd.strip():
            st.warning("Please paste a job description first.")
        elif not files:
            st.warning("Please upload at least one resume.")
        else:
            start = time.time()

            cleaned_jd = clean_text(jd)
            jd_vector = vectorizer.transform([cleaned_jd])
            job_skills = extract_skills(jd)
            required_count = len(job_skills)

            rows = []

            with st.spinner("Extracting and screening resumes..."):
                for file in files:
                    try:
                        raw = extract_text(file)

                        if not raw.strip():
                            rows.append({
                                "File Name": file.name,
                                "Predicted Category": "Unreadable",
                                "Match Score": 0.0,
                                "Skill Coverage": 0.0,
                                "Matched Skills": "No text extracted",
                            })
                            continue

                        cleaned = clean_text(raw)
                        resume_vector = vectorizer.transform([cleaned])
                        cosine_score = float(
                            cosine_similarity(jd_vector, resume_vector).flatten()[0]
                        )

                        resume_skills = set(extract_skills(raw))
                        matched = [s for s in job_skills if s in resume_skills]

                        hybrid, coverage = calculate_hybrid_score(
                            cosine_score,
                            len(matched),
                            required_count,
                        )

                        try:
                            category = predict_category(cleaned)
                        except Exception:
                            category = "Unknown"

                        rows.append({
                            "File Name": file.name,
                            "Predicted Category": category,
                            "Match Score": hybrid,
                            "Skill Coverage": coverage,
                            "Matched Skills": ", ".join(matched) if matched else "—",
                        })

                    except Exception as exc:
                        rows.append({
                            "File Name": file.name,
                            "Predicted Category": "Error",
                            "Match Score": 0.0,
                            "Skill Coverage": 0.0,
                            "Matched Skills": str(exc),
                        })

            result_df = (
                pd.DataFrame(rows)
                .sort_values("Match Score", ascending=False)
                .reset_index(drop=True)
            )

            elapsed = time.time() - start
            st.success(f"Screened {len(files)} resume(s) in {elapsed:.2f} seconds")

            show_df = result_df.copy()
            show_df["Match Score"] = show_df["Match Score"].map(lambda x: f"{x:.1%}")
            show_df["Skill Coverage"] = show_df["Skill Coverage"].map(lambda x: f"{x:.1%}")

            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# RESUME DATABASE
# ============================================================

elif selected_page.startswith("📁"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">DATASET EXPLORER</div>
            <div class="page-title">Resume Database</div>
            <div class="page-description">
                Explore the resumes currently indexed by TalentAI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns([2, 1])

    with q1:
        search = st.text_input(
            "Search resume text",
            placeholder="e.g. python, machine learning, finance...",
        )

    with q2:
        category_options = ["All Categories"] + sorted(df["Category"].dropna().unique().tolist())
        category = st.selectbox("Category", category_options)

    filtered = df.copy()

    if search.strip():
        mask = (
            filtered["Resume_str"]
            .fillna("")
            .str.contains(search.strip(), case=False, na=False)
        )
        filtered = filtered[mask]

    if category != "All Categories":
        filtered = filtered[filtered["Category"] == category]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} resumes")

    st.dataframe(
        filtered[["ID", "Category"]].head(250),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ANALYTICS
# ============================================================

elif selected_page.startswith("📊"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">SYSTEM ANALYTICS</div>
            <div class="page-title">Resume Analytics</div>
            <div class="page-description">
                A quick view of the resume database and category distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = (
        df["Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Resumes")
    )

    a1, a2 = st.columns([1.25, 1])

    with a1:
        st.markdown('<div class="section-title">Resume distribution</div>', unsafe_allow_html=True)
        st.bar_chart(counts.set_index("Category"))

    with a2:
        st.markdown('<div class="section-title">Dataset summary</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="analysis-box">
                <div class="analysis-label">Total Resumes</div>
                <div class="analysis-value">{len(df):,}</div>
            </div>
            <div class="analysis-box">
                <div class="analysis-label">Unique Categories</div>
                <div class="analysis-value">{df["Category"].nunique()}</div>
            </div>
            <div class="analysis-box">
                <div class="analysis-label">Tracked Technical Skills</div>
                <div class="analysis-value">{len(SKILLS)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT
# ============================================================

else:
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">ABOUT TALENTAI</div>
            <div class="page-title">Intelligent Recruitment, explained.</div>
            <div class="page-description">
                A practical AI/ML minor project for job description analysis and resume matching.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-number">THE PIPELINE</div>
                <div class="feature-title">How the matching engine works</div>
                <div class="feature-text">
                    Job descriptions and resumes are cleaned and transformed
                    through the existing TF-IDF vectorization pipeline.
                    Cosine similarity measures textual relevance, while a
                    technical skill engine measures overlap with required skills.
                    The final hybrid score combines both signals.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-number">IMPORTANT</div>
                <div class="feature-title">What the dataset category means</div>
                <div class="feature-text">
                    The Category shown for a resume is its original dataset
                    label. It is not a percentage qualification score and it
                    does not mean the candidate is necessarily unsuitable for
                    another technical role.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            TalentAI · AIML-451 · AI-Based Resume Screening System
        </div>
        """,
        unsafe_allow_html=True,
    )
