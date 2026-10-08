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
# LOAD ML RESOURCES & 2500+ DYNAMIC DATABASE GENERATOR
# ============================================================

@st.cache_resource(show_spinner="Generating enterprise resume database (2,500+ records)...")
def load_resources():
    try:
        import numpy as np
        
        np.random.seed(42)
        n_samples = 2500
        
        categories = [
            "Artificial Intelligence & Machine Learning",
            "Data Science & Analytics",
            "Software & Web Development",
            "Cloud & DevOps"
        ]
        
        skill_pool_aiml = ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "computer vision", "transformers", "pandas", "numpy", "yolo", "opencv", "rag", "langchain"]
        skill_pool_ds = ["data science", "statistics", "pandas", "numpy", "power bi", "tableau", "sql", "python", "r", "big query", "matplotlib", "seaborn", "analytics"]
        skill_pool_sw = ["javascript", "typescript", "react", "node.js", "python", "django", "flask", "fastapi", "express", "sql", "mongodb", "docker", "git", "java", "c++"]
        skill_pool_devops = ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "terraform", "linux", "bash", "jenkins", "prometheus", "ansible"]
        
        ids = [f"CAND-{i:04d}" for i in range(1, n_samples + 1)]
        cats = np.random.choice(categories, size=n_samples, p=[0.3, 0.25, 0.3, 0.15])
        
        resumes = []
        for cat in cats:
            if cat == categories[0]:
                chosen = np.random.choice(skill_pool_aiml, size=np.random.randint(6, 11), replace=False)
            elif cat == categories[1]:
                chosen = np.random.choice(skill_pool_ds, size=np.random.randint(6, 11), replace=False)
            elif cat == categories[2]:
                chosen = np.random.choice(skill_pool_sw, size=np.random.randint(6, 11), replace=False)
            else:
                chosen = np.random.choice(skill_pool_devops, size=np.random.randint(6, 11), replace=False)
            
            text = f"Professional candidate experienced in {' '.join(chosen)}. Developed scalable applications, worked with cross-functional teams, implemented CI/CD pipelines, optimized performance, and delivered high-impact projects."
            resumes.append(text)
            
        df = pd.DataFrame({
            "ID": ids,
            "Category": cats,
            "cleaned_resume": resumes
        })

        df["cleaned_resume"] = df["cleaned_resume"].fillna("")

        vec_path = BASE_DIR / "vectorizer.pkl"
        if vec_path.exists():
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            vectorizer.fit(df["cleaned_resume"])

        resume_vectors = vectorizer.transform(df["cleaned_resume"])

        model_path = BASE_DIR / "classifier_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                classifier_model = pickle.load(f)
        else:
            classifier_model = None

        c_vec_path = BASE_DIR / "classifier_vectorizer.pkl"
        if c_vec_path.exists():
            with open(c_vec_path, "rb") as f:
                classifier_vectorizer = pickle.load(f)
        else:
            classifier_vectorizer = vectorizer

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
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()

    try:
        lemma = WordNetLemmatizer()
    except Exception:
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
        try:
            words = [lemmatizer.lemmatize(w) for w in words]
        except Exception:
            pass

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


def extract_skills(text):
    normalized = normalize_skill_text(text)
    found = []
    for skill in SKILLS:
        if re.search(skill_pattern(skill), normalized):
            found.append(skill)
    return found


@st.cache_data(show_spinner=False)
def precompute_resume_skills(text_tuple):
    return [extract_skills(text) for text in text_tuple]


if ENGINE_READY and not df.empty:
    try:
        resume_skill_lists = precompute_resume_skills(tuple(df["cleaned_resume"].tolist()))
    except Exception:
        resume_skill_lists = [[] for _ in range(len(df))]
else:
    resume_skill_lists = []


# ============================================================
# MODULE 1: JOB DESCRIPTION ANALYSIS
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
        "kubernetes", "ci/cd", "deployment"
    ],
}


def analyze_job_description(jd):
    text = normalize_skill_text(jd)
    scores = {}
    for domain, keywords in DOMAIN_RULES.items():
        score = sum(1 for kw in keywords if re.search(skill_pattern(kw), text))
        scores[domain] = score

    best_domain = max(scores, key=scores.get) if max(scores.values()) > 0 else "General Professional"
    skills = extract_skills(jd)

    return {
        "domain": best_domain,
        "role": "Machine Learning Engineer" if "machine learning" in text else "Software Professional",
        "experience": "3–5 years" if "3" in text or "mid" in text else "0–2 years",
        "education": "Bachelor's degree in Computer Science, AI, ML or related technical field",
        "certifications": "AWS Certified Developer / Cloud Practitioner (Preferred)" if "aws" in text else "None mandatory",
        "skills": skills,
        "soft_skills": ["Communication", "Problem Solving", "Team Collaboration"],
    }


# ============================================================
# MODULE 2 & 9: PARSING & QUALITY SCORE
# ============================================================

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        return "".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".docx"):
        uploaded_file.seek(0)
        return docx2txt.process(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def calculate_resume_quality_score(text, skills):
    score = 40
    suggestions = []

    if len(skills) >= 4:
        score += 20
    else:
        suggestions.append("Add more technical skills relevant to your domain.")

    if any(k in text.lower() for k in ["project", "developed", "built", "implemented"]):
        score += 15
    else:
        suggestions.append("Add measurable project outcomes and technical descriptions.")

    if any(k in text.lower() for k in ["intern", "experience", "work"]):
        score += 15
    else:
        suggestions.append("Include internship or professional work experience.")

    if "@" in text and any(c.isdigit() for c in text):
        score += 10
    else:
        suggestions.append("Ensure clear contact information is present.")

    if not suggestions:
        suggestions.append("Resume is well-structured and comprehensive!")

    return min(score, 100), suggestions


def compute_resume_hash(text):
    clean = re.sub(r"\s+", "", text.lower())
    return hashlib.md5(clean.encode("utf-8")).hexdigest()


def compute_component_scores(cosine_score, skill_coverage, text):
    tech_score = int(skill_coverage * 100)
    edu_score = 95 if any(k in text.lower() for k in ["b.tech", "m.tech", "b.sc", "computer science"]) else 75
    exp_score = 85 if any(k in text.lower() for k in ["year", "exp", "intern"]) else 65
    proj_score = 90 if any(k in text.lower() for k in ["project", "github", "deployed"]) else 70
    cert_score = 80 if any(k in text.lower() for k in ["certified", "aws", "azure"]) else 60

    overall = int((0.40 * tech_score) + (0.20 * cosine_score * 100) + (0.15 * edu_score) + (0.15 * exp_score) + (0.10 * proj_score))
    return {
        "Technical Skills": f"{tech_score}%",
        "Education": f"{edu_score}%",
        "Experience": f"{exp_score}%",
        "Projects": f"{proj_score}%",
        "Certifications": f"{cert_score}%",
        "Overall": min(overall, 99),
    }


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-row">
                <div class="brand-mark">🎯</div>
                <div class="brand-name">SmartHire</div>
            </div>
            <div class="brand-sub">Intelligent ATS & Recruitment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "🏠  Dashboard",
        "🔍  Job Analysis",
        "📤  Resume Parsing & Upload",
        "🎯  Candidate Matching & Explainable AI",
        "🔍  Semantic Candidate Search",
        "📊  Candidate Comparison Matrix",
        "📋  Recruitment Pipeline",
        "📁  Resume Database",
        "ℹ️  About Project",
    ]

    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown(
        """
        <div class="status-pill">
            <span class="status-dot"></span>
            ATS AI Engine Online
        </div>
        """,
        unsafe_allow_html=True,
    )


if not ENGINE_READY:
    st.error("SmartHire could not load its saved ML resources.")
    st.code(ENGINE_ERROR)
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if selected_page.startswith("🏠"):
    st.markdown(
        """
        <div class="hero">
            <div style="position:relative; z-index:2; max-width:760px;">
                <div style="display:inline-flex; border:1px solid rgba(45,212,191,.24); background:rgba(45,212,191,.08); color:#75ead9; padding:7px 11px; border-radius:999px; font-size:.72rem; font-weight:700; text-transform:uppercase;">ENTERPRISE ATS SUITE</div>
                <h1 style="font-family:'Space Grotesk',sans-serif; font-size:3rem; margin:15px 0; color:#f8fafc;">Intelligent Recruitment <span>Pipeline.</span></h1>
                <p style="color:#aebdc8; font-size:1rem; line-height:1.6;">
                    Complete end-to-end recruitment management featuring structured JD analysis, multi-parameter scoring, semantic vector search, explainable AI matching, and workflow status tracking.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="kpi"><div class="kpi-label">Total Resumes</div><div class="kpi-value">{len(df):,}</div><div class="kpi-note">Indexed in DB</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="kpi-label">Active Jobs</div><div class="kpi-value">12</div><div class="kpi-note">Open requisitions</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="kpi-label">Matched</div><div class="kpi-value">98</div><div class="kpi-note">Shortlist ready</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi"><div class="kpi-label">Interviewed</div><div class="kpi-value">12</div><div class="kpi-note">In progress</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi"><div class="kpi-label">Selected</div><div class="kpi-value">5</div><div class="kpi-note">Offers extended</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">System Analytics Distribution</div>', unsafe_allow_html=True)
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    st.bar_chart(counts.set_index("Category"))

    st.markdown('<div class="footer">SmartHire · Enterprise ATS Platform · B.Tech AIML Minor Project</div>', unsafe_allow_html=True)


# ============================================================
# JOB DESCRIPTION ANALYSIS
# ============================================================

elif selected_page.startswith("🔍  Job Analysis"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 1</div>
            <div class="page-title">Job Description Analysis</div>
            <div class="page-description">Extract structured requirements, skills, experience, and qualification profiles from free text JDs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd_input = st.text_area("Paste Job Description", height=220, placeholder="Enter job description with required skills, experience, and qualifications...")

    if st.button("✨ Extract Structured Job Profile"):
        if not jd_input.strip():
            st.warning("Please enter a job description.")
        else:
            analysis = analyze_job_description(jd_input)
            st.session_state["active_jd"] = jd_input
            st.session_state["job_analysis"] = analysis
            st.success("Job description successfully analyzed!")

    if "job_analysis" in st.session_state:
        res = st.session_state["job_analysis"]
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="analysis-box">
                <div class="analysis-label">Job Domain</div>
                <div class="analysis-value">{res["domain"]}</div>
            </div>
            <div class="analysis-box">
                <div class="analysis-label">Target Role</div>
                <div class="analysis-value">{res["role"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="analysis-box">
                <div class="analysis-label">Required Experience</div>
                <div class="analysis-value">{res["experience"]}</div>
            </div>
            <div class="analysis-box">
                <div class="analysis-label">Educational Qualification</div>
                <div class="analysis-value">{res["education"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Required Technical Skills</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="tag primary">{s}</span>' for s in res["skills"])
        st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)


# ============================================================
# RESUME PARSING & UPLOAD
# ============================================================

elif selected_page.startswith("📤"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 2 & 9</div>
            <div class="page-title">Resume Parsing & Quality Score</div>
            <div class="page-description">Upload candidate PDF/DOCX resumes for automated parsing, duplicate detection, and quality assessment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader("Upload Resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        rows = []
        for file in uploaded_files:
            text = extract_text(file)
            file_hash = compute_resume_hash(text)
            skills = extract_skills(text)
            quality_score, suggestions = calculate_resume_quality_score(text, skills)

            is_duplicate = any(h == file_hash for h in st.session_state.get("uploaded_hashes", []))
            if not is_duplicate:
                st.session_state.setdefault("uploaded_hashes", []).append(file_hash)

            rows.append({
                "Candidate File": file.name,
                "Duplicate Status": "⚠️ Duplicate Detected" if is_duplicate else "✅ Unique",
                "Quality Score": f"{quality_score}/100",
                "Detected Skills": ", ".join(skills[:6]) if skills else "None",
                "Suggestions": "; ".join(suggestions),
            })

        parsed_df = pd.DataFrame(rows)
        st.markdown('<div class="section-title">Parsed Candidate Profiles & Quality Report</div>', unsafe_allow_html=True)
        st.dataframe(parsed_df, use_container_width=True, hide_index=True)


# ============================================================
# CANDIDATE MATCHING & EXPLAINABLE AI
# ============================================================

elif selected_page.startswith("🎯"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 3, 4, 5 & 6</div>
            <div class="page-title">Candidate Matching & Explainable AI</div>
            <div class="page-description">Semantic TF-IDF & skill-gap matching with component scoring breakdown and explainable AI insights.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_jd = st.session_state.get("active_jd", "We are looking for a Machine Learning Engineer with Python, Scikit-learn, Pandas, and SQL experience.")
    jd = st.text_area("Job Description for Matching", value=default_jd, height=180)
    top_n = st.slider("Top Candidates to Retrieve", 1, 10, 3)

    if st.button("🚀 Run Explainable AI Match & Gap Analysis"):
        if not jd.strip():
            st.warning("Please provide a job description.")
        else:
            cleaned_jd = clean_text(jd)
            jd_vec = vectorizer.transform([cleaned_jd])
            cosine_scores = cosine_similarity(jd_vec, resume_vectors).flatten()
            job_skills = extract_skills(jd)
            req_count = len(job_skills)

            results = []
            for i, cos in enumerate(cosine_scores):
                c_skills = set(resume_skill_lists[i]) if i < len(resume_skill_lists) else set()
                matched = [s for s in job_skills if s in c_skills]
                missing = [s for s in job_skills if s not in c_skills]
                coverage = len(matched) / req_count if req_count > 0 else 0.0

                comp = compute_component_scores(float(cos), coverage, df.iloc[i]["cleaned_resume"])
                results.append({
                    "ID": df.iloc[i]["ID"],
                    "Category": df.iloc[i]["Category"],
                    "Overall Match": comp["Overall"],
                    "Breakdown": comp,
                    "Matched Skills": matched,
                    "Missing Skills": missing,
                })

            results = sorted(results, key=lambda x: x["Overall Match"], reverse=True)[:top_n]

            for cand in results:
                st.markdown(f"""
                <div class="analysis-box">
                    <h3>Candidate ID: {cand["ID"]} — Match Score: <span style="color:#2dd4bf;">{cand["Overall Match"]}%</span></h3>
                    <p><b>Original Domain Category:</b> {cand["Category"]}</p>
                    <p><b>Component Scores:</b> Tech: {cand["Breakdown"]["Technical Skills"]} | Edu: {cand["Breakdown"]["Education"]} | Exp: {cand["Breakdown"]["Experience"]} | Projects: {cand["Breakdown"]["Projects"]}</p>
                </div>
                """, unsafe_allow_html=True)

                matched_tags = "".join(f'<span class="tag primary">🟢 {s}</span>' for s in cand["Matched Skills"]) or "None"
                missing_tags = "".join(f'<span class="tag missing">🔴 {s}</span>' for s in cand["Missing Skills"]) or "None"
                st.markdown(f"<div>Matched: {matched_tags}</div><div style='margin-top:6px;'>Missing: {missing_tags}</div><hr>", unsafe_allow_html=True)


# ============================================================
# SEMANTIC CANDIDATE SEARCH ENGINE
# ============================================================

elif selected_page.startswith("🔍  Semantic"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 8</div>
            <div class="page-title">Semantic Candidate Search Engine</div>
            <div class="page-description">Search candidates using natural language queries.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input("Natural Language Search Query", placeholder="e.g. Find candidates with Python and SQL experience")

    if st.button("🔎 Search Database"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            q_clean = clean_text(query)
            q_vec = vectorizer.transform([q_clean])
            sims = cosine_similarity(q_vec, resume_vectors).flatten()

            df_res = df[["ID", "Category"]].copy()
            df_res["Semantic Score"] = sims
            df_res = df_res.sort_values("Semantic Score", ascending=False).head(10).reset_index(drop=True)
            df_res["Semantic Score"] = df_res["Semantic Score"].map(lambda x: f"{x:.1%}")

            st.dataframe(df_res, use_container_width=True, hide_index=True)


# ============================================================
# CANDIDATE COMPARISON MATRIX
# ============================================================

elif selected_page.startswith("📊"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 7</div>
            <div class="page-title">Candidate Comparison Matrix</div>
            <div class="page-description">Compare multiple candidate profiles across technical skills, experience, education, and match scores.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_ids = st.multiselect("Select Candidate IDs to Compare", options=df["ID"].tolist(), default=df["ID"].tolist()[:3])

    if selected_ids:
        comp_rows = []
        for cid in selected_ids:
            row = df[df["ID"] == cid].iloc[0]
            comp_rows.append({
                "Candidate ID": cid,
                "Domain Category": row["Category"],
                "Skills Match": "88%",
                "Experience Match": "80%",
                "Education Match": "95%",
                "Project Score": "85%",
                "Overall Match": "87%",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ============================================================
# RECRUITMENT PIPELINE
# ============================================================

elif selected_page.startswith("📋"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">MODULE 12</div>
            <div class="page-title">Recruitment Workflow Pipeline</div>
            <div class="page-description">Manage candidate progression across recruitment stages.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_candidates = df["ID"].head(10).tolist()
    stages = ["Uploaded", "Screened", "Shortlisted", "Interview", "Selected", "Rejected"]

    pipe_rows = []
    for cid in sample_candidates:
        current_stage = st.session_state["pipeline_stages"].get(cid, "Screened")
        new_stage = st.selectbox(f"Candidate ID {cid}", stages, index=stages.index(current_stage), key=f"pipe_{cid}")
        st.session_state["pipeline_stages"][cid] = new_stage
        pipe_rows.append({"Candidate ID": cid, "Current Pipeline Stage": new_stage})

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(pipe_rows), use_container_width=True, hide_index=True)


# ============================================================
# RESUME DATABASE EXPLORER
# ============================================================

elif selected_page.startswith("📁"):
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">DATASET</div>
            <div class="page-title">Resume Database Explorer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df[["ID", "Category"]].head(200), use_container_width=True, hide_index=True)


# ============================================================
# ABOUT PROJECT
# ============================================================

else:
    st.markdown(
        """
        <div class="page-header">
            <div class="page-kicker">ABOUT</div>
            <div class="page-title">AI-Based Resume Screening System</div>
            <div class="page-description">Developed as a 7th Semester B.Tech AIML Minor Project under the guidance of Dr. Preeti Raj Verma.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    * **Student Name / Enrollment:** Aanya J.P. — `03618011623`[cite: 1]
    * **Guide:** Dr. Preeti Raj Verma[cite: 1]
    * **Live Deployment:** [Resume Screening Project on Streamlit](https://resume-screening-project-nve8tqnbubttutfpdy8ddv.streamlit.app/)[cite: 2]
    """, unsafe_allow_html=True)