import streamlit as st
import pandas as pd
import pickle
import re
import time
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx2txt


# ---------------- Page setup ----------------
st.set_page_config(page_title="📜AI Resume Screening System", layout="wide")
st.title(" AI-Based Resume Screening System")
st.caption("Minor Project — AIML-451")

# ---------------- Load saved models (only once, cached) ----------------
@st.cache_resource
def load_resources():
    df = pd.read_csv("data/cleaned_resumes.csv")
    df["cleaned_resume"] = df["cleaned_resume"].fillna("")

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("resume_vectors.pkl", "rb") as f:
        resume_vectors = pickle.load(f)
    with open("classifier_model.pkl", "rb") as f:
        classifier_model = pickle.load(f)
    with open("classifier_vectorizer.pkl", "rb") as f:
        classifier_vectorizer = pickle.load(f)

    return df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer

df, vectorizer, resume_vectors, classifier_model, classifier_vectorizer = load_resources()
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- Shared helper functions ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def predict_category(cleaned_text):
    vec = classifier_vectorizer.transform([cleaned_text])
    return classifier_model.predict(vec)[0]

# ---------------- Two modes via tabs ----------------
tab1, tab2 = st.tabs(["🔍 Search Existing Database", "📤 Upload New Resumes"])

# ===== TAB 1: Search existing dataset =====
with tab1:
    st.subheader("Rank resumes from the existing dataset")

    jd_text_1 = st.text_area(
        "Paste the Job Description here:",
        height=150,
        key="jd1",
        placeholder="e.g. We are looking for an HR Administrator with experience in recruitment, onboarding, and HR policy management..."
    )
    top_n = st.slider("Number of top matches to show:", 1, 20, 5)

    if st.button("Find Best Matches", key="btn1"):
        if jd_text_1.strip() == "":
            st.warning("Please paste a job description first.")
        else:
            start_time = time.time()

            cleaned_jd = clean_text(jd_text_1)
            jd_vector = vectorizer.transform([cleaned_jd])
            scores = cosine_similarity(jd_vector, resume_vectors).flatten()

            results = df.copy()
            results["Match Score"] = scores
            results = results.sort_values(by="Match Score", ascending=False).head(top_n)

            elapsed = time.time() - start_time

            st.success(f"Screening complete in {elapsed:.3f} seconds")
            st.dataframe(
                results[["ID", "Category", "Match Score"]].style.format({"Match Score": "{:.3f}"}),
                use_container_width=True
            )

# ===== TAB 2: Upload new resumes =====
with tab2:
    st.subheader("Upload real resumes and screen them against a job description")

    jd_text_2 = st.text_area(
        "Paste the Job Description here:",
        height=150,
        key="jd2",
        placeholder="Paste the job description you want to screen candidates against..."
    )

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF or DOCX, multiple allowed):",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("Screen Uploaded Resumes", key="btn2"):
        if jd_text_2.strip() == "":
            st.warning("Please paste a job description first.")
        elif not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            start_time = time.time()
            cleaned_jd = clean_text(jd_text_2)
            jd_vector = vectorizer.transform([cleaned_jd])

            rows = []
            for file in uploaded_files:
                raw_text = extract_text(file)
                cleaned = clean_text(raw_text)

                resume_vector = vectorizer.transform([cleaned])
                score = cosine_similarity(jd_vector, resume_vector).flatten()[0]

                predicted_category = predict_category(cleaned)

                rows.append({
                    "File Name": file.name,
                    "Predicted Category": predicted_category,
                    "Match Score": score
                })

            elapsed = time.time() - start_time

            results_df = pd.DataFrame(rows).sort_values(by="Match Score", ascending=False)

            st.success(f"Screened {len(uploaded_files)} uploaded resume(s) in {elapsed:.3f} seconds")
            st.dataframe(
                results_df.style.format({"Match Score": "{:.3f}"}),
                use_container_width=True
            )