# SmartHire: An Explainable AI-Based Resume Screening and Candidate Recommendation System

SmartHire ranks resumes against a job description and shows **why** each candidate got their score. Every recommendation comes with its score breakdown and lists of matched and missing skills, so a recruiter can check the result instead of trusting a black-box number.

**Live demo:** https://resume-screening-project-nve8tqnbubttutfpdy8ddv.streamlit.app/

B.Tech Minor Project (AIML 451), Department of Artificial Intelligence, Delhi Technical Campus, Greater Noida (affiliated to GGSIPU, New Delhi).

---

## Features

The Streamlit app has nine pages:

| Page | What it does |
|---|---|
| Dashboard | Overview of the resume database and system status |
| Job Analysis | Detects the job domain, role, experience level and required skills from a job description |
| Resume Parsing & Upload | Reads PDF and DOCX resumes, detects skills, flags duplicate uploads (MD5 hash) and gives a quality score out of 100 with suggestions |
| Candidate Matching & Explainable AI | Ranks the database for a job description and returns the top 1–10 candidates with their score breakdown and matched/missing skills |
| Semantic Candidate Search | Free-text search that returns the 10 closest profiles |
| Candidate Comparison Matrix | Compares selected candidates side by side |
| Recruitment Pipeline | Moves candidates through Uploaded → Screened → Shortlisted → Interview → Selected / Rejected |
| Resume Database | Browse the candidate profiles |
| About Project | Project details |

---

## How the score works

Each candidate gets one overall score from five named parts:

```
Overall = 0.40 × Skill coverage
        + 0.20 × TF-IDF cosine similarity
        + 0.15 × Education
        + 0.15 × Experience
        + 0.10 × Projects
```

- **Skill coverage** = required job skills found in the resume ÷ required job skills, using a 64-skill taxonomy with boundary-aware matching (so "Java" is not found inside "JavaScript").
- The weights add up to 1 and the score is capped at 99.
- Because the score is a plain weighted sum, the parts add up exactly to the score shown. No post-hoc approximation (such as SHAP or LIME) is needed to explain it.

A Linear SVM classifier also sorts resumes into 24 job categories.

---

## Results

**Category classifier** (497 held-out resumes, 24 categories, weighted averages):

| Metric | Logistic Regression | Linear SVM (selected) |
|---|---|---|
| Accuracy | 0.6499 | 0.7022 |
| Precision | 0.6581 | 0.7108 |
| Recall | 0.6499 | 0.7022 |
| F1-score | 0.6327 | 0.6971 |

**Matching engine** (2,500-profile demo database, one job description per domain):

- Correct domain detected for all 4 test job descriptions
- Precision@10, @50 and @100 = 1.00
- ROC-AUC between 0.986 and 1.000
- About 98 ms on average to rank all 2,500 profiles on a laptop CPU

Note: the demo database is generated with a fixed seed, so these matching results are a correctness check on controlled data, not an estimate of accuracy on real resumes.

---

## Tech stack

- Python 3.11
- Streamlit (web app, hosted on Streamlit Community Cloud)
- scikit-learn (TF-IDF, cosine similarity, Logistic Regression, LinearSVC)
- NLTK (stop words, WordNet lemmatiser)
- PyPDF2 and docx2txt (resume text extraction)
- pandas, NumPy, Matplotlib
- pickle and hashlib (saved models, MD5 duplicate detection)

---

## Project structure

```
resume-screening-project/
├── app.py                       # Streamlit application (9 pages)
├── preprocessing.py             # Cleans the raw resumes -> data/cleaned_resumes.csv
├── feature_extraction.py        # Fits the 5,000-term TF-IDF vectoriser
├── classifier.py                # Trains Logistic Regression and Linear SVM, saves the better one
├── compare_models.py            # Draws the classifier comparison chart
├── matching.py                  # Tests ranking from the command line
├── download_nltk.py             # Downloads the NLTK data used for cleaning
├── vectorizer.pkl               # Saved TF-IDF vectoriser (matching)
├── classifier_model.pkl         # Saved Linear SVM classifier
├── classifier_vectorizer.pkl    # Saved TF-IDF vectoriser (classifier)
├── classifier_comparison_chart.png
├── .streamlit/config.toml       # App theme
├── requirements.txt
└── runtime.txt
```

---

## Run it locally

```bash
git clone https://github.com/Aanya2604/resume-screening-project.git
cd resume-screening-project
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python download_nltk.py
streamlit run app.py
```

The saved `.pkl` models are included, so the app runs without retraining.

### Retrain the models (optional)

1. Download the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and save `Resume.csv` in a `data/` folder.
2. Run, in order:

```bash
python preprocessing.py
python feature_extraction.py
python classifier.py
python compare_models.py
```

---

## Limitations

- The TF-IDF vocabulary comes from the Kaggle data, so many modern tech terms (for example PyTorch, Docker, React) are unknown to it. The skill-coverage part makes up for this, but synonyms outside the taxonomy are still missed.
- Education, experience and project scores use simple keyword rules.
- Scanned (image-only) PDFs are not read, because there is no OCR.
- Uploads and pipeline stages last only for the browser session.

## Future work

- Sentence-BERT embeddings for semantic matching, keeping the score explainable
- Automatic growth of the skill taxonomy
- Structured parsing of degrees, dates and employers, plus OCR for scanned resumes
- Evaluation on real resumes with recruiter labels, and a fairness audit
- Persistent database and user login

---

## Author

**Aanya JP** (03618011623)
B.Tech, Artificial Intelligence and Machine Learning, Delhi Technical Campus, Greater Noida
Guide: Dr Preeti Raj Verma, Assistant Professor
