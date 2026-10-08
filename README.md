# 🤖 AI-Based Resume Screening System

An AI-powered Resume Screening System that automatically analyzes and ranks resumes based on their relevance to a given job description using Machine Learning and Natural Language Processing (NLP).

The project helps recruiters reduce manual effort by comparing candidate resumes against job requirements and displaying the most relevant matches.

---

## 📌 Features

- Upload and analyze multiple resumes
- Compare resumes against a job description
- Automatic resume ranking based on similarity score
- NLP-based text preprocessing
- Machine Learning-powered resume matching
- Interactive web interface built with Streamlit
- Fast and easy-to-use recruiter dashboard

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Pickle
- TF-IDF Vectorization
- Cosine Similarity
- Machine Learning

---

## 📂 Project Structure

```
resume-screening-project/
│
├── app.py                         # Streamlit application
├── classifier.py                  # Resume classification logic
├── preprocessing.py               # Text preprocessing
├── feature_extraction.py          # Feature extraction
├── matching.py                    # Resume matching
├── compare_models.py              # Model comparison
├── download_nltk.py               # Downloads NLTK resources
│
├── classifier_model.pkl
├── classifier_vectorizer.pkl
├── resume_vectors.pkl
├── vectorizer.pkl
│
├── data/
│   ├── Resume.csv
│   └── cleaned_resumes.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 How It Works

1. Enter a Job Description.
2. Upload one or more resumes.
3. The system preprocesses the text using NLP techniques.
4. The job description and resumes are converted into numerical vectors using TF-IDF.
5. Cosine Similarity is calculated between the job description and each resume.
6. Resumes are ranked based on their similarity score.
7. Results are displayed through an interactive Streamlit dashboard.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Aanya2604/resume-screening-project.git
```

### Navigate to the project folder

```bash
cd resume-screening-project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Workflow

```
Job Description
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Cosine Similarity
        │
        ▼
Resume Ranking
        │
        ▼
Results on Streamlit Dashboard
```

---

## 🎯 Future Enhancements

- Resume parsing using OCR
- Support for PDF and DOCX files
- Deep Learning-based semantic matching using BERT
- Skill gap analysis
- Candidate recommendation system
- Recruiter analytics dashboard
- Cloud deployment

---

## 👩‍💻 Author

**Aanya JP**

B.Tech – Artificial Intelligence & Machine Learning

Delhi Technical Campus, Greater Noida

---
