import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load everything we saved earlier
df = pd.read_csv("data/cleaned_resumes.csv")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("resume_vectors.pkl", "rb") as f:
    resume_vectors = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def rank_resumes(job_description, top_n=5):
    start_time = time.time()  # for measuring response time later

    cleaned_jd = clean_text(job_description)
    jd_vector = vectorizer.transform([cleaned_jd])

    scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    df["match_score"] = scores

    results = df.sort_values(by="match_score", ascending=False).head(top_n)

    elapsed = time.time() - start_time
    print(f"\nScreened {len(df)} resumes in {elapsed:.3f} seconds\n")

    return results[["ID", "Category", "match_score"]]

# ---- Test it with a sample job description ----
if __name__ == "__main__":
    sample_jd = """
    We are looking for an HR Administrator with experience in employee
    relations, recruitment, onboarding, and HR policy management.
    Strong communication and organizational skills required.
    """

    top_matches = rank_resumes(sample_jd, top_n=5)
    print(top_matches.to_string(index=False))