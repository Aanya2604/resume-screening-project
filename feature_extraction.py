import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load cleaned resumes
df = pd.read_csv("data/cleaned_resumes.csv")
df["cleaned_resume"] = df["cleaned_resume"].fillna("")  # safety check

# Build TF-IDF vectors from all resumes
vectorizer = TfidfVectorizer(max_features=5000)  # cap vocabulary size for speed
resume_vectors = vectorizer.fit_transform(df["cleaned_resume"])

print("TF-IDF matrix shape:", resume_vectors.shape)  # (num_resumes, num_features)

# Save the vectorizer and vectors so matching.py can reuse them
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("resume_vectors.pkl", "wb") as f:
    pickle.dump(resume_vectors, f)

print("Saved vectorizer.pkl and resume_vectors.pkl")