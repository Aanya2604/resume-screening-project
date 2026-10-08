import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv("data/Resume.csv")

# Quick check — real row count and column names
print("Number of resumes:", len(df))
print("Columns:", df.columns.tolist())
print(df[["Resume_str", "Category"]].head(3))

# Set up cleaning tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()           # remove extra whitespace
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Apply cleaning to every resume
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# Save the cleaned version for later steps
df.to_csv("data/cleaned_resumes.csv", index=False)
print("\nCleaning done! Saved to data/cleaned_resumes.csv")
print("\nExample before/after:")
print("BEFORE:", df["Resume_str"].iloc[0][:200])
print("AFTER:", df["cleaned_resume"].iloc[0][:200])