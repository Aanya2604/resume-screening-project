import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load cleaned resumes
df = pd.read_csv("data/cleaned_resumes.csv")
df["cleaned_resume"] = df["cleaned_resume"].fillna("")

print("Total resumes:", len(df))
print("Number of categories:", df["Category"].nunique())
print("\nResumes per category:")
print(df["Category"].value_counts())

# ---- Split into train/test sets ----
# 80% for training, 20% held out to test how well the model generalizes
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_resume"], df["Category"],
    test_size=0.2, random_state=42, stratify=df["Category"]
)

print(f"\nTraining on {len(X_train)} resumes, testing on {len(X_test)} resumes")

# ---- Build a FRESH TF-IDF vectorizer just for classification ----
# (trained only on the training set, to avoid leaking test data)
clf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = clf_vectorizer.fit_transform(X_train)
X_test_vec = clf_vectorizer.transform(X_test)

from sklearn.svm import LinearSVC

def evaluate_model(name, model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return y_pred, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ---- Model 1: Logistic Regression (balanced) ----
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_vec, y_train)
lr_pred, lr_scores = evaluate_model("Logistic Regression (balanced)", lr_model, X_test_vec, y_test)

# ---- Model 2: Linear SVM (balanced) ----
svm_model = LinearSVC(class_weight="balanced", max_iter=5000)
svm_model.fit(X_train_vec, y_train)
svm_pred, svm_scores = evaluate_model("Linear SVM (balanced)", svm_model, X_test_vec, y_test)

# ---- Pick the better model to save ----
best_name, best_model, best_pred = (
    ("Logistic Regression", lr_model, lr_pred) if lr_scores["f1"] >= svm_scores["f1"]
    else ("Linear SVM", svm_model, svm_pred)
)

print(f"\n>>> Best model: {best_name} (higher F1-score) — saving this one <<<")
print("\n===== Per-category breakdown (best model) =====")
print(classification_report(y_test, best_pred, zero_division=0))

# ---- Save the best model + its vectorizer ----
with open("classifier_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("classifier_vectorizer.pkl", "wb") as f:
    pickle.dump(clf_vectorizer, f)

print("Saved classifier_model.pkl and classifier_vectorizer.pkl")