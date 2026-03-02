import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["intent"] for item in data]

    return texts, labels

from sklearn.model_selection import GridSearchCV


def train_models(texts, labels):
    vectorizer = TfidfVectorizer()
    X_full = vectorizer.fit_transform(texts)

    param_grid_logistic = {
        "C": [0.1, 1, 5, 10, 20, 50, 100]
    }

    param_grid_svm = {
        "C": [0.1, 1, 5, 10, 20, 50, 100]
    }

    # GridSearch for Logistic Regression
    logistic_grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid_logistic,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    logistic_grid.fit(X_full, labels)

    print("Best Logistic Accuracy:", logistic_grid.best_score_)
    print("Best Logistic Params:", logistic_grid.best_params_)

    # GridSearch for Linear SVM
    svm_grid = GridSearchCV(
        LinearSVC(max_iter=5000),
        param_grid_svm,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    svm_grid.fit(X_full, labels)

    print("Best SVM Accuracy:", svm_grid.best_score_)
    print("Best SVM Params:", svm_grid.best_params_)

    if logistic_grid.best_score_ > svm_grid.best_score_:
        best_model = logistic_grid.best_estimator_
        print("\nFinal Model Selected: Logistic Regression")
    else:
        best_model = svm_grid.best_estimator_
        print("\nFinal Model Selected: Linear SVM")

    # Retrain best model on FULL dataset
    best_model.fit(X_full, labels)

    print("Final model retrained on full dataset.")

    return best_model, vectorizer

def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(text, model, vectorizer):
    X = vectorizer.transform([text])
    return model.predict(X)[0]