import os
from intent_classifier import load_model, predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


def main():
    model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)

    print("Desktop Assistant NLP\n")

    user_input = input("Enter command: ")

    intent = predict(user_input, model, vectorizer)

    print(f"Predicted Intent: {intent}")


if __name__ == "__main__":
    main()