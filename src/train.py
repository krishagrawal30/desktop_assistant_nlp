from intent_classifier import load_data, train_models, save_model

DATA_PATH = "data/intent_dataset.json"
MODEL_PATH = "models/intent_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

texts, labels = load_data(DATA_PATH)

print("Number of samples:", len(texts))

best_model, vectorizer = train_models(texts, labels)

save_model(best_model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

print("Training complete. Best model saved.")