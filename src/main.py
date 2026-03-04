import joblib
from ner import extract_entities
model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Desktop Assistant NLP")
print()
text = input("Enter command: ")
X = vectorizer.transform([text])
intent = model.predict(X)[0]
print("Predicted Intent:", intent)
entities = extract_entities(text, intent)
print("Extracted Entities:", entities)