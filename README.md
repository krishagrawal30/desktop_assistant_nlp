# 🌅 DAWN — Desktop Assistant and Web Navigator

DAWN (**Desktop Assistant and Web Navigator**) is a fully offline **NLP-based desktop assistant** that understands natural language commands and performs **real OS-level actions** without requiring internet, APIs, or cloud services.

The assistant converts user commands into executable desktop actions using **Natural Language Processing (NLP)**, **Intent Classification**, and **Named Entity Recognition (NER)**.

---

## 🚀 Features

### 📂 File & Folder Operations
- Create files
- Open files
- Delete files
- Rename files
- Search files
- Move files
- Copy files
- Select multiple files by extension

### 📦 Archive Operations
- Extract ZIP files

### 🌐 Web Navigation
- Search on Google
- Search on YouTube
- Search on Wikipedia

### 🧠 NLP Capabilities
- Intent Classification using **TF-IDF + Machine Learning**
- Named Entity Recognition using **Regex-based extraction**
- Natural language understanding for desktop commands

### 🔒 Offline Functionality
- No cloud APIs
- No internet dependency
- Fully local execution
- Privacy-friendly architecture

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning / NLP
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Linear SVM

### System Libraries
- OS
- Shutil
- Zipfile
- Webbrowser
- Regex (re)

### Dashboard / Evaluation
- Streamlit
- Plotly

---

## 🧠 How It Works

DAWN follows a multi-stage NLP pipeline:

```text
User Command
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Intent Classification
(Logistic Regression / SVM)
      ↓
Named Entity Recognition (NER)
      ↓
Executor Layer
(OS-level operations)
      ↓
Action Executed

---
## ⚙️ Setup Guide (For New Users)

Follow these steps to run the project locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/desktop-assistant-nlp.git
cd desktop-assistant-nlp
```

---

### 2️⃣ Create a Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / MacOS

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install scikit-learn pandas numpy streamlit plotly joblib
```

---

### 4️⃣ Verify Project Structure

Ensure your folder structure looks like this:

```text
desktop-assistant-nlp/
│── data/
│   └── intent_dataset.json
│
│── models/
│
│── src/
│   ├── main.py
│   ├── train.py
│   ├── intent_classifier.py
│   ├── ner.py
│   ├── executor.py
│
│── dashboard.py
│── requirements.txt
│── README.md
```

---

### 5️⃣ Train the NLP Model

Before using the assistant, train the model:

```bash
python src/train.py
```

This generates:

```text
models/
│── intent_model.pkl
│── vectorizer.pkl
```

---

### 6️⃣ Run the Desktop Assistant

Start the assistant:

```bash
python src/main.py
```

Example commands:

```text
create notes.txt
open report.pdf
delete temp.txt
rename old.txt to new.txt
move report.pdf to documents
copy notes.txt to backup
search python tutorial on google
```

---

### 7️⃣ Run Evaluation Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides:

- NLP model accuracy
- NER evaluation
- Intent-wise metrics
- End-to-end pipeline evaluation
- Confusion matrix
- Model comparison

---

### 8️⃣ Common Issues

#### Model Not Found Error

If you get:

```text
FileNotFoundError: intent_model.pkl
```

Run:

```bash
python src/train.py
```

---

#### Missing Dependencies

Run:

```bash
pip install -r requirements.txt
```

or

```bash
pip install scikit-learn streamlit plotly pandas numpy
```

---

#### Virtual Environment Not Activating

For Windows:

```bash
venv\Scripts\activate
```

If PowerShell blocks execution:

```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

---