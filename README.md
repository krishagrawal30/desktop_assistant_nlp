<div align="center">

```
██████╗  █████╗ ██╗    ██╗███╗   ██╗
██╔══██╗██╔══██╗██║    ██║████╗  ██║
██║  ██║███████║██║ █╗ ██║██╔██╗ ██║
██║  ██║██╔══██║██║███╗██║██║╚██╗██║
██████╔╝██║  ██║╚███╔███╔╝██║ ╚████║
╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝
```

### **Desktop Assistant & Web Navigator**
*An offline NLP-powered command interpreter — plain English in, real OS actions out.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold?style=for-the-badge)](LICENSE)
[![Offline](https://img.shields.io/badge/Works-100%25%20Offline-22c55e?style=for-the-badge)](.)

<br/>

> *"What if your computer just understood you?"*

</div>

---

## ⚡ What is DAWN?

DAWN is a **fully offline NLP desktop assistant** that converts plain English commands into real operating system actions — no internet, no cloud, no API keys required.

```
You type:   "rename report.pdf to final_report.pdf"
DAWN does:  renames the file on disk. instantly.

You type:   "search for machine learning on youtube"
DAWN does:  opens YouTube with that exact search.

You type:   "move notes.txt to Downloads"
DAWN does:  moves the file. no questions asked.
```

Under the hood: **TF-IDF vectorisation → LinearSVC intent classifier → Regex NER → OS execution**. Five stages. 16 intent classes. One pipeline.

---

## 📊 Performance

<div align="center">

| Metric | Score | Test Cases |
|:---|:---:|:---:|
| 🎯 Intent Classification Accuracy | **91.3%** | 80/20 split |
| 🔍 NER Exact Match | **84.3%** | 70 cases |
| ⚡ Full Pipeline (E2E) | **90%** | 48 cases |
| 📦 Intent Classes | **16** | — |
| 🧪 Total Test Cases | **118** | 3 layers |

</div>

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### 📂 File & Folder Operations
```
create notes.txt
open report.pdf
delete temp.txt
rename old.txt to new.txt
move report.pdf to Documents
copy notes.txt to Backup
search for budget.xlsx
select all pdf
```

</td>
<td width="50%">

### 🌐 Web Navigation
```
search for python tutorial on google
look for lofi music on youtube
find artificial intelligence on wikipedia
```

### 📦 Archive Operations
```
extract archive.zip
```

### 🔒 System Commands
```
shutdown the computer
restart the system
suspend
```

</td>
</tr>
</table>

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      USER COMMAND                           │
│         "rename report.pdf to final_report.pdf"             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  TF-IDF VECTORISER                          │
│           Converts text → sparse feature vector             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               INTENT CLASSIFIER (LinearSVC)                 │
│              Predicts →  RENAME_FILE  ✓                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   NER ENGINE (Regex)                        │
│   Extracts → old_name: report.pdf · new_name: final_report  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTOR                                │
│            os.rename(old_path, new_path)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
               ✅  File renamed successfully
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---|:---|:---|
| Language | `Python 3.10+` | Core runtime |
| Vectorisation | `TF-IDF (scikit-learn)` | Text → features |
| Classification | `LinearSVC · LogisticRegression` | Intent prediction |
| Hyperparameter tuning | `GridSearchCV · 5-fold CV` | Model selection |
| NER | `re (Python regex)` | Entity extraction |
| OS Execution | `os · shutil · zipfile · webbrowser` | System actions |
| Terminal UI | `rich` | Styled terminal output |
| Model persistence | `joblib` | Save / load models |
| Evaluation dashboard | `Streamlit` | Interactive UI |
| Charts | `Plotly` | Confusion matrix · F1 · radar |
| Data | `pandas` | Tabular results |

</div>

---

## ⚙️ Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/your-username/desktop-assistant-nlp.git
cd desktop-assistant-nlp
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is unavailable:
> ```bash
> pip install scikit-learn pandas numpy streamlit plotly joblib rich
> ```

### 4. Verify project structure

```
desktop-assistant-nlp/
├── data/
│   └── intent_dataset.json        ← Training data
├── models/                        ← Generated after training
│   ├── intent_model.pkl
│   └── vectorizer.pkl
├── src/
│   ├── main.py                    ← Entry point
│   ├── train.py                   ← Model training
│   ├── intent_classifier.py       ← TF-IDF + ML pipeline
│   ├── ner.py                     ← Regex NER engine
│   └── executor.py                ← OS-level actions
├── dashboard.py                   ← Streamlit evaluation dashboard
├── requirements.txt
└── README.md
```

### 5. Train the model

```bash
python src/train.py
```

This runs GridSearchCV across LogisticRegression and LinearSVC, selects the best model, and saves:
```
models/intent_model.pkl
models/vectorizer.pkl
```

### 6. Run the assistant

```bash
python src/main.py
```

Try these commands:
```
create notes.txt
rename notes.txt to summary.txt
move summary.txt to Downloads
search for python tutorial on google
look for lofi music on youtube
find neural networks on wikipedia
extract archive.zip
```

### 7. Launch the evaluation dashboard

```bash
streamlit run dashboard.py
```

The dashboard provides:
- Intent classification accuracy · confusion matrix · per-class F1
- NER exact-match accuracy per intent
- End-to-end pipeline evaluation
- LogisticRegression vs LinearSVC model comparison

---

## 🔍 Evaluation — 3 Layers

Most NLP projects report one accuracy number. DAWN reports three.

```
Layer 1 — Intent Classifier
  Tests the ML model in isolation.
  Metrics: accuracy · precision · recall · F1 · confusion matrix

Layer 2 — NER Engine
  Tests regex entity extraction in isolation.
  Metric: exact-match accuracy across 13 entity-bearing intent types

Layer 3 — End-to-End Pipeline
  Both intent AND entities must match ground truth.
  Metric: full pipeline pass rate (90% on 48 test cases)
```

---

## 🐛 Common Issues

<details>
<summary><b>FileNotFoundError: intent_model.pkl</b></summary>

You haven't trained the model yet. Run:
```bash
python src/train.py
```
</details>

<details>
<summary><b>Missing dependencies</b></summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Virtual environment won't activate on Windows PowerShell</b></summary>

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```
</details>

<details>
<summary><b>Command not recognised</b></summary>

The assistant handles 16 specific intent classes. Commands that don't match any class return "Command not recognised." Check the supported commands listed in the Features section above.
</details>

---

## 👨‍💻 Authors

<div align="center">

**Krish Agrawal** · **Prathmesh Agrawal**


---

<div align="center">

*Built with Python. Runs offline. Understands you.*

**⭐ Star this repo if you found it useful**

</div>