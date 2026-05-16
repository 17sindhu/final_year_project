# Sentiment Analysis of Romanized Kannada Product Reviews
### Flask Web Application

JSS Science & Technology University, Mysuru  
Dept. of Computer Science & Engineering — Final Year Project 2024–25

---

## Folder Structure

```
sentiment_analysis_flask/
│
├── app.py                        ← Flask entry point  →  python app.py
│
├── templates/
│   ├── base.html                 ← Shared sidebar layout
│   ├── index.html                ← Homepage / portfolio
│   ├── analyzer.html             ← Single review prediction
│   ├── bulk.html                 ← CSV/Excel bulk analysis
│   └── history.html              ← Prediction history dashboard
│
├── static/
│   ├── css/style.css             ← Full stylesheet
│   └── js/main.js                ← Shared JS utilities
│
├── utils/
│   ├── preprocessing.py          ← Text cleaning pipeline
│   ├── model_loader.py           ← SVM / BiLSTM / mBERT loaders
│   ├── keyword_rules.py          ← Keyword pre-prediction engine
│   └── database.py               ← SQLite history store
│
├── models/                       ← Place your trained model files here
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── bilstm_model.h5
│   ├── bilstm_tokenizer.pkl
│   ├── bilstm_label_encoder (1).pkl
│   └── mbert_model.pkl
│
├── dictionary.csv                ← Normalization dictionary
├── history.db                    ← Auto-created on first run
└── requirements.txt
```

---

## Setup Instructions

### Step 1 — Create virtual environment

```bash
cd C:\Users\sindhu\Desktop\sentiment_analysis_flask

python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you already have a working myenv from the Streamlit version,
> you only need to add Flask:
> ```bash
> pip install flask
> ```

### Step 3 — Add your model files

Copy these files into the `models/` folder:

| File | Description |
|------|-------------|
| `svm_model.pkl` | Trained SVM classifier |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer |
| `bilstm_model.h5` | BiLSTM Keras model |
| `bilstm_tokenizer.pkl` | BiLSTM tokenizer |
| `bilstm_label_encoder (1).pkl` | Label encoder |
| `mbert_model.pkl` | Fine-tuned mBERT |

Also place `dictionary.csv` in the root folder.

### Step 4 — Run the application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | Home | Academic portfolio — abstract, dataset, models, team |
| `/analyzer` | Sentiment Analyzer | Single review prediction with confidence charts |
| `/bulk` | Bulk Analysis | Upload CSV/Excel, predict all rows, download results |
| `/history` | Prediction History | Browse, filter, export all past predictions |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Single review prediction |
| POST | `/api/bulk` | Bulk file analysis |
| GET | `/api/history` | Get all history records |
| DELETE | `/api/history/delete/<id>` | Delete one record |
| POST | `/api/history/clear` | Clear all records |
| GET | `/api/history/export` | Download history CSV |

### Example API call (predict):

```python
import requests

response = requests.post("http://localhost:5000/api/predict", json={
    "text":  "product tumba chennagide delivery fast bantu",
    "model": "SVM"   # or "BiLSTM" or "mBERT"
})

print(response.json())
# {
#   "label": "Positive",
#   "confidence": 92.3,
#   "proba": {"Negative": 2.1, "Neutral": 5.6, "Positive": 92.3},
#   "clean_text": "product tumba chennagide delivery fast bantu",
#   "tokens": ["product", "tumba", "chennagide", "delivery", "fast"],
#   "model": "SVM",
#   "method": "keyword"
# }
```

---

## Difference from Streamlit Version

| Feature | Streamlit | Flask |
|---------|-----------|-------|
| Frontend | Python widgets | HTML + CSS + JS |
| Styling | Streamlit CSS | Custom Inter CSS |
| Charts | Plotly | Chart.js |
| API | None | REST JSON API |
| Deployment | `streamlit run` | `python app.py` |
| Port | 8501 | 5000 |

All backend logic (preprocessing, models, database) is identical.

---

## Important Notes

1. **sklearn version** — Models were saved with sklearn 1.6.1. Run:
   ```bash
   pip install scikit-learn==1.6.1
   ```

2. **mBERT on CPU** — The app forces CPU loading automatically.
   No GPU required. First load takes ~30 seconds.

3. **mBERT tokenizer** — Downloaded from HuggingFace on first run
   (~500KB, cached locally after that).

4. **history.db** — Created automatically in the project root
   on first prediction.

---

## Run in Production (optional)

```bash
pip install gunicorn

gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

*JSS S&T University, Mysuru — CS & Engineering — 2024–25*
