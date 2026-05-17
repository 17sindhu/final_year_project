# app.py

import warnings
warnings.filterwarnings("ignore")

import download_models

import os
import sys
import io

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file
)

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.preprocessing import (
    load_normalization_dict,
    preprocess
)

from utils.model_loader import (
    load_svm,
    load_mbert,
    predict_svm,
    predict_mbert,
)

from utils.keyword_rules import keyword_predict

from utils.aspect_analyzer import (
    analyse_aspects_single,
    analyse_aspects_bulk,
    generate_recommendation,
    generate_text_report,
)

from utils.database import (
    init_db,
    save as db_save,
    get_all,
    get_stats,
    delete_one,
    clear_all,
    export_csv,
)

# ─────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "rk_sentiment_secret"

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

init_db()

norm_dict = load_normalization_dict(
    os.path.join(ROOT, "dictionary.csv")
)

print("[INFO] Loading models...")

_svm_model, _svm_vec = load_svm()

_mbert_model, _mbert_tok = load_mbert()

print("[INFO] Models loaded successfully")

# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")


@app.route("/bulk")
def bulk():
    return render_template("bulk.html")


@app.route("/history")
def history():
    stats = get_stats()
    rows = get_all(200)

    return render_template(
        "history.html",
        stats=stats,
        rows=rows
    )


@app.route("/insights")
def insights():
    return render_template("insights.html")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def run_model(clean, model_name):

    if model_name == "mBERT":

        return predict_mbert(
            clean,
            _mbert_model,
            _mbert_tok
        )

    else:

        return predict_svm(
            clean,
            _svm_model,
            _svm_vec
        )


def smart_predict(clean, model_name):

    kw = keyword_predict(clean)

    if kw:

        label, conf, _, _ = kw

        remaining = round((1.0 - conf) / 2, 3)

        proba = [remaining, remaining, remaining]

        idx_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }

        proba[idx_map[label]] = round(conf, 3)

        return label, conf, proba, "keyword"

    else:

        label, conf, proba = run_model(
            clean,
            model_name
        )

        return label, conf, proba, "model"


def detect_review_column(df):

    hints = [
        "review",
        "text",
        "comment",
        "feedback",
        "description",
        "content",
        "message",
        "review_text"
    ]

    cols = {
        c.lower().strip(): c
        for c in df.columns
    }

    for h in hints:
        if h in cols:
            return cols[h]

    for c in df.columns:

        if df[c].dtype == object:

            avg = (
                df[c]
                .dropna()
                .astype(str)
                .str.len()
                .mean()
            )

            if avg and avg > 20:
                return c

    return None


# ─────────────────────────────────────────────
# API — Single Prediction
# ─────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    text = (data.get("text") or "").strip()

    model = (data.get("model") or "SVM").strip()

    if not text:

        return jsonify({
            "error": "No text provided"
        }), 400

    clean, tokens = preprocess(
        text,
        norm_dict
    )

    label, conf, proba, method = smart_predict(
        clean,
        model
    )

    aspect_result = analyse_aspects_single(clean)

    db_save(
        text,
        clean,
        model,
        method,
        label,
        conf,
        tokens
    )

    return jsonify({

        "label": label,

        "confidence": round(conf * 100, 1),

        "proba": {
            "Negative": round(proba[0] * 100, 1),
            "Neutral": round(proba[1] * 100, 1),
            "Positive": round(proba[2] * 100, 1),
        },

        "clean_text": clean,

        "tokens": tokens,

        "model": model,

        "method": method,

        "aspects": aspect_result["aspects"],

        "aspect_count": aspect_result["detected_count"]
    })


# ─────────────────────────────────────────────
# API — Bulk Prediction
# ─────────────────────────────────────────────

@app.route("/api/bulk", methods=["POST"])
def api_bulk():

    file = request.files.get("file")

    model = request.form.get("model", "SVM")

    column = request.form.get("column", "")

    if not file:

        return jsonify({
            "error": "No file uploaded"
        }), 400

    try:

        if file.filename.endswith(".csv"):

            df = pd.read_csv(file)

        else:

            df = pd.read_excel(file)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

    if df.empty:

        return jsonify({
            "error": "Empty file"
        }), 400

    if not column or column not in df.columns:

        column = detect_review_column(df)

    if not column:

        return jsonify({
            "error": "Review column not found"
        }), 400

    reviews = (
        df[column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cleaned = []
    labels = []
    confs = []

    for review in reviews:

        clean, tokens = preprocess(
            review,
            norm_dict
        )

        label, conf, _, method = smart_predict(
            clean,
            model
        )

        cleaned.append(clean)

        labels.append(label)

        confs.append(
            round(conf * 100, 2)
        )

    pos = labels.count("Positive")
    neg = labels.count("Negative")
    neu = labels.count("Neutral")

    aspect_summary = analyse_aspects_bulk(cleaned)

    recommendation = generate_recommendation(

        sentiment_counts={
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu
        },

        aspect_summary=aspect_summary,

        model_used=model,

        total_reviews=len(reviews)
    )

    out = df.copy()

    out["Cleaned Text"] = cleaned

    out["Predicted Sentiment"] = labels

    out["Confidence (%)"] = confs

    csv_buffer = io.StringIO()

    out.to_csv(
        csv_buffer,
        index=False
    )

    report_text = generate_text_report(
        recommendation,
        aspect_summary,
        file.filename
    )

    preview = out.head(10)[[
        column,
        "Predicted Sentiment",
        "Confidence (%)"
    ]].to_dict("records")

    return jsonify({

        "total": len(reviews),

        "positive": pos,

        "negative": neg,

        "neutral": neu,

        "verdict": recommendation["verdict"],

        "preview": preview,

        "csv": csv_buffer.getvalue(),

        "report": report_text,

        "aspect_summary": aspect_summary
    })


# ─────────────────────────────────────────────
# API — History
# ─────────────────────────────────────────────

@app.route("/api/history")
def api_history():

    return jsonify({
        "rows": get_all(500),
        "stats": get_stats()
    })


@app.route("/api/history/delete/<int:rid>", methods=["DELETE"])
def api_delete(rid):

    delete_one(rid)

    return jsonify({
        "success": True
    })


@app.route("/api/history/clear", methods=["POST"])
def api_clear():

    clear_all()

    return jsonify({
        "success": True
    })


@app.route("/api/history/export")
def api_export():

    buf = io.BytesIO(
        export_csv().encode("utf-8")
    )

    return send_file(
        buf,
        mimetype="text/csv",
        download_name="prediction_history.csv",
        as_attachment=True
    )


# ─────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )