"""
app.py — Flask backend for RK Sentiment Analysis
Run: python app.py
"""

import os, sys, json, io, csv
from datetime import datetime
from flask import (
    Flask, render_template, request,
    jsonify, send_file, redirect, url_for
)
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.preprocessing  import load_normalization_dict, preprocess
from utils.model_loader   import (
    load_svm, load_bilstm, load_mbert,
    predict_svm, predict_bilstm, predict_mbert,
)
from utils.keyword_rules  import keyword_predict
from utils.aspect_analyzer import (
    analyse_aspects_single, analyse_aspects_bulk,
    generate_recommendation, generate_text_report,
    ASPECT_ICONS, ASPECT_COLORS,
)
from utils.database import (
    init_db, save as db_save, get_all, get_stats,
    delete_one, clear_all, export_csv, DB_PATH
)

app = Flask(__name__)
app.secret_key = "rk_sentiment_2024"

# ── Startup ───────────────────────────────────────────────────────────────────
init_db()
norm_dict = load_normalization_dict(os.path.join(ROOT, "dictionary.csv"))

print("[INFO] Loading models...")
_svm_model,    _svm_vec                  = load_svm()
_bilstm_model, _bilstm_tok, _bilstm_enc  = load_bilstm()
_mbert_model,  _mbert_tok                = load_mbert()
print("[INFO] Models loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")

@app.route("/bulk")
def bulk():
    return render_template("bulk.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/history")
def history():
    stats = get_stats()
    rows  = get_all(200)
    return render_template("history.html", stats=stats, rows=rows)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_full(clean, model):
    if model == "SVM":
        return predict_svm(clean, _svm_model, _svm_vec)
    elif model == "BiLSTM":
        return predict_bilstm(clean, _bilstm_model, _bilstm_tok, encoder=_bilstm_enc)
    else:
        return predict_mbert(clean, _mbert_model, _mbert_tok)


def _run_proba(clean, model):
    _, _, proba = _run_full(clean, model)
    return proba or [0.1, 0.1, 0.8]


def _smart_predict(clean, model):
    """
    Keyword engine FIRST — if strong sentiment keywords found, use them.
    Falls back to selected model only if no keywords detected.
    Keyword engine always wins over model for clear cases.
    """
    kw = keyword_predict(clean)
    if kw:
        label, conf, _, _ = kw
        # Still run model to get probability distribution for charts
        try:
            _, _, proba = _run_full(clean, model)
        except Exception:
            proba = [0.1, 0.1, 0.8]
        method = "keyword"
        return label, conf, proba, method
    else:
        label, conf, proba = _run_full(clean, model)
        return label, conf, proba, "model"


def _detect_col(df):
    hints = ["review", "text", "comment", "feedback", "description",
             "opinion", "content", "message", "review_text"]
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for h in hints:
        if h in cols_lower:
            return cols_lower[h]
    for c in df.columns:
        if df[c].dtype == object:
            avg = df[c].dropna().astype(str).str.len().mean()
            if avg and avg > 20:
                return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# API — Single prediction + aspect analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data  = request.get_json()
    text  = (data.get("text") or "").strip()
    model = (data.get("model") or "SVM").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean, tokens = preprocess(text, norm_dict)
    label, confidence, proba, method = _smart_predict(clean, model)

    # Aspect analysis on the single review
    aspect_result = analyse_aspects_single(clean)

    db_save(text, clean, model, method, label, confidence, tokens)

    return jsonify({
        "label":      label,
        "confidence": round(confidence * 100, 1),
        "proba": {
            "Negative": round(proba[0] * 100, 1),
            "Neutral":  round(proba[1] * 100, 1),
            "Positive": round(proba[2] * 100, 1),
        },
        "clean_text":  clean,
        "tokens":      tokens,
        "model":       model,
        "method":      method,
        "aspects":     aspect_result["aspects"],
        "aspect_count": aspect_result["detected_count"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Bulk prediction + aspect analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/bulk", methods=["POST"])
def api_bulk():
    f     = request.files.get("file")
    model = request.form.get("model", "SVM")
    col   = request.form.get("column", "")

    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        if f.filename.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if df.empty:
        return jsonify({"error": "File is empty"}), 400

    fname = f.filename

    if not col or col not in df.columns:
        col = _detect_col(df)
    if not col:
        return jsonify({"error": "Could not detect review column"}), 400

    reviews = df[col].fillna("").astype(str).tolist()
    total   = len(reviews)

    cleaned, labels, confs, methods = [], [], [], []

    for rev in reviews:
        clean, tokens = preprocess(rev, norm_dict)
        label, conf, _, method = _smart_predict(clean, model)
        cleaned.append(clean)
        labels.append(label)
        confs.append(round(conf * 100, 2))
        methods.append(method)

    # Sentiment counts
    pos = labels.count("Positive")
    neg = labels.count("Negative")
    neu = labels.count("Neutral")

    # Aspect analysis on all cleaned reviews
    aspect_summary = analyse_aspects_bulk(cleaned)

    # Recommendation
    rec = generate_recommendation(
        sentiment_counts={"Positive": pos, "Negative": neg, "Neutral": neu},
        aspect_summary=aspect_summary,
        model_used=model,
        total_reviews=total,
    )

    # Result dataframe
    df_out = df.copy()
    df_out["Cleaned Text"]        = cleaned
    df_out["Predicted Sentiment"] = labels
    df_out["Confidence (%)"]      = confs

    csv_buf = io.StringIO()
    df_out.to_csv(csv_buf, index=False)
    csv_str = csv_buf.getvalue()

    # Text report
    report_txt = generate_text_report(rec, aspect_summary, fname)

    # Preview
    preview = df_out.head(10)[[col, "Predicted Sentiment", "Confidence (%)"]].to_dict("records")

    # Serialize aspect_summary (remove non-JSON fields for response)
    asp_json = {
        k: {
            "positive": v["positive"], "negative": v["negative"],
            "mixed":    v["mixed"],    "total":    v["total"],
            "pos_pct":  v["pos_pct"],  "neg_pct":  v["neg_pct"],
            "overall":  v["overall"],  "icon":     v["icon"],
            "color":    v["color"],
        }
        for k, v in aspect_summary.items()
    }

    return jsonify({
        "total":          total,
        "pos":            pos, "neg": neg, "neu": neu,
        "pos_pct":        rec["pos_pct"],
        "neg_pct":        rec["neg_pct"],
        "neu_pct":        rec["neu_pct"],
        "verdict":        rec["verdict"],
        "verdict_color":  rec["verdict_color"],
        "verdict_bg":     rec["verdict_bg"],
        "model":          model,
        "col_used":       col,
        "preview":        preview,
        "csv":            csv_str,
        "report":         report_txt,
        "aspect_summary": asp_json,
        "recommendation": {
            "top_positive":    [[a, d["pos_pct"]] for a, d in rec["top_positive"]],
            "top_negative":    [[a, d["neg_pct"]] for a, d in rec["top_negative"]],
            "most_praised":    rec["most_praised"],
            "most_complained": rec["most_complained"],
            "reasons":         rec["reasons"],
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Single review aspect analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/aspects", methods=["POST"])
def api_aspects():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    clean, _ = preprocess(text, norm_dict)
    result   = analyse_aspects_single(clean)
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# API — History
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    return jsonify({"rows": get_all(500), "stats": get_stats()})

@app.route("/api/history/delete/<int:rid>", methods=["DELETE"])
def api_delete(rid):
    delete_one(rid)
    return jsonify({"ok": True})

@app.route("/api/history/clear", methods=["POST"])
def api_clear():
    clear_all()
    return jsonify({"ok": True})

@app.route("/api/history/export")
def api_export():
    buf = io.BytesIO(export_csv().encode("utf-8"))
    return send_file(buf, mimetype="text/csv",
                     download_name="prediction_history.csv",
                     as_attachment=True)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
