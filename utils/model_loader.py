"""
model_loader.py — Flask-compatible (no streamlit dependency).
Uses logging instead of st.warning/st.error.
"""
import warnings, os, io, pickle, logging
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

log = logging.getLogger(__name__)

LABEL_MAP  = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI_MAP  = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
COLOR_MAP  = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}
BG_MAP     = {"Positive": "#ecfdf5", "Negative": "#fef2f2", "Neutral": "#fffbeb"}
BORDER_MAP = {"Positive": "#6ee7b7", "Negative": "#fca5a5", "Neutral": "#fcd34d"}
VALID_LABELS = {"Positive", "Negative", "Neutral"}


def _normalise_label(raw) -> str:
    s = str(raw).strip().title()
    if s in VALID_LABELS: return s
    INT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
    try: return INT_MAP.get(int(raw), "Neutral")
    except (ValueError, TypeError): pass
    for v in VALID_LABELS:
        if v.lower() == s.lower(): return v
    return "Neutral"


import torch

class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)
        return super().find_class(module, name)

def _cpu_pickle_load(path):
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _CPUUnpickler(f).load()

def _cpu_joblib_load(path):
    import joblib
    _orig = torch.load
    def _p(*a, **kw):
        kw["map_location"] = torch.device("cpu")
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _p
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = joblib.load(path)
    finally:
        torch.load = _orig
    return r

# Module-level cache
_svm_cache, _bilstm_cache, _mbert_cache = None, None, None

def load_svm():
    global _svm_cache
    if _svm_cache: return _svm_cache
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm   = joblib.load("models/svm_model.pkl")
            tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        _svm_cache = (svm, tfidf)
        log.info("SVM loaded.")
        return _svm_cache
    except Exception as e:
        log.warning(f"SVM load error: {e}")
        return None, None

def load_bilstm():
    global _bilstm_cache
    if _bilstm_cache: return _bilstm_cache
    try:
        from tensorflow.keras.models import load_model as klm
        model     = klm("models/bilstm_model.h5")
        tokenizer = _cpu_pickle_load("models/bilstm_tokenizer.pkl")
        encoder   = None
        for p in ["models/bilstm_label_encoder.pkl",
                  "models/bilstm_label_encoder (1).pkl",
                  "models/label_encoder.pkl"]:
            if os.path.exists(p):
                encoder = _cpu_pickle_load(p)
                break
        _bilstm_cache = (model, tokenizer, encoder)
        log.info("BiLSTM loaded.")
        return _bilstm_cache
    except Exception as e:
        log.warning(f"BiLSTM load error: {e}")
        return None, None, None

def load_mbert():
    global _mbert_cache
    if _mbert_cache: return _mbert_cache
    path = "models/mbert_model.pkl"
    if not os.path.exists(path):
        log.warning(f"mBERT not found at {path}")
        return None, None
    payload = None
    for loader in [_cpu_joblib_load, _cpu_pickle_load]:
        try: payload = loader(path); break
        except Exception as e: log.warning(f"mBERT loader: {e}")
    if payload is None: return None, None
    model = _extract_model(payload)
    if model is None: return None, None
    try: model = model.to("cpu"); model.eval()
    except Exception: pass
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    except Exception as e:
        log.warning(f"mBERT tokenizer: {e}")
    _mbert_cache = (model, tokenizer)
    log.info("mBERT loaded.")
    return _mbert_cache

def _extract_model(p):
    if p is None: return None
    if isinstance(p, dict):
        for k in ("model","classifier","clf","estimator"):
            if k in p and hasattr(p[k],"forward"): return p[k]
        for v in p.values():
            if hasattr(v,"forward"): return v
        return None
    if isinstance(p,(list,tuple)) and len(p)==2:
        a,b = p
        if hasattr(a,"forward"): return a
        if hasattr(b,"forward"): return b
    return p if hasattr(p,"forward") else None

def predict_svm(text, model, vec):
    if model is None or vec is None: return _demo(text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X     = vec.transform([text])
            raw   = model.predict(X)[0]
            label = _normalise_label(raw)
            if hasattr(model,"predict_proba"):
                prob    = model.predict_proba(X)[0].tolist()
                classes = [_normalise_label(c) for c in model.classes_]
                ordered = _reorder(prob, classes)
                conf    = max(ordered)
            else:
                ordered, conf = [0.1,0.1,0.8], 0.8
        return label, float(conf), ordered
    except Exception as e:
        log.warning(f"SVM predict error: {e}")
        return _demo(text)

def _reorder(probs, classes):
    target = ["Negative","Neutral","Positive"]
    m = {c:p for c,p in zip(classes,probs)}
    return [m.get(t,1/3) for t in target]

def predict_bilstm(text, model, tok, max_len=100, encoder=None):
    if model is None or tok is None: return _demo(text)
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq  = tok.texts_to_sequences([text])
        pad  = pad_sequences(seq,maxlen=max_len,padding="post",truncating="post")
        prob = model.predict(pad,verbose=0)[0]
        idx  = int(np.argmax(prob))
        if encoder is not None:
            try: label = _normalise_label(encoder.inverse_transform([idx])[0])
            except Exception: label = _normalise_label(idx)
        else: label = _normalise_label(idx)
        return label, float(np.max(prob)), prob.tolist()
    except Exception as e:
        log.warning(f"BiLSTM predict error: {e}")
        return _demo(text)

def predict_mbert(text, model, tok):
    if model is None: return _demo(text)
    if tok is None:   return _demo(text)
    try:
        import torch.nn.functional as F
        inp = tok(text,return_tensors="pt",truncation=True,max_length=128,padding=True)
        inp = {k:v.to("cpu") for k,v in inp.items()}
        model.eval()
        with torch.no_grad():
            out = model(**inp)
        prob  = F.softmax(out.logits,dim=-1).squeeze().cpu().numpy()
        idx   = int(np.argmax(prob))
        try:    label = _normalise_label(model.config.id2label[idx])
        except: label = _normalise_label(idx)
        return label, float(np.max(prob)), prob.tolist()
    except Exception as e:
        log.warning(f"mBERT predict error: {e}")
        return _demo(text)

def _demo(text):
    try:
        import importlib.util, sys
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p    = os.path.join(base,"utils","keyword_rules.py")
        spec = importlib.util.spec_from_file_location("keyword_rules",p)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["keyword_rules"] = mod
        spec.loader.exec_module(mod)
        result = mod.keyword_predict(text)
        if result:
            label,conf,_,_ = result
            scores = {"Positive":0.1,"Negative":0.1,"Neutral":0.1}
            scores[label] = conf
            return label, conf, list(scores.values())
    except Exception: pass
    return "Neutral", 0.60, [0.18,0.62,0.20]
