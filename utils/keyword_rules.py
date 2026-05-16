"""
keyword_rules.py
─────────────────────────────────────────────────────────────
Fast keyword-based pre-prediction for Romanized Kannada.

FIX: Now properly checks BOTH single-token AND multi-word
phrase keywords against the full text string (not just tokens).
This ensures "ketta product", "late aythu", "damage aagidhe"
etc. are correctly detected.
"""

# ── POSITIVE keywords ─────────────────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    # Strong multi-word phrases (checked first)
    "tumba chennagide", "bahala chennagide", "tumba chenna",
    "must buy", "super product", "very good product",
    "fast delivery", "time ge bantu", "value for money",
    "loved it", "tumba happy", "bahala happy",
    "quality superb", "quality super", "quality chenna",
    "price reasonable", "good packaging", "well packed",
    "quick delivery", "early delivery", "fast bantu",
    "ishtavagide", "khushiyagide",
    # Single words
    "chennagide", "chenagide", "chennagidhe", "chennaagide",
    "chenna", "chenaagi", "chennagi",
    "superb", "super", "excellent", "perfect",
    "amazing", "wonderful", "awesome", "outstanding",
    "happy", "satisfied", "satisfaction",
    "original", "genuine", "durable", "sturdy",
    "worth", "premium", "recommended", "recommend",
    "wow", "fantastic", "fabulous",
    "good", "nice", "best",
]

# ── NEGATIVE keywords ─────────────────────────────────────────────────────────
NEGATIVE_KEYWORDS = [
    # Strong multi-word phrases
    "waste of money", "money waste", "never buy", "dont buy",
    "return maadtini", "refund beku", "very bad", "very worst",
    "damage aagidhe", "not working", "work agalla",
    "late aythu", "late agidhe", "late aagidhe",
    "tade aagide", "tade aythu", "bahala late", "tumba late",
    "delivery late", "delivery tade", "delivery delay",
    "ketta product", "ketta quality", "ketta agide",
    "sari illa", "chennagilla", "sariilla",
    "packaging damage", "box damage", "broken packaging",
    "size sari illa", "colour different", "color different",
    "photo different", "as shown alla", "not original",
    "cheating", "fraud", "fake product",
    # Single words
    "ketta", "kettadu", "kettadagide",
    "waste", "bad", "worst", "poor",
    "horrible", "terrible", "pathetic",
    "broken", "damaged", "damage",
    "defective", "defect", "fake", "duplicate",
    "useless", "disappointed", "disappointment",
    "delay", "delayed",
    "expensive", "costly",
    "fraud", "cheat",
    "bejaar",
]

# ── NEUTRAL keywords ──────────────────────────────────────────────────────────
NEUTRAL_KEYWORDS = [
    # Multi-word
    "ok ok", "so so", "not bad", "hage ide",
    "nothing special", "average product", "product ok",
    "packaging ok", "delivery ok", "price ok",
    "swalpa ok", "not bad not good",
    "as described", "as expected", "as shown",
    # Single words
    "parvagilla", "okay", "average",
    "normal", "decent", "moderate",
    "swalpa", "received", "delivered",
]

# ── Strong phrase sets for confidence boosting ────────────────────────────────
STRONG_POSITIVE_PHRASES = {
    "tumba chennagide", "bahala chennagide", "tumba chenna",
    "must buy", "fast delivery", "time ge bantu", "value for money",
    "loved it", "ishtavagide", "wow", "quality superb",
}
STRONG_NEGATIVE_PHRASES = {
    "waste of money", "money waste", "never buy", "dont buy",
    "return maadtini", "refund beku", "damage aagidhe",
    "late aythu", "tade aagide", "ketta product", "sari illa",
    "chennagilla", "not working", "cheating", "fraud",
}
STRONG_NEUTRAL_PHRASES = {
    "parvagilla", "ok ok", "so so", "hage ide", "average product",
}


def keyword_predict(text: str):
    """
    Fast keyword prediction on raw or preprocessed text.

    Checks full text for multi-word phrases AND single token matches.

    Returns:
        (label, confidence, "keyword", matched_dict)  — if keywords found
        None  — if no keywords detected → fall back to ML model
    """
    t = " " + text.lower().strip() + " "

    # ── Score each sentiment ───────────────────────────────────────────────
    pos_score = _score(t, POSITIVE_KEYWORDS, STRONG_POSITIVE_PHRASES)
    neg_score = _score(t, NEGATIVE_KEYWORDS, STRONG_NEGATIVE_PHRASES)
    neu_score = _score(t, NEUTRAL_KEYWORDS,  STRONG_NEUTRAL_PHRASES)

    total = pos_score + neg_score + neu_score
    if total == 0:
        return None   # no keywords at all → use model

    scores = {"Positive": pos_score, "Negative": neg_score, "Neutral": neu_score}
    winner = max(scores, key=scores.get)
    vals   = sorted(scores.values(), reverse=True)

    # Reject if tie between top two with very low total evidence
    if vals[0] == vals[1] and total < 3:
        return None

    # ── Confidence ────────────────────────────────────────────────────────
    raw_conf   = scores[winner] / max(total, 1)
    confidence = round(0.72 + raw_conf * 0.24, 3)
    confidence = min(confidence, 0.96)

    # ── Collect matched keywords per class ────────────────────────────────
    matched = {
        "Positive": _find_matches(t, POSITIVE_KEYWORDS),
        "Negative": _find_matches(t, NEGATIVE_KEYWORDS),
        "Neutral":  _find_matches(t, NEUTRAL_KEYWORDS),
    }

    return winner, confidence, "keyword", matched


def _score(text: str, keyword_list: list, strong_set: set) -> float:
    """
    Score a text against a keyword list.
    Strong phrases count double. Multi-word matched before single words.
    """
    score    = 0.0
    matched  = set()

    # Sort by length descending so longer phrases matched first
    for kw in sorted(keyword_list, key=len, reverse=True):
        kw_l = kw.lower()
        # Check with word boundary (space padding handles edges)
        if (" " + kw_l + " ") in text or \
           text.startswith(kw_l + " ") or \
           text.endswith(" " + kw_l):
            # Avoid double-counting sub-phrases
            if not any(kw_l in m for m in matched):
                matched.add(kw_l)
                score += 2.0 if kw_l in strong_set else 1.0

    return score


def _find_matches(text: str, keyword_list: list) -> list:
    """Return list of keywords that matched in the text (top 5)."""
    hits = []
    for kw in sorted(keyword_list, key=len, reverse=True):
        kw_l = kw.lower()
        if (" " + kw_l + " ") in text or \
           text.startswith(kw_l + " ") or \
           text.endswith(" " + kw_l):
            hits.append(kw)
            if len(hits) >= 5:
                break
    return hits
