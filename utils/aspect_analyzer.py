"""
utils/aspect_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Aspect-Based Sentiment Analysis for Romanized Kannada Product Reviews.

Identifies 9 product aspects and their sentiment from review text using
keyword mapping. Works for single reviews and bulk datasets.
No ML model changes required — pure keyword + rule-based engine.
─────────────────────────────────────────────────────────────────────────────
"""

import re
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# ASPECT KEYWORD DICTIONARY
# Each aspect has positive and negative Romanized Kannada + English keywords
# ─────────────────────────────────────────────────────────────────────────────

ASPECT_KEYWORDS = {

    "Quality": {
        "positive": [
            "chennagide", "chennaagide", "chenna", "super quality", "good quality",
            "high quality", "top quality", "best quality", "superb", "super",
            "excellent", "perfect", "original", "genuine", "durable", "strong",
            "sturdy", "long lasting", "tumba chenna", "bahala chenna",
            "quality tumba", "quality superb", "build quality", "quality good",
            "well made", "nicely made", "ishtavagide", "satisfied", "khushiyagide",
        ],
        "negative": [
            "ketta", "kettadu", "kettadagide", "poor quality", "bad quality",
            "low quality", "cheap quality", "worst quality", "broken", "damage",
            "damaged", "defective", "defect", "not working", "work agalla",
            "fake", "duplicate", "not original", "not genuine", "useless",
            "waste", "pathetic", "horrible", "terrible", "torn", "cracked",
            "quality illa", "quality ketta", "cheap", "sari illa",
        ],
    },

    "Delivery": {
        "positive": [
            "fast delivery", "quick delivery", "fast aagidhe", "quick bantu",
            "time ge bantu", "early delivery", "fast bantu", "jaldi bantu",
            "time ge barthide", "fast aythu", "quick aythu", "on time",
            "speedy delivery", "same day", "next day", "early bantu",
            "delivery fast", "delivery super", "delivery chenna",
        ],
        "negative": [
            "late aythu", "late agidhe", "late aagidhe", "late bantu",
            "delay", "delayed", "tade aagide", "tade aythu", "late delivery",
            "bahala late", "tumba late", "deliver agilla", "not delivered",
            "delivery late", "delivery tade", "delivery delay", "slow delivery",
            "bahala time", "time agilla", "week aythu", "month aythu",
        ],
    },

    "Packaging": {
        "positive": [
            "good packaging", "good packing", "safe packing", "safe packaging",
            "nice packing", "well packed", "properly packed", "secure packing",
            "packaging chenna", "packing super", "packaging good", "box chenna",
            "nice box", "good box", "protective packaging", "sealed properly",
        ],
        "negative": [
            "damaged package", "damaged packaging", "poor packing", "bad packing",
            "damage aagidhe", "packaging damage", "box damage", "torn package",
            "open package", "leaking", "broken packaging", "poor packaging",
            "packaging ketta", "packing ketta", "box ketta", "damaged box",
            "no packaging", "no packing", "loosely packed", "open box",
        ],
    },

    "Price": {
        "positive": [
            "value for money", "worth", "worth it", "reasonable price",
            "reasonable ide", "good price", "affordable", "cheap price",
            "price ok", "price chenna", "price super", "price reasonable",
            "cost effective", "budget friendly", "good deal", "best price",
            "price ge value", "price sari", "money worth", "bargain",
        ],
        "negative": [
            "expensive", "costly", "overpriced", "waste of money", "money waste",
            "not worth", "not worth it", "bahala costly", "tumba costly",
            "price jaasti", "price tumba", "price ketta", "waste maney",
            "price high", "costly agide", "too expensive", "poor value",
        ],
    },

    "Customer Service": {
        "positive": [
            "good service", "nice service", "helpful", "support chenna",
            "customer service good", "service super", "quick response",
            "fast response", "good support", "helpful team", "resolved",
            "problem solved", "friendly staff", "good experience",
        ],
        "negative": [
            "bad service", "poor service", "no response", "no support",
            "customer service ketta", "service ketta", "rude", "not helpful",
            "no reply", "no help", "cheating", "fraud", "scam",
            "fake seller", "bad seller", "poor response", "ignored",
            "no refund", "refund illa", "return problem",
        ],
    },

    "Performance": {
        "positive": [
            "works well", "working good", "performance super", "performance chenna",
            "fast performance", "smooth", "works perfectly", "good performance",
            "excellent performance", "performs well", "working perfectly",
            "no issues", "no problem", "works great", "functions well",
        ],
        "negative": [
            "not working", "work agalla", "stopped working", "performance ketta",
            "slow", "hanging", "lag", "lagging", "crash", "crashing",
            "not functioning", "malfunctioning", "issues", "problem",
            "bug", "error", "performance bad", "poor performance",
            "restart agatte", "switch off agatte",
        ],
    },

    "Durability": {
        "positive": [
            "durable", "long lasting", "strong", "sturdy", "solid",
            "good build", "well built", "durable agide", "strong agide",
            "long life", "years agidhe", "still working", "lasts long",
        ],
        "negative": [
            "broke", "broken", "cracked", "melted", "rusted",
            "stopped working", "dead", "not durable", "low durability",
            "broke quickly", "month alley", "week alley", "days alley",
            "life kamma", "life illa", "cheap build",
        ],
    },

    "Design": {
        "positive": [
            "beautiful", "nice design", "good design", "looks good",
            "attractive", "stylish", "good looking", "design super",
            "design chenna", "design ishtavagide", "premium look",
            "good appearance", "nice color", "good color", "trendy",
        ],
        "negative": [
            "ugly", "bad design", "design ketta", "not attractive",
            "cheap look", "looks cheap", "design illa", "color different",
            "color faded", "color ketta", "color bad", "size wrong",
            "size different", "photo different", "image different",
            "not as shown", "as shown alla",
        ],
    },

    "Refund / Return": {
        "positive": [
            "easy return", "return ok", "refund ok", "refund bantu",
            "return accepted", "money back", "refund aythu", "return aythu",
            "good return policy", "hassle free return",
        ],
        "negative": [
            "return maadtini", "refund beku", "return problem", "refund illa",
            "no refund", "no return", "return reject", "return agalla",
            "refund agalla", "money back agilla", "return ketta",
        ],
    },
}

# Icons for each aspect
ASPECT_ICONS = {
    "Quality":          "⭐",
    "Delivery":         "🚚",
    "Packaging":        "📦",
    "Price":            "💰",
    "Customer Service": "🎧",
    "Performance":      "⚡",
    "Durability":       "🔩",
    "Design":           "🎨",
    "Refund / Return":  "↩️",
}

ASPECT_COLORS = {
    "Quality":          "#1d4ed8",
    "Delivery":         "#0ea5e9",
    "Packaging":        "#8b5cf6",
    "Price":            "#10b981",
    "Customer Service": "#f59e0b",
    "Performance":      "#ef4444",
    "Durability":       "#6366f1",
    "Design":           "#ec4899",
    "Refund / Return":  "#64748b",
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_aspects_single(review_text: str) -> dict:
    """
    Analyse aspects in a single review.

    Returns:
    {
        "aspects": {
            "Quality":   {"sentiment": "Positive", "keywords": ["super", "durable"]},
            "Delivery":  {"sentiment": "Negative", "keywords": ["late aythu"]},
            ...
        },
        "detected_count": 3,
    }
    """
    text    = review_text.lower().strip()
    aspects = {}

    for aspect, kw_map in ASPECT_KEYWORDS.items():
        pos_hits = _find_hits(text, kw_map["positive"])
        neg_hits = _find_hits(text, kw_map["negative"])

        if not pos_hits and not neg_hits:
            continue  # aspect not mentioned

        if len(pos_hits) > len(neg_hits):
            sentiment = "Positive"
            keywords  = pos_hits
        elif len(neg_hits) > len(pos_hits):
            sentiment = "Negative"
            keywords  = neg_hits
        else:
            sentiment = "Mixed"
            keywords  = pos_hits + neg_hits

        aspects[aspect] = {
            "sentiment": sentiment,
            "keywords":  keywords[:5],  # top 5 matched keywords
        }

    return {
        "aspects":        aspects,
        "detected_count": len(aspects),
    }


def analyse_aspects_bulk(reviews: list) -> dict:
    """
    Analyse aspects across a list of reviews.

    Returns aggregate counts and percentages per aspect per sentiment.
    """
    # aspect → {"Positive": count, "Negative": count, "Mixed": count, "total": count}
    agg = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Mixed": 0, "total": 0})

    for review in reviews:
        result = analyse_aspects_single(str(review))
        for aspect, data in result["aspects"].items():
            agg[aspect][data["sentiment"]] += 1
            agg[aspect]["total"] += 1

    # Build final summary
    summary = {}
    for aspect, counts in agg.items():
        total = counts["total"]
        if total == 0:
            continue
        pos_pct = round(counts["Positive"] / total * 100, 1)
        neg_pct = round(counts["Negative"] / total * 100, 1)
        mix_pct = round(counts["Mixed"]    / total * 100, 1)

        if counts["Positive"] > counts["Negative"]:
            overall = "Positive"
        elif counts["Negative"] > counts["Positive"]:
            overall = "Negative"
        else:
            overall = "Mixed"

        summary[aspect] = {
            "positive":     counts["Positive"],
            "negative":     counts["Negative"],
            "mixed":        counts["Mixed"],
            "total":        total,
            "pos_pct":      pos_pct,
            "neg_pct":      neg_pct,
            "mix_pct":      mix_pct,
            "overall":      overall,
            "icon":         ASPECT_ICONS.get(aspect, "📌"),
            "color":        ASPECT_COLORS.get(aspect, "#64748b"),
        }

    return summary


def generate_recommendation(
    sentiment_counts: dict,
    aspect_summary: dict,
    model_used: str = "SVM",
    total_reviews: int = 0,
) -> dict:
    """
    Generate final product recommendation with reasons.

    sentiment_counts: {"Positive": n, "Negative": n, "Neutral": n}
    aspect_summary:   output of analyse_aspects_bulk()
    """
    pos = sentiment_counts.get("Positive", 0)
    neg = sentiment_counts.get("Negative", 0)
    neu = sentiment_counts.get("Neutral",  0)
    total = total_reviews or (pos + neg + neu) or 1

    pos_pct = round(pos / total * 100, 1)
    neg_pct = round(neg / total * 100, 1)
    neu_pct = round(neu / total * 100, 1)

    # Overall verdict
    if pos_pct >= 60:
        verdict       = "HIGHLY RECOMMENDED"
        verdict_color = "#10b981"
        verdict_bg    = "#f0fdf4"
    elif pos_pct >= 45:
        verdict       = "GOOD PRODUCT"
        verdict_color = "#10b981"
        verdict_bg    = "#f0fdf4"
    elif neg_pct >= 50:
        verdict       = "NOT RECOMMENDED"
        verdict_color = "#ef4444"
        verdict_bg    = "#fef2f2"
    elif neg_pct >= 35:
        verdict       = "NEEDS IMPROVEMENT"
        verdict_color = "#f59e0b"
        verdict_bg    = "#fffbeb"
    else:
        verdict       = "AVERAGE PRODUCT"
        verdict_color = "#f59e0b"
        verdict_bg    = "#fffbeb"

    # Top positive factors
    top_positive = sorted(
        [(a, d) for a, d in aspect_summary.items() if d["overall"] == "Positive"],
        key=lambda x: x[1]["pos_pct"], reverse=True
    )[:3]

    # Top negative factors
    top_negative = sorted(
        [(a, d) for a, d in aspect_summary.items() if d["overall"] == "Negative"],
        key=lambda x: x[1]["neg_pct"], reverse=True
    )[:3]

    # Most complained aspect
    most_complained = None
    if aspect_summary:
        most_complained = max(
            [(a, d) for a, d in aspect_summary.items()],
            key=lambda x: x[1]["negative"]
        )[0]

    # Most praised aspect
    most_praised = None
    if aspect_summary:
        most_praised = max(
            [(a, d) for a, d in aspect_summary.items()],
            key=lambda x: x[1]["positive"]
        )[0]

    # Build reason bullets
    reasons = []
    if top_positive:
        for aspect, d in top_positive:
            reasons.append({
                "type":    "positive",
                "text":    f"{d['pos_pct']}% of users gave positive feedback on {aspect}",
                "aspect":  aspect,
            })
    if top_negative:
        for aspect, d in top_negative:
            reasons.append({
                "type":    "negative",
                "text":    f"{d['neg_pct']}% of users complained about {aspect}",
                "aspect":  aspect,
            })

    return {
        "verdict":         verdict,
        "verdict_color":   verdict_color,
        "verdict_bg":      verdict_bg,
        "pos":             pos, "neg": neg, "neu": neu,
        "pos_pct":         pos_pct, "neg_pct": neg_pct, "neu_pct": neu_pct,
        "total":           total,
        "top_positive":    top_positive,
        "top_negative":    top_negative,
        "most_complained": most_complained,
        "most_praised":    most_praised,
        "reasons":         reasons,
        "model_used":      model_used,
    }


def generate_text_report(rec: dict, aspect_summary: dict, filename: str = "") -> str:
    """Generate plain-text summary report for download."""
    from datetime import datetime
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "=" * 56

    lines = [
        line,
        "  PRODUCT REVIEW ANALYTICS REPORT",
        "  Romanized Kannada Sentiment Analysis",
        line,
        f"  Generated    : {ts}",
    ]
    if filename:
        lines.append(f"  Source File  : {filename}")
    lines += [
        f"  Model Used   : {rec['model_used']}",
        "",
        line,
        "  OVERALL SENTIMENT",
        line,
        f"  Total Reviews : {rec['total']}",
        f"  Positive      : {rec['pos']}  ({rec['pos_pct']}%)",
        f"  Negative      : {rec['neg']}  ({rec['neg_pct']}%)",
        f"  Neutral       : {rec['neu']}  ({rec['neu_pct']}%)",
        "",
        line,
        "  PRODUCT VERDICT",
        line,
        f"  {rec['verdict']}",
        "",
    ]

    if rec.get("most_praised"):
        lines.append(f"  Top Positive Aspect : {rec['most_praised']}")
    if rec.get("most_complained"):
        lines.append(f"  Top Negative Aspect : {rec['most_complained']}")

    lines += ["", line, "  ASPECT-WISE ANALYSIS", line]
    for aspect, d in aspect_summary.items():
        icon = ASPECT_ICONS.get(aspect, "-")
        lines.append(
            f"  {aspect:<20} | {d['overall']:<10} | "
            f"Pos:{d['pos_pct']}%  Neg:{d['neg_pct']}%  ({d['total']} mentions)"
        )

    lines += [
        "", line,
        "  KEY INSIGHTS",
        line,
    ]
    for r in rec.get("reasons", []):
        prefix = "  [+]" if r["type"] == "positive" else "  [-]"
        lines.append(f"{prefix} {r['text']}")

    lines += [
        "",
        line,
        "  JSS Science & Technology University, Mysuru",
        "  Dept. of CS & Engineering | Final Year Project 2024-25",
        line,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _find_hits(text: str, keyword_list: list) -> list:
    """Return matched keywords found in text (multi-word phrases checked first)."""
    text   = " " + text + " "
    hits   = []
    # Sort by length descending so longer phrases match before sub-phrases
    for kw in sorted(keyword_list, key=len, reverse=True):
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
            hits.append(kw)
    return hits
