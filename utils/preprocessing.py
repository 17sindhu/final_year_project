"""preprocessing.py — 7-step pipeline for Romanized Kannada text."""

import re
import pandas as pd

STOPWORDS = {
    "alli","inda","ide","idhe","alla"
    ,"enu","mele","kelage",
    "naanu","neenu","avanu","avalu","avaru","nimma","namma","ella",
    "ondu","eradu","modalu","hege","yaarige","yaaru","aaga","maadi",
    "hodha","baro","baratte","hogatte","anta","andre","adhu","idhu",
    "karana","heli","helide","matthe","innu","the","a","is","was",
    "are","were","for","in","on","at","to","of","this","that","it",
    "with","from","an","i","my","me","we","our","they","their",
}

DEMO_DICT = {
    "chenagide":"chennagide","chennagidi":"chennagide",
    "superu":"super","suuper":"super","sooooper":"super",
    "swalppa":"swalpa","swlpa":"swalpa",
    "bajaaru":"bejaar","bejaaru":"bejaar",
    "bahala":"tumba","bahalla":"tumba",
    "kettadhu":"ketta","kettadu":"ketta",
    "maadidare":"maadidhaare",
    "gottilva":"gottilla",
}


def load_normalization_dict(path="dictionary.csv"):
    try:
        df = pd.read_csv(path, header=None, dtype=str)
        first = str(df.iloc[0,0]).strip().lower()
        if first in ("variant","raw","word","spelling","original","from","key"):
            df = pd.read_csv(path, header=0, dtype=str)
        df.columns = ["variant","canonical"] + list(df.columns[2:])
        df = df[["variant","canonical"]].dropna()
        df["variant"]   = df["variant"].str.strip().str.lower()
        df["canonical"] = df["canonical"].str.strip().str.lower()
        return dict(zip(df["variant"], df["canonical"]))
    except Exception:
        return DEMO_DICT


def preprocess(text, norm_dict):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    tokens = [norm_dict.get(t, t) for t in tokens]
    clean  = " ".join(tokens)
    return clean, tokens
