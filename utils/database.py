"""database.py — SQLite prediction history store."""

import sqlite3, json, os, csv, io
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history.db")


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp  TEXT NOT NULL,
        raw_text   TEXT NOT NULL,
        clean_text TEXT NOT NULL,
        model      TEXT NOT NULL,
        method     TEXT NOT NULL,
        label      TEXT NOT NULL,
        confidence REAL NOT NULL,
        tokens     TEXT NOT NULL
    )""")
    c.commit(); c.close()


def save(raw, clean, model, method, label, conf, tokens):
    c = _conn()
    cur = c.execute(
        "INSERT INTO predictions(timestamp,raw_text,clean_text,model,method,label,confidence,tokens)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), raw, clean,
         model, method, label, round(float(conf),4), json.dumps(tokens))
    )
    c.commit(); rid = cur.lastrowid; c.close()
    return rid


def get_all(limit=500):
    c = _conn()
    rows = c.execute(
        "SELECT id,timestamp,raw_text,clean_text,model,method,label,confidence,tokens"
        " FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [{"id":r[0],"timestamp":r[1],"raw_text":r[2],"clean_text":r[3],
             "model":r[4],"method":r[5],"label":r[6],"confidence":r[7],
             "tokens":json.loads(r[8])} for r in rows]


def get_stats():
    c = _conn()
    total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    labels = dict(c.execute("SELECT label,COUNT(*) FROM predictions GROUP BY label").fetchall())
    models = {r[0]:{"count":r[1],"avg_conf":round(r[2]*100,1)}
              for r in c.execute("SELECT model,COUNT(*),AVG(confidence) FROM predictions GROUP BY model").fetchall()}
    methods = dict(c.execute("SELECT method,COUNT(*) FROM predictions GROUP BY method").fetchall())
    avg_conf = c.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    daily = [{"date":r[0],"count":r[1]} for r in
             c.execute("SELECT DATE(timestamp),COUNT(*) FROM predictions GROUP BY DATE(timestamp) ORDER BY DATE(timestamp) DESC LIMIT 30").fetchall()]
    c.close()
    return {"total":total,"by_label":labels,"by_model":models,
            "by_method":methods,"avg_conf":round(avg_conf*100,1),"daily":daily}


def delete_one(rid):
    c = _conn(); c.execute("DELETE FROM predictions WHERE id=?", (rid,)); c.commit(); c.close()


def clear_all():
    c = _conn(); c.execute("DELETE FROM predictions"); c.commit(); c.close()


def export_csv():
    rows = get_all(10000)
    if not rows:
        return "id,timestamp,raw_text,clean_text,model,method,label,confidence\n"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["id","timestamp","raw_text","clean_text","model","method","label","confidence"])
    w.writeheader()
    for r in rows:
        w.writerow({k:v for k,v in r.items() if k!="tokens"})
    return buf.getvalue()
