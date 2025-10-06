# services/utils.py
from datetime import datetime, UTC
import pandas as pd, hashlib

# ───── 공통 유틸 ─────
def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_str(x) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return str(x).strip()

def parse_date_any(x):
    if x is None: return None
    d = pd.to_datetime(x, errors="coerce")
    if pd.isna(d): return None
    return d.strftime("%Y-%m-%d")

def deterministic_review_id(product_id, review_date, review_text) -> str:
    pid = safe_str(product_id) or "unknown"
    d   = safe_str(review_date) or "0000-00-00"
    t   = " ".join(safe_str(review_text).split())
    h   = hashlib.sha256(f"{pid}|{d}|{t}".encode("utf-8")).hexdigest()[:16]
    return f"{pid}|{d}|{h}"

def normalize_category(s):
    if not s: return None
    z = str(s).strip().lower()
    if z in {"running","run","러닝"}: return "RUNNING"
    if z in {"hiking","trekking","등산","하이킹","등산/하이킹"}: return "HIKING"
    return z.upper()

def category_korean_label(std):
    return {"RUNNING":"러닝", "HIKING":"등산/하이킹"}.get(std, std)

