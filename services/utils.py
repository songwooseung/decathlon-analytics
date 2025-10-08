# services/utils.py
from datetime import datetime, UTC
import pandas as pd, hashlib, re
from typing import Optional

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

# (호환성용) idx 인자를 받아도 무시하도록 추가
def deterministic_review_id(product_id, review_date, review_text, idx=None) -> str:
    pid = safe_str(product_id) or "unknown"
    d   = safe_str(review_date) or "0000-00-00"
    t   = " ".join(safe_str(review_text).split())
    salt = f"|{idx}" if idx is not None else ""
    h   = hashlib.sha256(f"{pid}|{d}|{t}{salt}".encode("utf-8")).hexdigest()[:16]
    return f"{pid}|{d}|{h}"

def normalize_category(s):
    if not s: return None
    z = str(s).strip().lower()
    if z in {"running","run","러닝"}: return "RUNNING"
    if z in {"hiking","trekking","등산","하이킹","등산/하이킹","스포츠"}: return "HIKING"
    return z.upper()

def category_korean_label(std):
    return {"RUNNING":"러닝", "HIKING":"등산/하이킹"}.get(std, std)

# ── 숫자/금액/평점 정제 ──
def to_int_any(x):
    s = safe_str(x)
    if not s:
        return 0

    # 1소수점 처리 (99.0 → 99)
    if re.match(r"^\d+(\.\d+)?$", s):
        try:
            return int(float(s))
        except Exception:
            pass

    # 콤마/원 단위 등 제거
    s = re.sub(r"[^\d\-]", "", s)
    try:
        return int(s)
    except Exception:
        return 0

def to_float_any(x):
    s = safe_str(x).replace(",", "")
    try:
        return float(s)
    except Exception:
        return None