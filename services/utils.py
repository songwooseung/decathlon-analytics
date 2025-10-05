# services/utils.py
from datetime import datetime, UTC
import pandas as pd, hashlib
import re, html, unicodedata
from typing import Optional, Dict, List
from sqlalchemy import text
from core.db import engine
from functools import lru_cache

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

# ───── 제품 메타 조회 ─────
@lru_cache(maxsize=10000)
def get_product_summary(pid: str) -> Optional[Dict]:
    sql = """
    SELECT product_id, product_name, category, subcategory, brand,
           price, avg_rating, total_reviews, url, thumbnail_url
    FROM product_summary
    WHERE product_id = :pid
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"pid": pid}).mappings().first()
    return dict(row) if row else None

# ───── 리뷰 문구 정제 ─────
_HANGUL = re.compile(r"[가-힣]")
def _clean_quote(s: str) -> str:
    if not s: return ""
    s = html.unescape(unicodedata.normalize("NFKC", str(s)))
    # 쓰레기/광고 제거
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"(?:^|[\s])(?:홈|Home)\s*>\s*[^>]+(?:>\s*[^>]+)*", " ", s, flags=re.I)
    s = re.sub(r"(쿠폰 ?코드|전품목|장바구니 담기|사이즈\s*선택|판매자\s*DECATHLON|배송 옵션|매장 이용|모델번호|%의 고객이.*추천)", " ", s, flags=re.I)
    # 반복/짧은문장 제거
    s = re.sub(r"(.{2,10})\1{2,}", " ", s)  # 같은 패턴 반복
    if re.fullmatch(r"(좋아요|만족합니다|잘 받았습니다)[.! ]*", s): return ""
    s = re.sub(r"[~!@#%^&*_=+]{3,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 4 or not _HANGUL.search(s): return ""
    return s[:160]

def _dedupe_keep_order(strings: List[str]) -> List[str]:
    seen, out = set(), []
    for t in strings:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def get_reviews(pid: str, sentiment: str, limit: int = 2) -> List[str]:
    sql = """
    SELECT review_text
    FROM reviews
    WHERE product_id = :pid AND sentiment = :sent
    ORDER BY review_date DESC
    LIMIT :limit
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(sql),
            {"pid": pid, "sent": sentiment, "limit": max(limit*3, limit)},
        ).scalars().all()

    cleaned = _dedupe_keep_order([_clean_quote(x) for x in rows])
    return cleaned[:limit]

def pick_positive_texts(pid: str, limit: int = 2) -> List[str]:
    texts = get_reviews(pid, "positive", limit=limit)
    if not texts:
        texts = get_reviews(pid, "mixed", limit=limit)
    return texts[:limit]

def pick_negative_texts(pid: str, limit: int = 2) -> List[str]:
    texts = get_reviews(pid, "negative", limit=limit)
    if not texts:
        texts = get_reviews(pid, "mixed", limit=limit)
    return texts[:limit]