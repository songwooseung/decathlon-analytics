# services/embeddings.py
import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from openai import OpenAI

from core.db import engine
from core.config import (
    EMBEDDING_MODEL,          # ex) "text-embedding-3-small"
    EMBEDDING_CACHE_PATH,     # ex) "data/embeddings.pkl"
    OPENAI_API_KEY,
)

# =========================
# 1) 텍스트 정제 유틸
# =========================
import re
import html
import unicodedata

# --- 정규식 패턴 정의 ---
BREADCRUMB_RE = re.compile(r"(?:^|[\s])(?:홈|Home)\s*>\s*[^>]+(?:>\s*[^>]+)*", re.IGNORECASE)
URL_RE        = re.compile(r"https?://\S+")
PRICE_RE      = re.compile(r"(?:[₩]?\s?\d{1,3}(?:,\d{3})+|\d+\s?원)")
SKU_RE        = re.compile(r"\b[A-Z]{2,}\d{3,}\b")  # 대충 모델코드/상품코드
SIZE_RE       = re.compile(r"\b(?:XS|S|M|L|XL|2XL|3XL|52\s?M-L|52M|52\s?ML)\b")
USER_RE       = re.compile(r"^[A-Za-z가-힣0-9_.\- ]{2,20}\s*$")
MULTI_SPACE   = re.compile(r"\s+")
HANGUL_RE     = re.compile(r"[가-힣]")

# --- 광고·잡음 문장 감지 ---
AD_NOISE_PATTERNS = [
    r"쿠폰 ?코드", r"할인", r"전품목", r"장바구니 담기", r"사이즈(를|를)? 선택",
    r"판매자\s*DECATHLON", r"배송 옵션", r"매장 이용", r"내\s*사이즈\s*찾기",
    r"모델번호", r"%의 고객이.*추천", r"홈>모든\s*스포츠",
]

def _cut_english_prefix(s: str) -> str:
    """
    영어/닉네임 같은 앞부분 제거하고, 첫 한글부터만 남김.
    ex) 'good man 재질이...' → '재질이...'
        'CHANLAN 기존에 구매...' → '기존에 구매...'
    """
    m = re.search(r"[가-힣].*", s)
    return m.group(0) if m else s

def _drop_noise_line(line: str) -> bool:
    """광고, 메뉴, 불필요한 리뷰 제외"""
    if not line or len(line.strip()) < 2:
        return True
    if any(re.search(p, line, flags=re.I) for p in AD_NOISE_PATTERNS):
        return True
    if line.strip().startswith(("홈>", "Home>", "장바구니")):
        return True
    return False

def clean_review(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""

    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)

    lines = []
    for raw in t.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _drop_noise_line(s):
            continue
        s = BREADCRUMB_RE.sub("", s)
        s = URL_RE.sub("", s)
        s = PRICE_RE.sub("", s)
        s = SKU_RE.sub("", s)
        s = SIZE_RE.sub("", s)
        s = _cut_english_prefix(s)
        s = re.sub(r"[~!@#%^&*_=+]{3,}", " ", s)
        s = re.sub(r"[^0-9A-Za-z가-힣 .,!?()\-/]", " ", s)
        s = MULTI_SPACE.sub(" ", s).strip()

        # ❗ 동일 문장 반복 제거
        if re.search(r"(.{2,10})\1{2,}", s):
            continue

        # ❗ 의미 없는 짧은 문장, ‘잘 받았습니다’류 제거
        if len(s) < 4 or not HANGUL_RE.search(s):
            continue
        if re.search(r"(잘 받았습니다|좋아요|만족합니다){2,}", s):
            continue
        if re.fullmatch(r"(잘 받았습니다|좋아요|만족합니다|감사합니다)\s*", s):
            continue

        lines.append(s)

    seen, dedup = set(), []
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)

    cleaned = " ".join(dedup).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned

# =========================
# 2) 내부 유틸
# =========================
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

SUBTOKENS = {
    "자켓": ["자켓", "바람막이", "윈드재킷", "재킷", "wind"],
    "가방": ["가방", "백팩", "배낭", "데이팩", "daypack", "pack"],
    "모자": ["모자", "캡", "헤드밴드", "비니", "버킷햇", "hat", "cap"],
    "양말": ["양말", "삭스", "socks"],
    "티"  : ["티", "티셔츠", "top", "shirt"],
    "쇼츠": ["쇼츠", "숏츠", "반바지", "shorts"],
    "바지": ["바지", "팬츠", "하의", "pants"],
    "벨트": ["벨트", "belt"],
    "플라스크": ["플라스크", "물병", "보틀", "soft flask", "flask"],
    "신발": ["신발", "러닝화", "하이킹화", "트레일러닝화", "슈즈", "shoes"],
}

def _match_alias(text: str, key: Optional[str]) -> bool:
    """제품명/서브카테고리에서 key의 동의어 포함 여부"""
    if not key:
        return True
    aliases = [key] + SUBTOKENS.get(key, [])
    t = (text or "").lower()
    return any(a.lower() in t for a in aliases)

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _approx_tokens(s: str) -> int:
    """토큰 대략 추정 (영/한 혼합 가정: 1토큰 ≈ 3문자)"""
    return max(1, len(s) // 3)


# =========================
# 3) 코퍼스 생성(DB → 문서)
# =========================
def build_corpus() -> List[Dict[str, Any]]:
    """제품별 리뷰를 하나의 문서로 구성"""
    with engine.begin() as conn:
        rv = pd.read_sql(text("SELECT product_id, review_text, rating FROM reviews"), conn)
        sm = pd.read_sql(text("""
            SELECT product_id, product_name, category, subcategory, brand, price, avg_rating
            FROM product_summary
        """), conn)

    if rv.empty and sm.empty:
        return []

    if rv.empty:
        rv_agg = pd.DataFrame(columns=["product_id", "reviews_blob"])
    else:
        rv["review_text"] = rv["review_text"].fillna("").astype(str).map(clean_review)
        rv = rv[rv["review_text"].str.len() > 0]
        rv = rv.drop_duplicates(subset=["product_id", "review_text"])  # 중복 제거
        rv_agg = (
            rv.groupby("product_id")["review_text"]
              .apply(lambda s: "\n".join(s.tolist()[:400]))  # 문장 수 제한
              .reset_index()
              .rename(columns={"review_text": "reviews_blob"})
        )

    df = sm.merge(rv_agg, on="product_id", how="left")
    df["reviews_blob"] = df["reviews_blob"].fillna("")

    # build_corpus() 내부 row_to_doc 수정
    def row_to_doc(r):
        head = f"[{r.get('category')}/{r.get('subcategory')}] {r.get('product_name')} ({r.get('brand')})"
        meta = f"평점 {r.get('avg_rating')}, 가격 {r.get('price')}"
        body = r.get("reviews_blob") or ""
        if len(body) > 7_000:
            body = body[:7_000]
        text_ = f"{head}\n{meta}\n---\n{body}"
        return {
            "doc_id": str(r["product_id"]),
            "product_id": str(r["product_id"]),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "product_name": r.get("product_name"),
            "text": text_,
        }

    return [row_to_doc(r) for _, r in df.iterrows()]


# =========================
# 4) 임베딩 생성/저장/로드
# =========================
def _approx_tokens(s: str) -> int:
    # 보수적으로: 1토큰 ≈ 2 문자
    return max(1, len(s) // 2)

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")  # text-embedding-3-small 차원

    SAFE_INPUT_TOKENS_PER_ITEM = 3_500   # 1개 입력 상한
    SAFE_BATCH_TOKENS          = 3_900   # 배치 상한(사실상 1개)

    # per-item clip (≈ 7,000자)
    clipped = []
    for t in texts:
        if _approx_tokens(t) > SAFE_INPUT_TOKENS_PER_ITEM:
            t = t[:SAFE_INPUT_TOKENS_PER_ITEM * 2]  # 1토큰≈2문자 가정
        clipped.append(t)

    vecs: List[np.ndarray] = []
    i = 0
    while i < len(clipped):
        batch, tok_sum = [], 0
        while i < len(clipped):
            est = _approx_tokens(clipped[i])
            if batch and tok_sum + est > SAFE_BATCH_TOKENS:
                break
            # 배치 한도를 3,900으로 뒀기 때문에 결국 1개만 들어갑니다.
            batch.append(clipped[i])
            tok_sum += est
            i += 1

        # 안전망: 만약 드물게 400이 나면 더 잘라서 재시도
        try:
            out = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except Exception:
            shorter = [x[:4000] for x in batch]  # 강제 추가 클리핑
            out = client.embeddings.create(model=EMBEDDING_MODEL, input=shorter)

        vecs.append(np.array([d.embedding for d in out.data], dtype="float32"))

    return np.vstack(vecs)


def save_index(index: Dict[str, Any]):
    _ensure_dir(EMBEDDING_CACHE_PATH)
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(index, f)


def load_index() -> Optional[Dict[str, Any]]:
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        return None
    with open(EMBEDDING_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def rebuild_index() -> Dict[str, Any]:
    """DB → 임베딩 → 캐시 저장"""
    docs = build_corpus()
    client = OpenAI(api_key=OPENAI_API_KEY)
    embs = embed_texts(client, [d["text"] for d in docs])

    index = {
        "model": EMBEDDING_MODEL,
        "docs": docs,
        "embeddings": embs,
    }
    save_index(index)
    return index


def ensure_index() -> Dict[str, Any]:
    idx = load_index()
    if idx is None:
        idx = rebuild_index()
    return idx


# =========================
# 5) 검색
# =========================
def search_similar(
    query: str,
    top_k: int,
    category: Optional[str] = None,
    subkw: Optional[str] = None,
) -> List[Dict[str, Any]]:
    idx = ensure_index()
    docs = idx["docs"]
    embs = idx["embeddings"]

    # 1) 후보 필터 (메타 기반)
    cand_idx = []
    for i, d in enumerate(docs):
        if category and d.get("category") != category:
            continue
        name_sub = f"{d.get('product_name','')} {d.get('subcategory','')}"
        if not _match_alias(name_sub, subkw):
            continue
        cand_idx.append(i)

    # 후보가 없으면 전체에서 검색
    if not cand_idx:
        cand_idx = list(range(len(docs)))

    # 2) 쿼리 확장 (동의어/카테고리 같이 넣어 신호 강화)
    expanded = query
    if category:
        expanded += f" {category}"
    if subkw:
        expanded += " " + " ".join(set([subkw] + SUBTOKENS.get(subkw, [])))

    client = OpenAI(api_key=OPENAI_API_KEY)
    qv = embed_texts(client, [expanded])                    # (1, D)
    sub_embs = embs[cand_idx]                               # (Ncand, D)
    D = _cosine_sim(qv, sub_embs)                           # (1, Ncand)
    order_local = np.argsort(-D[0])[:top_k]
    scores_local = D[0, order_local]

    results = []
    for idx_i, sc in zip([cand_idx[j] for j in order_local], scores_local):
        results.append({**docs[int(idx_i)], "score": float(sc)})
    return results