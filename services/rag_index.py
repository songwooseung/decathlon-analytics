# services/rag_index.py
from typing import List, Dict, Any, Optional, Tuple
import os, re, pickle, json
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PKL_PATH = os.environ.get("RAG_EMB_PATH", "data/embeddings.pkl")

def _fetch_source_dataframe(db: Session) -> pd.DataFrame:
    q = """
    SELECT
        r.product_id,
        COALESCE(r.category, ps.category)          AS category,
        COALESCE(r.subcategory, ps.subcategory)    AS subcategory,
        COALESCE(r.product_name, ps.product_name)  AS name,
        ps.price                                    AS price,
        ps.avg_rating                               AS rating_avg,
        ps.total_reviews                            AS review_count,
        ps.url                                      AS url,
        ps.thumbnail_url                            AS thumbnail_url,
        r.review_text                               AS review_text,
        'reviews'                                   AS source
    FROM reviews r
    LEFT JOIN product_summary ps USING (product_id)
    WHERE r.review_text IS NOT NULL
      AND length(r.review_text) >= 10
    """
    return pd.read_sql(q, db.bind)

def _basic_clean(text_: str) -> str:
    if not isinstance(text_, str):
        return ""
    t = text_.strip()

    # 광고/배너/브레드크럼/장바구니 등 패턴 제거 (필요 시 추가)
    bad_patterns = [
        r"홈>.*", r"장바구니.*", r"사이즈를 선택하세요.*", r"쿠폰코드\s*:\s*\w+",
        r"판매자\s*DECATHLON.*", r"\b\d+/?\d+\b", r"[|｜]\s*쿠폰.*", r"리뷰 작성하기.*"
    ]
    for p in bad_patterns:
        t = re.sub(p, " ", t)

    # 닉네임 프리픽스 제거 (영숫자/밑줄 + 공백/콜론)
    t = re.sub(r"^[A-Za-z0-9_]+[\s:]+", "", t)

    # 연속 공백 정리
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def build_index_from_db(db: Session) -> Dict[str, Any]:
    df = _fetch_source_dataframe(db)
    if df.empty:
        raise RuntimeError("No rows to index.")

    # 텍스트 정제
    df["clean_text"] = df["review_text"].map(_basic_clean)
    df = df[df["clean_text"].str.len() >= 10].copy()

    # 문서 목록 (메타 포함)
    docs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        docs.append({
            "product_id": r.get("product_id"),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "name": r.get("name"),
            "price": r.get("price"),
            "rating_avg": r.get("rating_avg"),
            "review_count": r.get("review_count"),
            "url": r.get("url"),
            "thumbnail_url": r.get("thumbnail_url"),
            "source": r.get("source"),
            "text": r.get("clean_text"),
        })

    corpus = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(corpus)

    os.makedirs(os.path.dirname(PKL_PATH) or ".", exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "matrix": X,
        "docs": docs,
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "embed_model": "tfidf",
    }
    with open(PKL_PATH, "wb") as f:
        pickle.dump(payload, f)

    return { "chunks": len(docs), "built_at": payload["built_at"], "path": PKL_PATH }

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"Index not found: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)

def _extract_preferences(query: str) -> Dict[str, Any]:
    """
    사용자의 질의에서
      - 선호 카테고리/서브카테고리
      - 액세서리 제외 여부(의류 요청시)
      - 예산(최소/최대/타겟)
      - 시즌/기능 키워드
    를 추출한다.
    """
    q = (query or "").lower()

    # ---- 카테고리/서브카테고리 키워드 사전 (원하면 계속 보강)
    cat_map = {
        "RUNNING":  ["러닝","running","조깅","트랙"],
        "HIKING":   ["하이킹","등산","hiking","트레킹","trek"],
        "FITNESS":  ["피트니스","헬스","짐","gym","weight"],
        "SWIMMING": ["수영","swim","워터"],
        "CYCLING":  ["사이클","자전거","cycling","bike"],
    }
    sub_map = {
        # 의류류
        "자켓": ["자켓","재킷","바람막이","윈드브레이커","패딩","베스트","플리스","다운","롱패딩"],
        "상의": ["상의","후드","후디","스웻","맨투맨","티셔츠","티","롱슬리브","폴라","셔츠"],
        "하의": ["바지","팬츠","쇼츠","레깅스","타이츠","트레이닝","조거"],
        # 신발/가방
        "신발": ["신발","슈즈","러닝화","등산화","트레일러닝화"],
        "가방": ["백팩","배낭","가방","히프색","크로스백"],
        # 액세서리
        "액세서리": ["장갑","모자","버프","워머","헤드밴드","양말","토시","넥워머","글러브","비니"],
    }

    want_cats = set()
    want_subs = set()
    for cat, kws in cat_map.items():
        if any(k in q for k in kws):
            want_cats.add(cat)
    for sub, kws in sub_map.items():
        if any(k in q for k in kws):
            want_subs.add(sub)

    # 의류류 요청이면 액세서리 제외
    exclude_accessory = any(k in q for k in (sub_map["자켓"] + sub_map["상의"] + sub_map["하의"] + ["옷","의류"]))

    # ---- 예산/가격 파싱 (만원/원/under/over/이하/이상/~ 사이 범위)
    price_min = None
    price_max = None
    target_price = None

    # 1) 범위: "3만~6만", "30000~60000원"
    m = re.search(r"(\d+)\s*만?\s*[~\-]\s*(\d+)\s*만?", q)
    if m:
        price_min = int(m.group(1)) * 10000
        price_max = int(m.group(2)) * 10000
    else:
        m = re.search(r"(\d{2,6})\s*원?\s*[~\-]\s*(\d{2,6})\s*원?", q)
        if m:
            price_min = int(m.group(1))
            price_max = int(m.group(2))

    # 2) 이하/이상/under/over
    if price_min is None and price_max is None:
        m = re.search(r"(\d+)\s*만?\s*이하|under\s*(\d+)", q)
        if m:
            v = int(m.group(1) or m.group(2)) * (10000 if "만" in (m.group(0) or "") else 1)
            price_max = v
        m = re.search(r"(\d+)\s*만?\s*이상|over\s*(\d+)", q)
        if m:
            v = int(m.group(1) or m.group(2)) * (10000 if "만" in (m.group(0) or "") else 1)
            price_min = v

    # 3) “5만원대/약 6만/한 7만원쯤” → 타겟값
    if target_price is None:
        m = re.search(r"(\d+)\s*만\s*(원|대)?", q)
        if m:
            target_price = int(m.group(1)) * 10000
    if target_price is None:
        m = re.search(r"(\d{2,6})\s*원\s*(대)?", q)
        if m:
            target_price = int(m.group(1))

    # ---- 시즌/기능 키워드 (간단 부스팅용)
    key_features = []
    if any(k in q for k in ["겨울","한파","보온","따뜻","패딩","다운"]): key_features.append("warm")
    if any(k in q for k in ["여름","통풍","시원","쿨","메쉬"]):           key_features.append("cool")
    if any(k in q for k in ["방수","비","눈","우천","워터프루프"]):        key_features.append("waterproof")
    if any(k in q for k in ["방풍","바람","윈드"]):                       key_features.append("windproof")
    if any(k in q for k in ["경량","라이트","가벼운","라이트웨이트"]):      key_features.append("light")

    return {
        "want_cats": want_cats,           # {"RUNNING", ...}
        "want_subs": want_subs,           # {"자켓","하의",...}
        "exclude_accessory": exclude_accessory,
        "price_min": price_min,
        "price_max": price_max,
        "target_price": target_price,
        "key_features": key_features,     # ["warm","waterproof",...]
    }
def _price_score(price: Optional[float], pmin, pmax, ptarget) -> float:
    """
    0~1 스코어. 예산 범위/타겟에 가까울수록 가점.
    """
    if price is None or pd.isna(price):
        return 0.0
    price = float(price)

    # 범위가 있으면 범위 내에서 1, 살짝 벗어나면 0.6~0.9로 부드럽게
    if pmin is not None or pmax is not None:
        if pmin is not None and price < pmin:
            gap = (pmin - price) / max(pmin, 1.0)
            return max(0.0, 1.0 - gap * 1.5)
        if pmax is not None and price > pmax:
            gap = (price - pmax) / max(pmax, 1.0)
            return max(0.0, 1.0 - gap * 1.5)
        return 1.0

    # 타겟 가격이 있으면 타겟에 가까울수록 높게 (가우시안 느낌)
    if ptarget is not None:
        dev = abs(price - ptarget) / max(ptarget, 1.0)
        return float(np.exp(-(dev**2) / 0.08))  # dev≈0.2 → 약 0.6

    return 0.0


def _keyword_boost(name: str, sub: str, features: List[str]) -> float:
    """
    시즌/기능 키워드가 제품명/서브카테고리에 드러나면 가점.
    """
    text = f"{name or ''} {sub or ''}".lower()
    score = 0.0
    if not features:
        return 0.0
    for f in features:
        if f == "warm" and any(k in text for k in ["warm","보온","다운","패딩","fleece","플리스"]):
            score += 0.15
        if f == "cool" and any(k in text for k in ["쿨","mesh","통풍","透氣","summer"]):
            score += 0.12
        if f == "waterproof" and any(k in text for k in ["waterproof","방수","gore","고어"]):
            score += 0.18
        if f == "windproof" and any(k in text for k in ["wind","방풍"]):
            score += 0.12
        if f == "light" and any(k in text for k in ["light","경량","ultra light"]):
            score += 0.10
    return min(score, 0.35)

# --- 제품 단위 검색: 유사도 + 카테고리/서브카테고리 가중치 + 평점 가점 ---
def search_products(
    query: str,
    top_k: int = 1,
    *,
    offset: int = 0,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    price_bias: Optional[str] = None,      # 'cheaper' | 'pricier' | None
    prefer: Optional[str] = None,          # 'popular' | 'value' | None
    prefer_similar: bool = False,
) -> List[Dict[str, Any]]:
    idx = _load_index()
    vec = idx["vectorizer"].transform([query])
    sims = cosine_similarity(vec, idx["matrix"])[0]

    # 문서 → 행
    rows = []
    for i, s in enumerate(sims):
        d = idx["docs"][int(i)]
        rows.append({
            "product_id": d["product_id"],
            "name": d["name"],
            "price": d["price"],
            "category": (d["category"] or "").upper(),
            "subcategory": (d["subcategory"] or ""),
            "rating_avg": d["rating_avg"],
            "review_count": d["review_count"],
            "url": d.get("url"),
            "thumbnail_url": d.get("thumbnail_url"),
            "snippet": d["text"][:220],
            "sim": float(s),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []

    # 선호/예산/키워드(질문 기반)
    prefs = _extract_preferences(query)
    want_cats: set = prefs.get("want_cats", set())
    want_subs: set = prefs.get("want_subs", set())
    exclude_acc: bool = prefs.get("exclude_accessory", False)

    # 액세서리 제외(의류 요청시)
    if exclude_acc:
        ban = ["장갑","모자","양말","액세서리","헤드밴드","워머","글러브","버프"]
        mask = ~(
            df["subcategory"].str.contains("|".join(ban), case=False, na=False) |
            df["name"].str.contains("|".join(ban), case=False, na=False)
        )
        df = df[mask]
        if df.empty:
            return []

    # 카테고리/서브카테고리 가중치
    def cat_boost(row):
        c = row["category"]
        s = (row["subcategory"] or "").lower()
        b = 0.0
        if want_cats:
            b += 0.6 if c in want_cats else 0.0
        if want_subs and any(ws.lower() in s for ws in want_subs):
            b += 0.6
        # 제목에 직접 포함되면 보너스
        if want_subs and any(ws.lower() in (row["name"] or "").lower() for ws in want_subs):
            b += 0.25
        # "비슷한" 요청이면 카테고리/서브카테고리 신뢰도 추가
        if prefer_similar:
            b += 0.2
        return min(b, 1.0)

    # 제품단위 집계
    df = df.sort_values("sim", ascending=False)
    agg = df.groupby("product_id", as_index=False).first()

    # 기본 점수 요소
    sim_score   = agg["sim"].astype(float)
    cat_score   = agg.apply(cat_boost, axis=1)
    rating_norm = agg["rating_avg"].fillna(0).clip(0, 5) / 5.0

    # 리뷰수 정규화
    rc = agg["review_count"].fillna(0)
    rc_norm = np.log1p(rc) / np.log(1 + max(1.0, rc.max()))

    # 가격 적합도(예산 있으면 가깝게, 없으면 0)
    def _price_fit_row(p):
        if pd.isna(p):
            return 0.0
        if min_price is None and max_price is None:
            return 0.0
        # 타겟 미지정: 범위 중앙을 선호
        lo = min_price if min_price is not None else p
        hi = max_price if max_price is not None else p
        mid = (float(lo) + float(hi)) / 2.0
        return float(np.exp(-abs(float(p) - mid) / max(1.0, mid)))  # 가운데 가까울수록 1에 근접
    price_fit = agg["price"].apply(_price_fit_row)

    # 가격 편향(“더 저렴/비싼”)
    price = agg["price"].fillna(agg["price"].median() if not agg["price"].dropna().empty else 0.0)
    if price_bias == "cheaper":
        # 더 저렴할수록 보너스 (가격 역정규화)
        pn = (price - price.min()) / max(1.0, (price.max() - price.min()))
        cheap_bonus = (1.0 - pn)  # 낮을수록 ↑
    elif price_bias == "pricier":
        pn = (price - price.min()) / max(1.0, (price.max() - price.min()))
        cheap_bonus = pn          # 높을수록 ↑
    else:
        cheap_bonus = pd.Series([0.0] * len(agg), index=agg.index)

    # 선호 모드 가중치 조정
    w_sim, w_cat, w_pricefit, w_rating, w_rc, w_pricebias, w_value = 0.50, 0.20, 0.12, 0.10, 0.05, 0.03, 0.00
    if prefer == "popular":
        w_rc += 0.07; w_sim -= 0.02
    if prefer == "value":
        # 가성비: 평점*리뷰수 대비 가격
        val = (rating_norm * (0.5 + 0.5 * rc_norm)) / (1e-9 + (price / max(1.0, price.median())))
        val = (val - val.min()) / max(1e-9, (val.max() - val.min()))
        value_term = val
        w_value = 0.12
    else:
        value_term = pd.Series([0.0] * len(agg), index=agg.index)

    final = (
        w_sim * sim_score +
        w_cat * cat_score +
        w_pricefit * price_fit +
        w_rating * rating_norm +
        w_rc * rc_norm +
        w_pricebias * cheap_bonus +
        w_value * value_term
    )

    # 서브카테고리 선호 패널티
    if want_subs:
        good = agg["subcategory"].str.lower().apply(lambda s: any(ws.lower() in (s or "") for ws in want_subs))
        final.loc[~good] *= 0.85

    agg["score"] = final

    # 예산 하드 필터
    if min_price is not None or max_price is not None:
        mask = pd.Series([True] * len(agg))
        if min_price is not None: mask &= (agg["price"].fillna(0) >= float(min_price) * 0.8)
        if max_price is not None: mask &= (agg["price"].fillna(9e12) <= float(max_price) * 1.2)
        agg = agg[mask]

    # 정렬 + 오프셋/리밋
    agg = agg.sort_values("score", ascending=False)
    if offset > 0:
        agg = agg.iloc[offset:]
    agg = agg.head(top_k)

    out = []
    for _, r in agg.iterrows():
        out.append({
            "product_id": r["product_id"],
            "name": r["name"],
            "price": int(r["price"]) if pd.notna(r["price"]) else None,
            "link": r["url"],
            "score": round(float(r["score"]), 4),
            "category": r["category"],
            "subcategory": r["subcategory"],
            "snippet": r["snippet"],
            "rating": float(r["rating_avg"]) if pd.notna(r["rating_avg"]) else None,
            "source": "reviews",
        })
    return out


def index_meta() -> Dict[str, Any]:
    idx = _load_index()
    return {
        "chunks": len(idx["docs"]),
        "built_at": idx.get("built_at"),
        "embed_model": idx.get("embed_model", "tfidf"),
        "path": PKL_PATH
    }

# 선택: 내보낼 심볼을 고정(오타/편집 실수 예방)
__all__ = ["build_index_from_db", "search_products", "index_meta"]