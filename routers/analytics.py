import pandas as pd
import re
import math 
from datetime import datetime, UTC, timedelta
from fastapi import APIRouter, Query
from sqlalchemy import text
from core.db import engine
from services.cards import build_product_card
from services.utils import now_iso

CATEGORY_MAP = {
    "RUNNING":"RUNNING","running":"RUNNING","run":"RUNNING","러닝":"RUNNING",
    "HIKING":"HIKING","hiking":"HIKING","trekking":"HIKING","등산":"HIKING","하이킹":"HIKING","등산/하이킹":"HIKING","스포츠":"HIKING"
}

def _norm_cat(cat: str|None) -> str|None:
    if not cat: return None
    return CATEGORY_MAP.get(str(cat).strip(), None)

TOKEN_RE = re.compile(r"[^\w가-힣]+", re.UNICODE)

STOPWORDS = set("""
그리고 그러나 그런데 그래서 또한 하지만 너무 매우 좀 역시 그냥 정말 진짜
사용 사용중 사용감 제품 상품 물건 데카트론 데카 제품명 모델 모델명 가격
구매 구매후 후기 리뷰 리뷰어 장바구니 옵션 쿠폰 사이즈 선택
""".split())

# 광고/메뉴/가격/URL/빵부스러기 제거용
AD_NOISE_RE = re.compile("|".join([
    r"쿠폰 ?코드", r"전품목", r"장바구니 ?담기", r"사이즈(를|는)? ?선택",
    r"판매자\s*DECATHLON", r"배송 옵션", r"매장 이용", r"내\s*사이즈\s*찾기",
    r"모델번호", r"%의 고객이.*추천", r"홈>모든\s*스포츠", r"홈\s*>\s*[^>]+",
    r"https?://\S+", r"\d{1,3}(?:,\d{3})+원", r"\d+\s?원"
]), re.I)

# 흔한 불용어 조금만 보강
STOPWORDS |= {
    "입니다","있습니다","합니다","같아요","같습니다","좋아요","좋습니다","좋네요","정말",
    "진짜","조금","약간","살짝","그냥","다만","그리고","하지만","또한","또",
    "작성하기","고객이","별점","신제품","자세히","알아보기","모든","제품을","담기",
    "가격이","가격도","가격에","만원",
}

def _preclean_text(s: str) -> str:
    s = s or ""
    s = AD_NOISE_RE.sub(" ", s)     # 광고/가격/URL/빵부스러기 컷
    s = TOKEN_RE.sub(" ", s).strip()
    return s

def _tokenize_ko(s: str) -> list[str]:
    s = _preclean_text(s)                    # <-- 이 한 줄만 추가
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) >= 2]
    toks = [t for t in toks if not re.fullmatch(r"[0-9A-Za-z]+", t)]
    return toks

def _bin_label(v: float, bin_size: int) -> str:
    b = int(v // bin_size)
    lo, hi = b*bin_size, b*bin_size + bin_size - 1
    return f"{lo:,}-{hi:,}"

router = APIRouter(prefix="", tags=["analytics"])

def _cat_top_by(conn, where: str, params: dict, order: str, limit: int):
    rows = conn.execute(text(f"""
        SELECT product_id
        FROM product_summary
        WHERE {where}
        ORDER BY {order}
        LIMIT :limit
    """), {**params, "limit": limit}).mappings().all()
    items = []
    for r in rows:
        card = build_product_card(r["product_id"])
        if card: items.append(card)
    return items

@router.get("/total/best-review")
def total_best_review(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "total_reviews >= :min_reviews",
            {"min_reviews": min_reviews},
            "positive_reviews DESC, avg_rating DESC, total_reviews DESC",
            limit
        )
    return {"data": {"bestReviews": items}, "meta": {"generatedAt": now_iso()}}

@router.get("/total/top-by-reviewcount")
def total_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "total_reviews >= :min_reviews",
            {"min_reviews": min_reviews},
            "total_reviews DESC, positive_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topByReviewCount": items}, "meta": {"generatedAt": now_iso()}}

@router.get("/total/top-this-month-one")
def total_top_this_month_one():
    cutoff = datetime.now(UTC) - timedelta(days=30)
    with engine.begin() as conn:
        row = conn.execute(text("""
            WITH recent AS (
                SELECT product_id, COUNT(*) AS cnt
                FROM reviews
                WHERE review_date >= :cutoff
                GROUP BY product_id
            )
            SELECT r.product_id
            FROM recent r
            LEFT JOIN product_summary s ON s.product_id = r.product_id
            ORDER BY r.cnt DESC, s.avg_rating DESC, s.positive_reviews DESC
            LIMIT 1
        """), {"cutoff": cutoff.strftime("%Y-%m-%d")}).mappings().first()
    if not row:
        return {"data": {"monthlyTopOne": None}, "meta": {"generatedAt": now_iso(), "range": "last_30d"}}
    return {"data": {"monthlyTopOne": build_product_card(row["product_id"])},
            "meta": {"generatedAt": now_iso(), "range": "last_30d"}}

@router.get("/running/top-rated")
def running_top_rated(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": "RUNNING", "min_reviews": min_reviews},
            "avg_rating DESC, positive_reviews DESC, total_reviews DESC",
            limit
        )
    return {"data": {"topRated": items}, "meta": {"category": "러닝", "generatedAt": now_iso()}}

@router.get("/running/top-by-reviewcount")
def running_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": "RUNNING", "min_reviews": min_reviews},
            "total_reviews DESC, positive_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topByReviewCount": items}, "meta": {"category": "러닝", "generatedAt": now_iso()}}

@router.get("/running/top-by-subcategory")
def running_top_by_subcategory(subcategory: str = Query(...), limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category='RUNNING' AND subcategory=:sub AND total_reviews >= :min_reviews",
            {"sub": subcategory, "min_reviews": min_reviews},
            "total_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topBySubcategory": items},
            "meta": {"category":"러닝","subcategory": subcategory,"generatedAt": now_iso()}}


@router.get("/hiking/top-rated")
def hiking_top_rated(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": "HIKING", "min_reviews": min_reviews},
            "avg_rating DESC, positive_reviews DESC, total_reviews DESC",
            limit
        )
    return {"data": {"topRated": items}, "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}

@router.get("/hiking/top-by-reviewcount")
def hiking_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": "HIKING", "min_reviews": min_reviews},
            "total_reviews DESC, positive_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topByReviewCount": items}, "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}


@router.get("/hiking/top-by-subcategory")
def hiking_top_by_subcategory(subcategory: str = Query(...), limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "category='HIKING' AND subcategory=:sub AND total_reviews >= :min_reviews",
            {"sub": subcategory, "min_reviews": min_reviews},
            "total_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topBySubcategory": items},
            "meta": {"category":"등산/하이킹","subcategory": subcategory,"generatedAt": now_iso()}}


# ---- TOTAL  ---- -> 여기서 부터 차트용 api

@router.get("/total/price-bins")
def total_price_bins(bin_size: int = 20000):
    """가격대별 평균 평점/평균 리뷰수/상품수"""
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT price, avg_rating, total_reviews
            FROM product_summary
            WHERE price IS NOT NULL
        """), conn)

    if df.empty:
        return {"data": {"bins": []}, "meta": {"generatedAt": now_iso(), "binSize": bin_size}}

    df["bin"] = df["price"].astype(float).apply(lambda v: _bin_label(v, bin_size))
    grp = (
        df.groupby("bin")
          .agg(avg_rating=("avg_rating","mean"),
               avg_total_reviews=("total_reviews","mean"),
               n_products=("price","count"))
          .reset_index()
    )
    grp = grp.sort_values("bin", key=lambda s: s.str.extract(r"^(\d+)", expand=False).astype(int))

    bins = []
    for _, r in grp.iterrows():
        ar = None if pd.isna(r["avg_rating"]) else round(float(r["avg_rating"]), 3)
        if pd.isna(r["avg_total_reviews"]):
            atr = 0
        else:
            atr = int(math.floor(float(r["avg_total_reviews"])))  # <-- 내림 정수
        bins.append({
            "bin_label": r["bin"],
            "avg_rating": ar,
            "avg_total_reviews": atr,   # 정수
            "n_products": int(r["n_products"]),
        })
    return {"data": {"bins": bins}, "meta": {"generatedAt": now_iso(), "binSize": bin_size}}

@router.get("/total/monthly-reviews")
def total_monthly_reviews():
    """월별 리뷰 갯수(시즌성)"""
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT review_date FROM reviews"), conn)

    if df.empty:
        return {"data": {"series": []}, "meta": {"generatedAt": now_iso()}}

    ds = pd.to_datetime(df["review_date"], errors="coerce")
    mon = ds.dt.strftime("%Y-%m")
    grp = (mon.value_counts().rename_axis("month").reset_index(name="count").sort_values("month"))
    series = [{"month": m, "count": int(c)} for m, c in zip(grp["month"], grp["count"])]

    return {"data": {"series": series}, "meta": {"generatedAt": now_iso()}}

def _price_distribution_for(cat_std: str, bin_size: int = 20000):
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT price
            FROM product_summary
            WHERE category = :cat AND price IS NOT NULL
        """), conn, params={"cat": cat_std})

    if df.empty:
        return {"prices": [], "hist": []}

    prices = [int(p) for p in df["price"].astype(float).tolist()]
    df["bin"] = df["price"].astype(float).apply(lambda v: _bin_label(v, bin_size))
    hist = (df.groupby("bin").size()
              .reset_index(name="count")
              .sort_values("bin", key=lambda s: s.str.extract(r"^(\d+)", expand=False).astype(int)))
    hist_json = [{"bin_label": r["bin"], "count": int(r["count"])} for _, r in hist.iterrows()]
    return {"prices": prices, "hist": hist_json}

def _wordcloud_for(cat_std: str, top_n: int = 100):
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT r.review_text
            FROM reviews r
            JOIN product_summary p ON p.product_id = r.product_id
            WHERE p.category = :cat
        """), conn, params={"cat": cat_std})

    if df.empty:
        return {"words": []}

    counts: dict[str,int] = {}
    for t in df["review_text"].fillna("").astype(str):
        for w in _tokenize_ko(t):
            counts[w] = counts.get(w, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {"words": [{"text": k, "count": int(v)} for k, v in items]}

# ---- RUNNING ----
@router.get("/running/price-distribution")
def running_price_distribution(bin_size: int = 20000):
    data = _price_distribution_for("RUNNING", bin_size)
    return {"data": data, "meta": {"generatedAt": now_iso(), "category": "러닝", "binSize": bin_size}}

@router.get("/running/wordcloud")
def running_wordcloud(top_n: int = 100):
    data = _wordcloud_for("RUNNING", top_n)
    return {"data": data, "meta": {"generatedAt": now_iso(), "category": "러닝"}}

# ---- HIKING ----
@router.get("/hiking/price-distribution")
def hiking_price_distribution(bin_size: int = 20000):
    data = _price_distribution_for("HIKING", bin_size)
    return {"data": data, "meta": {"generatedAt": now_iso(), "category": "등산/하이킹", "binSize": bin_size}}

@router.get("/hiking/wordcloud")
def hiking_wordcloud(top_n: int = 100):
    data = _wordcloud_for("HIKING", top_n)
    return {"data": data, "meta": {"generatedAt": now_iso(), "category": "등산/하이킹"}}