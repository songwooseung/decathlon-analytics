# main.py
from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any, List
import os, io, json, hashlib

import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# =========================
# 기본 설정 (Postgres/SQLite 겸용)
# =========================
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///reviews.db")

if DB_URL.startswith("postgresql"):
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
else:
    engine = create_engine(DB_URL, future=True)

app = FastAPI(
    title="Decathlon Review Analytics",
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def parse_date_any(x) -> Optional[str]:
    if x is None: return None
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d): return None
        return d.strftime("%Y-%m-%d")
    except Exception:
        return None

def to_json_array(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return "[]"
    if isinstance(val, list): return json.dumps(val, ensure_ascii=False)
    s = str(val).strip()
    if not s: return "[]"
    if s.startswith("[") and s.endswith("]"): return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return json.dumps(parts, ensure_ascii=False)

def deterministic_review_id(product_id, review_date, review_text) -> str:
    pid = safe_str(product_id) or "unknown"
    d   = safe_str(review_date) or "0000-00-00"
    t   = safe_str(review_text)
    t_norm = " ".join(t.split())
    h = hashlib.sha256(f"{pid}|{d}|{t_norm}".encode("utf-8")).hexdigest()[:16]
    return f"{pid}|{d}|{h}"

def normalize_category(s: Optional[str]) -> Optional[str]:
    if not s: return None
    z = str(s).strip().lower()
    if z in {"running", "run", "러닝"}: return "RUNNING"
    if z in {"hiking", "trekking", "등산", "하이킹", "등산/하이킹"}: return "HIKING"
    return z.upper()

def category_korean_label(std_cat: Optional[str]) -> Optional[str]:
    if not std_cat: return None
    return {"RUNNING":"러닝", "HIKING":"등산/하이킹"}.get(std_cat, std_cat)

# =========================
# 스키마 (CSV 있는 컬럼 그대로 수용)
# =========================
def ensure_tables():
    with engine.begin() as conn:
        # reviews: complete.csv 컬럼을 그대로 수용(+몇개는 NULL 허용)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id     VARCHAR PRIMARY KEY,
            product_id    VARCHAR,
            product_name  VARCHAR,
            category      VARCHAR,
            subcategory   VARCHAR,
            brand         VARCHAR,
            rating        FLOAT,
            review_text   TEXT,
            sentiment     VARCHAR,
            review_date   TIMESTAMP
        );
        """))

        # product_summary: summary.csv의 컬럼 전부 포함
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS product_summary (
            product_id        VARCHAR PRIMARY KEY,
            product_name      VARCHAR,
            category          VARCHAR,
            subcategory       VARCHAR,
            brand             VARCHAR,
            price             INT,
            total_reviews     INT,
            positive_reviews  INT,
            mixed_reviews     INT,
            negative_reviews  INT,
            avg_rating        FLOAT,
            url               TEXT,
            thumbnail_url     TEXT,
            updated_at        TIMESTAMP
        );
        """))

# =========================
# 컬럼 매핑 (CSV 원본 이름 중심)
# =========================
REVIEW_MAP = {
    "product_id":   ["product_id","pid","p_id"],
    "product_name": ["product_name","name","title","product"],
    "category":     ["category","cat"],
    "subcategory":  ["subcategory","sub_cat","subCategory"],
    "brand":        ["brand","maker","product_brand"],
    "rating":       ["rating","score","stars"],
    "review_text":  ["review_text","content","body","excerpt"],
    "sentiment":    ["sentiment"],
    "review_date":  ["date","review_date","created_at","createdAt"],
}

SUMMARY_MAP = {
    "product_id":       ["product_id","pid","p_id"],
    "product_name":     ["product_name","name","title"],
    "category":         ["category","cat"],
    "subcategory":      ["subcategory","sub_cat","subCategory"],
    "brand":            ["brand","maker","product_brand"],
    "price":            ["price","가격","product_price"],
    "total_reviews":    ["total_reviews","review_count","reviews_total"],
    "positive_reviews": ["positive_reviews","positive"],
    "mixed_reviews":    ["mixed_reviews","neutral","mixed"],
    "negative_reviews": ["negative_reviews","negative"],
    "avg_rating":       ["avg_rating","rating_avg","avgScore"],
    "url":              ["url","product_url"],
    "thumbnail_url":    ["thumbnail_url","thumbnail","thumb_url","image","image_url"],
    "updated_at":       ["updated_at","as_of","refreshed_at"],
}

def map_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    for std, cands in mapping.items():
        src = None
        for c in cands:
            if c in df.columns:
                src = c; break
            if c.lower() in lower:
                src = lower[c.lower()]; break
        if src:
            rename_map[src] = std
        else:
            df[std] = None
    df = df.rename(columns=rename_map)
    # keep only mapped keys (but ensure all std keys exist)
    cols = list(mapping.keys())
    for k in cols:
        if k not in df.columns:
            df[k] = None
    return df[cols]

# =========================
# Ingest
# =========================
@app.post("/ingest/reviews/file")
async def ingest_reviews_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return _ingest_reviews_df(df)

@app.post("/ingest/summary/file")
async def ingest_summary_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return _ingest_summary_df(df)

@app.get("/ingest/reviews")
def ingest_reviews(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(400, f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return _ingest_reviews_df(df)

@app.get("/ingest/summary")
def ingest_summary(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(400, f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return _ingest_summary_df(df)

def _ingest_reviews_df(df: pd.DataFrame):
    ensure_tables()
    df = map_columns(df, REVIEW_MAP)

    # 최소 가공(타입 변환만)
    df["product_id"]   = df["product_id"].apply(safe_str)
    df["product_name"] = df["product_name"].apply(safe_str)
    df["category"]     = df["category"].apply(normalize_category)
    df["subcategory"]  = df["subcategory"].apply(safe_str)
    df["brand"]        = df["brand"].apply(safe_str)
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"]  = df["review_text"].apply(safe_str)
    df["sentiment"]    = df["sentiment"].apply(safe_str)
    df["review_date"]  = df["review_date"].apply(parse_date_any)

    # 결정적 review_id 생성
    df["review_id"] = df.apply(lambda r: deterministic_review_id(
        r.get("product_id"), r.get("review_date"), r.get("review_text")), axis=1)

    # 업로드 파일 내부 중복 제거
    df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    with engine.begin() as conn:
        # DB에 이미 있는 키 제외
        existing = pd.read_sql(text("SELECT review_id FROM reviews;"), conn)
        if not existing.empty:
            have = set(existing["review_id"].astype(str).tolist())
            df = df[~df["review_id"].astype(str).isin(have)]

        # 성능/안정용 옵션(선택)
        df.to_sql("reviews", con=conn, if_exists="append", index=False, method="multi", chunksize=1000)

    return {"ok": True, "inserted": int(len(df))}

def _ingest_summary_df(df: pd.DataFrame):
    ensure_tables()
    df = map_columns(df, SUMMARY_MAP)

    # 타입 변환만(가공 최소)
    df["product_id"]   = df["product_id"].apply(safe_str)
    df = df[df["product_id"] != ""]
    df["product_name"] = df["product_name"].apply(safe_str)
    df["category"]     = df["category"].apply(normalize_category)
    df["subcategory"]  = df["subcategory"].apply(safe_str)
    df["brand"]        = df["brand"].apply(safe_str)
    df["price"]        = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(int)
    
    for c in ["total_reviews","positive_reviews","mixed_reviews","negative_reviews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["avg_rating"]    = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["url"]           = df["url"].apply(safe_str)
    df["thumbnail_url"] = df["thumbnail_url"].apply(safe_str)
    df["updated_at"]    = df["updated_at"].fillna(now_iso())

    # 최신만 유지
    df = df.sort_values(by=["product_id","updated_at"]).drop_duplicates(subset=["product_id"], keep="last")

    with engine.begin() as conn:
        ids = df["product_id"].dropna().astype(str).unique().tolist()
        if ids:
            placeholders = ",".join([f":id{i}" for i in range(len(ids))])
            conn.execute(text(f"DELETE FROM product_summary WHERE product_id IN ({placeholders});"),
                         {f"id{i}": ids[i] for i in range(len(ids))})
        df.to_sql("product_summary", con=conn, if_exists="append", index=False)

    return {"ok": True, "upserted": int(len(df))}

# =========================
# 카드 빌더 (최종 포맷)
# =========================
def build_product_card(product_id: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        srow = conn.execute(text("""
            SELECT product_id, product_name, category, subcategory, brand, price,
                   total_reviews, positive_reviews, avg_rating, url, thumbnail_url
            FROM product_summary
            WHERE product_id = :pid
        """), {"pid": product_id}).mappings().first()

    if not srow:
        return None

    # 대표 리뷰 1건: rating 높은 순 → review_date 최신 순
    with engine.begin() as conn:
        rrow = conn.execute(text("""
            SELECT review_id, review_text, rating
            FROM reviews
            WHERE product_id = :pid
            ORDER BY rating DESC, review_date DESC
            LIMIT 1
        """), {"pid": product_id}).mappings().first()

    review_id   = rrow["review_id"]   if rrow else None
    review_text = rrow["review_text"] if rrow else None

    # 대표 이미지 (summary.thumbnail_url → 없으면 "")
    image_url = srow.get("thumbnail_url") or ""

    return {
        "reviewId": review_id,
        "productId": srow["product_id"],
        "category": category_korean_label(srow["category"]),
        "subcategory": srow.get("subcategory"),
        "brand": srow.get("brand"),
        "productName": srow["product_name"],
        "price": srow.get("price"),
        "rating": srow["avg_rating"],
        "review_text": review_text,
        "total_reviews": srow["total_reviews"],
        "positive_reviews": srow["positive_reviews"],
        "images": image_url,    # 문자열 하나
        "url": srow["url"] or "",
    }

# =========================
# TOTAL / CATEGORY 엔드포인트
# =========================
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

@app.get("/total/best-review")
def total_best_review(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "total_reviews >= :min_reviews",
            {"min_reviews": min_reviews},
            "positive_reviews DESC, avg_rating DESC, total_reviews DESC",
            limit
        )
    return {"data": {"bestReviews": items}, "meta": {"generatedAt": now_iso()}}

@app.get("/total/top-by-reviewcount")
def total_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        items = _cat_top_by(conn,
            "total_reviews >= :min_reviews",
            {"min_reviews": min_reviews},
            "total_reviews DESC, positive_reviews DESC, avg_rating DESC",
            limit
        )
    return {"data": {"topByReviewCount": items}, "meta": {"generatedAt": now_iso()}}

@app.get("/total/top-this-month-one")
def total_top_this_month_one():
    cutoff = datetime.now(UTC) - timedelta(days=30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
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
        """), {"cutoff": cutoff_str}).mappings().first()
    if not row:
        return {"data": {"monthlyTopOne": None}, "meta": {"generatedAt": now_iso(), "range": "last_30d"}}
    return {"data": {"monthlyTopOne": build_product_card(row["product_id"])},
            "meta": {"generatedAt": now_iso(), "range": "last_30d"}}

def _category_top_rated(std_cat: str, limit: int, min_reviews: int):
    with engine.begin() as conn:
        return _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": std_cat, "min_reviews": min_reviews},
            "avg_rating DESC, positive_reviews DESC, total_reviews DESC",
            limit
        )

def _category_top_by_reviewcount(std_cat: str, limit: int, min_reviews: int):
    with engine.begin() as conn:
        return _cat_top_by(conn,
            "category = :cat AND total_reviews >= :min_reviews",
            {"cat": std_cat, "min_reviews": min_reviews},
            "total_reviews DESC, positive_reviews DESC, avg_rating DESC",
            limit
        )

@app.get("/running/top-rated")
def running_top_rated(limit: int = 5, min_reviews: int = 1):
    return {"data": {"topRated": _category_top_rated("RUNNING", limit, min_reviews)},
            "meta": {"category": "러닝", "generatedAt": now_iso()}}

@app.get("/running/top-by-reviewcount")
def running_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    return {"data": {"topByReviewCount": _category_top_by_reviewcount("RUNNING", limit, min_reviews)},
            "meta": {"category": "러닝", "generatedAt": now_iso()}}

@app.get("/hiking/top-rated")
def hiking_top_rated(limit: int = 5, min_reviews: int = 1):
    return {"data": {"topRated": _category_top_rated("HIKING", limit, min_reviews)},
            "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}

@app.get("/hiking/top-by-reviewcount")
def hiking_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    return {"data": {"topByReviewCount": _category_top_by_reviewcount("HIKING", limit, min_reviews)},
            "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}

# 세부 카테고리 Top
@app.get("/running/top-by-subcategory")
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

@app.get("/hiking/top-by-subcategory")
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

# 디버그
@app.get("/debug/reviews")
def list_reviews(limit: Optional[int] = None):
    sql = "SELECT * FROM reviews"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"; params["lim"] = int(limit)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}

@app.get("/debug/summary")
def list_summary(limit: Optional[int] = None):
    sql = "SELECT * FROM product_summary"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"; params["lim"] = int(limit)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}

@app.get("/")
def root():
    return {"ok": True, "service": "Decathlon Review Analytics", "time": now_iso()}