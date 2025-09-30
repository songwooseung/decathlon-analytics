from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any, List
import os, io, json, hashlib

import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# =========================
# 기본 설정 (PG/SQLite 겸용)
# =========================
load_dotenv()  # .env 로컬용 (배포에선 Render env 사용)
DB_URL = os.getenv("DATABASE_URL", "sqlite:///reviews.db")

# Postgres일 때 SSL/헬스체크 옵션
from sqlalchemy import create_engine
connect_args = {}
if DB_URL.startswith("postgresql"):
    # Render/Neon 등은 sslmode=require가 보통 접속문자열에 포함됨.
    # 그래도 안전하게 pool_pre_ping으로 헬스체크.
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
else:
    # SQLite (로컬 개발 기본)
    engine = create_engine(DB_URL, future=True)

# FastAPI 앱 생성 (기존 그대로)
app = FastAPI(
    title="Decathlon Review Analytics",
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
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
        import pandas as pd
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

# =========================
# 테이블 스키마 생성 (PG 친화 타입)
# =========================
def ensure_tables():
    with engine.begin() as conn:
        # 리뷰 원천
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id     VARCHAR PRIMARY KEY,
            product_id    VARCHAR,
            product_name  VARCHAR,
            category      VARCHAR,
            rating        FLOAT,
            review_text   TEXT,
            review_date   TIMESTAMP,   -- PG: TIMESTAMP(또는 DATE)
            helpful_votes INT,
            image_urls    TEXT,        -- JSON 배열 문자열
            review_url    TEXT
        );
        """))

        # 제품 요약(집계)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS product_summary (
            product_id        VARCHAR PRIMARY KEY,
            product_name      VARCHAR,
            category          VARCHAR,
            brand             VARCHAR,
            total_reviews     INT,
            positive_reviews  INT,
            mixed_reviews     INT,
            negative_reviews  INT,
            avg_rating        FLOAT,
            updated_at        TIMESTAMP
        );
        """))

# =========================
# CSV 컬럼 매핑(유연 매핑)
# =========================
REVIEW_COL_CANDIDATES: Dict[str, List[str]] = {
    "review_id":     ["review_id", "id", "rv_id"],
    "product_id":    ["product_id", "pid", "p_id"],
    "product_name":  ["product_name", "product", "name", "title"],
    "category":      ["category", "cat"],
    "rating":        ["rating", "score", "stars"],
    "review_text":   ["review_text", "content", "body", "excerpt"],
    "review_date":   ["review_date", "date", "created_at", "createdAt"],
    "helpful_votes": ["helpful_votes", "helpful", "helpfulCount", "likes"],
    "image_urls":    ["image_urls", "images", "image", "imgs"],
    "review_url":    ["review_url", "url", "link"],
}

SUMMARY_COL_CANDIDATES: Dict[str, List[str]] = {
    "product_id":       ["product_id", "pid", "p_id"],
    "product_name":     ["product_name", "name", "title"],
    "category":         ["category", "cat"],
    "brand":            ["brand", "product_brand", "maker"],
    "total_reviews":    ["total_reviews", "reviews_total", "review_count"],
    "positive_reviews": ["positive_reviews", "pos_reviews", "positive"],
    "mixed_reviews":    ["mixed_reviews", "mix_reviews", "neutral", "complex"],
    "negative_reviews": ["negative_reviews", "neg_reviews", "negative"],
    "avg_rating":       ["avg_rating", "rating_avg", "avgScore"],
    "updated_at":       ["updated_at", "as_of", "refreshed_at"],
}

def map_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for std, candidates in mapping.items():
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand; break
            if cand.lower() in lower_cols:
                found = lower_cols[cand.lower()]; break
        if found:
            rename_map[found] = std
        else:
            df[std] = None
    df = df.rename(columns=rename_map)
    keep = list(mapping.keys())
    for k in keep:
        if k not in df.columns:
            df[k] = None
    return df[keep]

def parse_date_any(x) -> Optional[str]:
    if pd.isna(x) or x is None: return None
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

# =========================
# 카테고리 표준화 & 라벨
# =========================
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
# Ingest (경로/업로드)
# =========================


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

def _ingest_reviews_df(df: pd.DataFrame):
    ensure_tables()
    df = map_columns(df, REVIEW_COL_CANDIDATES)

   # 타입/정규화
    df["product_id"] = df["product_id"].apply(safe_str)
    df.loc[df["product_id"] == "", "product_id"] = "unknown"

    df["review_text"]  = df["review_text"].apply(safe_str)
    df["review_date"]  = df["review_date"].apply(parse_date_any)
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"]= pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["image_urls"]   = df["image_urls"].apply(to_json_array)
    df["category"]     = df["category"].apply(normalize_category)

    # PK 채우기(결정적 생성)
    need_id = df["review_id"].isna() | (df["review_id"].apply(safe_str) == "")
    df.loc[need_id, "review_id"] = df[need_id].apply(
        lambda r: deterministic_review_id(r.get("product_id"), r.get("review_date"), r.get("review_text")), axis=1
    )

    # 중복 제거
    df = df.drop_duplicates(subset=["review_id"])

    with engine.begin() as conn:
        # 기존 키 제외 후 append (간단 upsert)
        existing = pd.read_sql(text("SELECT review_id FROM reviews;"), conn)
        if not existing.empty:
            df = df[~df["review_id"].isin(set(existing["review_id"].tolist()))]
        df.to_sql("reviews", con=conn, if_exists="append", index=False)

    return {"ok": True, "inserted": int(len(df))}

def _ingest_summary_df(df: pd.DataFrame):
    ensure_tables()
    df = map_columns(df, SUMMARY_COL_CANDIDATES)

    # product_id를 문자열로 강제
    df["product_id"] = df["product_id"].apply(safe_str)
    df = df[df["product_id"] != ""]   # 빈값은 제거

    # 타입/정규화
    for c in ["total_reviews","positive_reviews","mixed_reviews","negative_reviews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["updated_at"] = df["updated_at"].fillna(now_iso())
    df["category"]   = df["category"].apply(normalize_category)

    # 같은 product_id는 최신 updated_at 1개만
    df = df.sort_values(by=["product_id","updated_at"]).drop_duplicates(subset=["product_id"], keep="last")

    with engine.begin() as conn:
        ids = [str(x) for x in df["product_id"].dropna().unique().tolist()]  # ✅ 문자열 리스트로!
        if ids:
            placeholders = ",".join([f":id{i}" for i in range(len(ids))])
            q = text(f"DELETE FROM product_summary WHERE product_id IN ({placeholders});")
            params = {f"id{i}": ids[i] for i in range(len(ids))}              # ✅ 문자열 파라미터
            conn.execute(q, params)

        df.to_sql("product_summary", con=conn, if_exists="append", index=False)

    return {"ok": True, "upserted": int(len(df))}

# =========================
# 카드 빌더 (합의된 JSON 포맷)
# =========================
def build_product_card(product_id: str) -> Optional[Dict[str, Any]]:
    # 1) summary (필수 정보: avg_rating, total/positive, category, brand, product_name)
    srow = None
    with engine.begin() as conn:
        srow = conn.execute(text("""
            SELECT product_id, product_name, category, brand, total_reviews, positive_reviews, avg_rating
            FROM product_summary
            WHERE product_id = :pid
        """), {"pid": product_id}).mappings().first()

    if not srow:
        return None

    # 2) 대표 리뷰 1건 (rating DESC -> helpful_votes DESC -> review_date DESC)
    rrow = None
    with engine.begin() as conn:
        rrow = conn.execute(text("""
            SELECT review_id, product_id, product_name, category, rating, review_text, review_date, helpful_votes, image_urls, review_url
            FROM reviews
            WHERE product_id = :pid
            ORDER BY rating DESC, helpful_votes DESC, review_date DESC
            LIMIT 1
        """), {"pid": product_id}).mappings().first()

    # 대표 리뷰 정보 (텍스트/리뷰 링크/이미지)
    review_id = None
    review_text = None
    review_url = None
    images = []
    if rrow:
        review_id   = rrow["review_id"]
        review_text = rrow["review_text"]
        review_url  = rrow["review_url"]
        try:
            images = json.loads(rrow["image_urls"] or "[]")
        except Exception:
            images = []

    # 응답 카드(최종 스펙)
    return {
        "reviewId": review_id,                         # 대표 리뷰 id (없으면 null)
        "productId": srow["product_id"],
        "category": category_korean_label(srow["category"]),
        "brand": srow.get("brand"),
        "productName": srow["product_name"],
        "rating": srow["avg_rating"],                  # 항상 summary.avg_rating
        "review_text": review_text,
        "total_reviews": srow["total_reviews"],
        "positive_reviews": srow["positive_reviews"],
        "images": images or [],                        # 현재 미수집이면 []
        "url": review_url
    }

# =========================
# TOTAL 섹션
# =========================
@app.get("/total/best-review")
def total_best_review(limit: int = 5, min_reviews: int = 1):
    """
    베스트 리뷰 Top N: positive_reviews DESC (tie: avg_rating DESC -> total_reviews DESC)
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT product_id
            FROM product_summary
            WHERE total_reviews >= :min_reviews
            ORDER BY positive_reviews DESC, avg_rating DESC, total_reviews DESC
            LIMIT :limit
        """), {"min_reviews": min_reviews, "limit": limit}).mappings().all()

    items = []
    for r in rows:
        card = build_product_card(r["product_id"])
        if card: items.append(card)
    return {"data": {"bestReviews": items}, "meta": {"generatedAt": now_iso()}}

@app.get("/total/top-by-reviewcount")
def total_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    """
    제일 잘 팔린 Top N(프록시=누적 리뷰수): total_reviews DESC (tie: positive_reviews DESC -> avg_rating DESC)
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT product_id
            FROM product_summary
            WHERE total_reviews >= :min_reviews
            ORDER BY total_reviews DESC, positive_reviews DESC, avg_rating DESC
            LIMIT :limit
        """), {"min_reviews": min_reviews, "limit": limit}).mappings().all()

    items = []
    for r in rows:
        card = build_product_card(r["product_id"])
        if card: items.append(card)
    return {"data": {"topByReviewCount": items}, "meta": {"generatedAt": now_iso()}}

@app.get("/total/top-this-month-one")
def total_top_this_month_one():
    """
    최근 30일 리뷰 최다 상품 1개 (tie: avg_rating DESC -> positive_reviews DESC)
    """
    cutoff = datetime.now(UTC) - timedelta(days=30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # 최근 30일 product_id별 리뷰 수
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

    card = build_product_card(row["product_id"])
    return {"data": {"monthlyTopOne": card}, "meta": {"generatedAt": now_iso(), "range": "last_30d"}}

# =========================
# 카테고리 섹션 – 고정 라우트
# =========================
def _category_top_rated(std_cat: str, limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT product_id
            FROM product_summary
            WHERE category = :cat AND total_reviews >= :min_reviews
            ORDER BY avg_rating DESC, positive_reviews DESC, total_reviews DESC
            LIMIT :limit
        """), {"cat": std_cat, "min_reviews": min_reviews, "limit": limit}).mappings().all()
    items = []
    for r in rows:
        card = build_product_card(r["product_id"])
        if card: items.append(card)
    return items

def _category_top_by_reviewcount(std_cat: str, limit: int = 5, min_reviews: int = 1):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT product_id
            FROM product_summary
            WHERE category = :cat AND total_reviews >= :min_reviews
            ORDER BY total_reviews DESC, positive_reviews DESC, avg_rating DESC
            LIMIT :limit
        """), {"cat": std_cat, "min_reviews": min_reviews, "limit": limit}).mappings().all()
    items = []
    for r in rows:
        card = build_product_card(r["product_id"])
        if card: items.append(card)
    return items

@app.get("/running/top-rated")
def running_top_rated(limit: int = 5, min_reviews: int = 1):
    items = _category_top_rated("RUNNING", limit, min_reviews)
    return {"data": {"topRated": items}, "meta": {"category": "러닝", "generatedAt": now_iso()}}

@app.get("/running/top-by-reviewcount")
def running_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    items = _category_top_by_reviewcount("RUNNING", limit, min_reviews)
    return {"data": {"topByReviewCount": items}, "meta": {"category": "러닝", "generatedAt": now_iso()}}

@app.get("/hiking/top-rated")
def hiking_top_rated(limit: int = 5, min_reviews: int = 1):
    items = _category_top_rated("HIKING", limit, min_reviews)
    return {"data": {"topRated": items}, "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}

@app.get("/hiking/top-by-reviewcount")
def hiking_top_by_reviewcount(limit: int = 5, min_reviews: int = 1):
    items = _category_top_by_reviewcount("HIKING", limit, min_reviews)
    return {"data": {"topByReviewCount": items}, "meta": {"category": "등산/하이킹", "generatedAt": now_iso()}}

# 선택: 루트 상태 체크
@app.get("/")
def root():
    return {"ok": True, "service": "Decathlon Review Analytics", "time": now_iso()}


# db 값 전체 출력
@app.get("/debug/reviews")
def list_reviews(limit: Optional[int] = None):
    sql = "SELECT * FROM reviews"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"
        params["lim"] = int(limit)

    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}

@app.get("/debug/summary")
def list_summary(limit: Optional[int] = None):
    sql = "SELECT * FROM product_summary"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"
        params["lim"] = int(limit)

    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}