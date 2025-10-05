from typing import Dict, List
import io, os, json
import pandas as pd
from sqlalchemy import text
from core.db import engine, ensure_tables
from services.utils import safe_str, parse_date_any, deterministic_review_id, normalize_category, now_iso

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

def ingest_reviews_df(df: pd.DataFrame):
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

def ingest_summary_df(df: pd.DataFrame):
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