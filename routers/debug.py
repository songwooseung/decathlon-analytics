from fastapi import APIRouter
from sqlalchemy import text
from core.db import engine
from services.utils import now_iso

router = APIRouter(tags=["debug"])

@router.get("/reviews")
def list_reviews(limit: int | None = None):
    sql = "SELECT * FROM reviews"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"; params["lim"] = int(limit)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}

@router.get("/summary")
def list_summary(limit: int | None = None):
    sql = "SELECT * FROM product_summary"
    params = {}
    if limit is not None:
        sql += " LIMIT :lim"; params["lim"] = int(limit)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"count": len(rows), "data": [dict(r) for r in rows]}

@router.get("")  # ← prefix가 /debug 이므로 최종 경로는 /debug
def root():
    return {"ok": True, "service": "Decathlon Review Analytics", "time": now_iso()}