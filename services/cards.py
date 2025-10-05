import json
from sqlalchemy import text
from core.db import engine
from services.utils import category_korean_label

def build_product_card(product_id: str):
    # summary
    with engine.begin() as conn:
        srow = conn.execute(text("""
            SELECT product_id, product_name, category, subcategory, brand, price,
                   total_reviews, positive_reviews, avg_rating, url, thumbnail_url
            FROM product_summary
            WHERE product_id = :pid
        """), {"pid": product_id}).mappings().first()
    if not srow: return None

    # 대표 리뷰 1건 (rating DESC → review_date DESC)
    with engine.begin() as conn:
        rrow = conn.execute(text("""
            SELECT review_id, review_text
            FROM reviews
            WHERE product_id = :pid
            ORDER BY rating DESC, review_date DESC
            LIMIT 1
        """), {"pid": product_id}).mappings().first()

    review_id   = rrow["review_id"]   if rrow else None
    review_text = rrow["review_text"] if rrow else None
    image_url   = srow.get("thumbnail_url") or ""

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
        "images": image_url,  # 문자열 하나
        "url": srow["url"] or "",
    }

