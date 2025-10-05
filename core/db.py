from sqlalchemy import create_engine, text
from core.config import DB_URL

if DB_URL.startswith("postgresql"):
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
else:
    engine = create_engine(DB_URL, future=True)

def ensure_tables():
    with engine.begin() as conn:
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