# core/db.py
import os
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.config import DB_URL

if DB_URL.startswith("postgresql"):
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
else:
    engine = create_engine(DB_URL, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

def get_engine():
    return engine

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- 코어 테이블 ----------
def ensure_reviews_table(conn):
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

def ensure_summary_table(conn):
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

# ---------- 챗봇 테이블 ----------
def ensure_chatbot_tables(conn=None):
    if conn is None:
        with engine.begin() as c:
            _ensure_chat_tables_sql(c)
    else:
        _ensure_chat_tables_sql(conn)

def _ensure_chat_tables_sql(conn):
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL
    );
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(16) NOT NULL,
        content TEXT NOT NULL,
        meta JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS session_summaries (
        session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        rolling_summary TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """))
    conn.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at DESC);
    """))

# ---------- 외부 진입점(앱 스타트업에서 호출) ----------
def ensure_tables():
    with engine.begin() as conn:
        ensure_reviews_table(conn)
        ensure_summary_table(conn)
        ensure_chatbot_tables(conn)   # ★ 세션/메시지까지 항상 보장