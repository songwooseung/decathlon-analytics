from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import APP_META
from core.db import ensure_tables  # ★ 스타트업에서 테이블 자동 생성
from routers import ingest, analytics, debug, chatbot
import os

app = FastAPI(**APP_META)

# --------- CORS (쿠키 사용) ---------
ENV = os.getenv("ENV", "dev").lower()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

# credentials=True이면 * 사용 불가. 정확한 Origin 지정
allow_origins = [FRONTEND_ORIGIN]
if ENV == "dev":
    # 로컬 개발 편의
    allow_origins += ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 스타트업 훅 ---------
@app.on_event("startup")
def on_startup():
    # 코어 테이블(reviews/product_summary) + 챗봇 테이블(sessions/messages/summary) 모두 보장
    ensure_tables()

@app.get("/", tags=["meta"])
def root():
    return {"ok": True, "service": "decathlon-analytics"}

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok"}   

# --------- 라우터 등록 ---------
app.include_router(ingest.router)
app.include_router(analytics.router)
app.include_router(debug.router)
app.include_router(chatbot.router)
