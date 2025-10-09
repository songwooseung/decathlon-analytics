from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import APP_META
from core.db import ensure_tables  # ★ 스타트업에서 테이블 자동 생성
from routers import ingest, analytics, debug, chatbot
import os

app = FastAPI(**APP_META)

# --------- CORS (쿠키 사용) ---------
ENV = os.getenv("ENV", "dev").lower()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

# 쉼표로 여러 개 넣기 지원
origins = [o.strip() for o in FRONTEND_ORIGIN.split(",") if o.strip()]

# 필요하면 추가(중복 방지)
if ENV == "dev":
    for o in ["http://localhost:3000", "http://127.0.0.1:3000"]:
        if o not in origins:
            origins.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,   # ★ 쿠키 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 스타트업 훅 ---------
@app.on_event("startup")
def on_startup():
    # 코어 테이블(reviews/product_summary) + 챗봇 테이블(sessions/messages/summary) 모두 보장
    ensure_tables()

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return PlainTextResponse("ok")

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

# --------- 라우터 등록 ---------
app.include_router(ingest.router)
app.include_router(analytics.router)
app.include_router(debug.router, prefix="/debug")
app.include_router(chatbot.router)
