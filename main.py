from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import APP_META
from routers import ingest, analytics, debug, chatbot

app = FastAPI(**APP_META)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(ingest.router)
app.include_router(analytics.router)
app.include_router(debug.router)
app.include_router(chatbot.router)
