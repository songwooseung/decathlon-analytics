from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd, io, os
from services.ingest import ingest_reviews_df, ingest_summary_df

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/reviews/file")
async def ingest_reviews_file(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    return ingest_reviews_df(df)

@router.post("/summary/file")
async def ingest_summary_file(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    return ingest_summary_df(df)

@router.get("/reviews")
def ingest_reviews(csv_path: str):
    if not os.path.exists(csv_path): raise HTTPException(400, f"CSV not found: {csv_path}")
    return ingest_reviews_df(pd.read_csv(csv_path))

@router.get("/summary")
def ingest_summary(csv_path: str):
    if not os.path.exists(csv_path): raise HTTPException(400, f"CSV not found: {csv_path}")
    return ingest_summary_df(pd.read_csv(csv_path))