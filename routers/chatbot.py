# routers/chatbot.py
from fastapi import APIRouter, Depends, Request, Response, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
import os, time, json, re, requests

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from core.db import get_db

from services.rag_index import search_products, build_index_from_db, index_meta
from openai import OpenAI

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# ----- 환경 -----
ENV = os.getenv("ENV", "dev").lower()          # dev | prod
COOKIE_NAME = "session_id"
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
ANALYTICS_BASE = os.getenv("ANALYTICS_BASE", "http://127.0.0.1:8000")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----- 스키마 -----
class ChatIn(BaseModel):
    message: str

class Recommendation(BaseModel):
    product_id: Optional[str] = None
    name: Optional[str] = None
    price: Optional[float] = None
    link: Optional[str] = None
    score: Optional[float] = None
    rating: Optional[float] = None
    evidence: Optional[List[Dict[str, Any]]] = None

class ChatOut(BaseModel):
    answer: str
    recommendations: Optional[List[Recommendation]] = None
    used_contexts: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# ----- 쿠키/세션 -----
def _set_session_cookie(response: Response, sid: str):
    secure = (ENV == "prod")
    samesite = "none" if secure else "lax"
    response.set_cookie(
        key=COOKIE_NAME,
        value=sid,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=SESSION_TTL_HOURS * 3600,
    )

def _ensure_session(request: Request, response: Response, db: Session) -> str:
    sid = request.cookies.get(COOKIE_NAME)
    now = datetime.utcnow()
    exp = now + timedelta(hours=SESSION_TTL_HOURS)

    # 공통 UPSERT SQL (행이 없어도 넣고, 있으면 갱신)
    upsert_sql = text("""
        INSERT INTO sessions (id, started_at, last_active, expires_at)
        VALUES (:id, NOW(), NOW(), :exp)
        ON CONFLICT (id) DO UPDATE
            SET last_active = EXCLUDED.last_active,
                expires_at  = EXCLUDED.expires_at;
    """)

    try:
        if not sid:
            sid = str(uuid4())
            _set_session_cookie(response, sid)
        db.execute(upsert_sql, {"id": sid, "exp": exp})
        db.commit()
        return sid
    except ProgrammingError as e:
        # 테이블이 없어서 실패한 경우 – 방어적으로 테이블 만들고 재시도
        ensure_chatbot_tables()
        db.execute(upsert_sql, {"id": sid, "exp": exp})
        db.commit()
        return sid

def _load_recent_context(db: Session, session_id: str, limit_turns: int = 6) -> List[Dict[str, Any]]:
    rows = db.execute(text("""
      SELECT role, content
      FROM messages
      WHERE session_id=:sid
      ORDER BY created_at DESC
      LIMIT :lim
    """), {"sid": session_id, "lim": limit_turns}).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows[::-1]]

def _save_message(db, session_id, role, content, meta=None):
    stmt = text("""
        INSERT INTO messages (session_id, role, content, meta)
        VALUES (:sid, :role, :content, CAST(:meta AS JSONB))
    """)
    db.execute(
        stmt,
        {
            "sid": str(session_id),
            "role": role,
            "content": content,
            "meta": json.dumps(meta or {}, ensure_ascii=False),
        },
    )
    db.commit()

# ----- 라우팅/LLM & 스몰톡 -----
def _route_kind(q: str) -> str:
    if any(k in q.lower() for k in ["top", "count", "trend", "distribution", "average"]):
        return "analytics"
    if any(k in q for k in ["탑", "순위", "많이", "평균", "분포", "월별", "추이", "개수", "비율"]):
        return "analytics"
    return "rag"

_HELLOS = ["안녕", "안뇽", "하이", "hello", "hi"]
_THANKS = ["고마워", "감사", "thanks", "thank you", "땡큐"]
_INTENT_HINTS = [
    "추천", "자켓", "재킷", "상의", "하의", "바지", "패딩", "점퍼", "베스트",
    "티셔츠", "셔츠", "러닝화", "신발", "등산화", "가방", "가격", "예산", "링크"
]
def _is_smalltalk(q: str) -> Optional[str]:
    low = (q or "").strip().lower()
    if any(h in low for h in [w.lower() for w in _INTENT_HINTS]):
        return None
    if any(w in low for w in [h.lower() for h in _HELLOS]):
        return "hello"
    if any(w in low for w in [t.lower() for t in _THANKS]):
        return "thanks"
    return None

def _is_non_product(q: str) -> bool:
    """날씨/일상/비상품 대화는 여기서 컷."""
    low = (q or "").lower()
    return any(k in low for k in ["날씨", "기온", "시간", "뉴스", "주가", "교통", "요리 레시피"])

def _call_llm(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    comp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.3,
        max_tokens=350,
    )
    return comp.choices[0].message.content.strip()

SYSTEM_PROMPT = (
    "당신은 데카트론 상품 리뷰 기반 어시스턴트입니다. "
    "항상 리뷰/요약에서 찾은 근거로만 답하고, 확실하지 않으면 근거 부족을 알리세요. "
    "추천은 제품명/대략 가격/핵심 장점 1-2개를 간단히 제시하세요. "
    "출력은 간단한 문장 2-5개와 근거 스니펫 1-3개를 포함하세요."
)

# ----- 분석 API 매핑 -----
def _analytics_dispatch(text_query: str) -> Dict[str, Any]:
    mapping = [
        (["러닝", "평점", "탑", "top"], "/running/top-rated"),
        (["러닝", "리뷰", "많이", "top"], "/running/top-by-reviewcount"),
        (["하이킹", "평점", "탑", "top"], "/hiking/top-rated"),
        (["하이킹", "리뷰", "많이", "top"], "/hiking/top-by-reviewcount"),
        (["월별", "추이"], "/total/monthly-reviews"),
        (["가격", "분포", "전체"], "/total/price-bins"),
        (["러닝", "가격", "분포"], "/running/price-distribution"),
        (["하이킹", "가격", "분포"], "/hiking/price-distribution"),
        (["워드클라우드", "러닝"], "/running/wordcloud"),
        (["워드클라우드", "하이킹"], "/hiking/wordcloud"),
    ]
    q = text_query.lower()
    for keys, path in mapping:
        if all(k.lower() in q for k in keys):
            url = f"{ANALYTICS_BASE}{path}"
            try:
                r = requests.get(url, timeout=5)
                if r.ok:
                    return {"endpoint": path, "data": r.json()}
            except Exception as e:
                return {"endpoint": path, "error": str(e)}
    try:
        r = requests.get(f"{ANALYTICS_BASE}/total/top-by-reviewcount", timeout=5)
        if r.ok:
            return {"endpoint": "/total/top-by-reviewcount", "data": r.json()}
    except Exception as e:
        return {"endpoint": "/total/top-by-reviewcount", "error": str(e)}
    return {"endpoint": None, "error": "analytics not available"}

# ----- 후속 질의/제어 신호 파서 -----
def _control_from_text(q: str) -> Dict[str, Any]:
    low = (q or "").lower()
    ctrl: Dict[str, Any] = {}

    # 더 저렴/비싼
    if any(k in low for k in ["더 싸", "저렴", "싸진", "낮은 가격"]):
        ctrl["price_bias"] = "cheaper"
    if any(k in low for k in ["더 비싸", "고급", "프리미엄", "비싼"]):
        ctrl["price_bias"] = "pricier"

    # 비슷한
    if any(k in low for k in ["비슷한", "유사한", "같은 라인", "같은 종류"]):
        ctrl["prefer_similar"] = True

    # 인기(리뷰수)
    if any(k in low for k in ["인기", "리뷰 많", "베스트", "판매량"]):
        ctrl["prefer"] = "popular"

    # 가성비
    if "가성비" in low:
        ctrl["prefer"] = "value"

    # O만원대 / 이하 / 이상
    m = re.search(r"(\d+)\s*만\s*원\s*대", low)
    if m:
        n = int(m.group(1)) * 10000
        ctrl["min_price"] = n
        ctrl["max_price"] = n + 9999
    m = re.search(r"(\d+)\s*만\s*원\s*(이하|밑|까지)", low)
    if m:
        ctrl["max_price"] = int(m.group(1)) * 10000
    m = re.search(r"(\d+)\s*만\s*원\s*(이상|부터)", low)
    if m:
        ctrl["min_price"] = int(m.group(1)) * 10000

    # “다른 제품” → 첫 결과는 건너뛰기
    if any(k in low for k in ["다른 제품", "다른 거", "또 다른", "하나 더", "다른거"]):
        ctrl["offset"] = 1

    # “N개” 요청
    m = re.search(r"(\d+)\s*개", low)
    if m:
        ctrl["top_k"] = max(1, min(5, int(m.group(1))))

    return ctrl

# ---------- Routes ----------
@router.get("/health")
def health():
    try:
        meta = index_meta()
    except Exception as e:
        meta = {"error": str(e)}
    return {"ok": True, "index": meta, "env": ENV}

@router.post("/reindex")
def reindex(db: Session = Depends(get_db)):
    info = build_index_from_db(db)
    return {"ok": True, "index": info}

@router.post("/chat", response_model=ChatOut)
def chat(
    req: ChatIn,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    debug: bool = Query(False),
):
    t0 = time.time()
    session_id = _ensure_session(request, response, db)
    history = _load_recent_context(db, session_id, limit_turns=6)

    # 비상품/스몰톡 가드
    if _is_non_product(req.message):
        answer = "죄송해요, 저는 제품 관련 질문에만 답할 수 있어요. 원하시는 품목(예: 방수 자켓)이나 예산을 알려주시면 바로 추천해드릴게요."
        _save_message(db, session_id, "user", req.message, meta={"route": "non-product"})
        _save_message(db, session_id, "assistant", answer, meta={"route": "non-product"})
        latency_ms = int((time.time() - t0) * 1000)
        return ChatOut(answer=answer, recommendations=None, used_contexts=None,
                       session_id=session_id, meta={"latency_ms": latency_ms, "route": "non-product", "env": ENV})

    st = _is_smalltalk(req.message)
    if st == "hello":
        answer = "안녕하세요! 무엇을 도와드릴까요? 자켓/바지/러닝화처럼 원하는 품목이나 예산을 말해주시면 바로 찾아드릴게요."
        _save_message(db, session_id, "user", req.message, meta={"route": "smalltalk"})
        _save_message(db, session_id, "assistant", answer, meta={"route": "smalltalk"})
        latency_ms = int((time.time() - t0) * 1000)
        return ChatOut(answer=answer, recommendations=None, used_contexts=None,
                       session_id=session_id, meta={"latency_ms": latency_ms, "route": "smalltalk", "env": ENV})
    if st == "thanks":
        answer = "도움이 되어 기뻐요! 더 필요한 게 있으면 언제든 편하게 물어보세요 🙂"
        _save_message(db, session_id, "user", req.message, meta={"route": "smalltalk"})
        _save_message(db, session_id, "assistant", answer, meta={"route": "smalltalk"})
        latency_ms = int((time.time() - t0) * 1000)
        return ChatOut(answer=answer, recommendations=None, used_contexts=None,
                       session_id=session_id, meta={"latency_ms": latency_ms, "route": "smalltalk", "env": ENV})

    # 분석 라우팅
    route = _route_kind(req.message)
    recs: List[Dict[str, Any]] = []
    used_contexts: Optional[List[Dict[str, Any]]] = None

    if route == "analytics":
        a = _analytics_dispatch(req.message)
        if "data" in a:
            data_preview = str(a["data"])[:300]
            answer = f"분석형 질의로 판단되어 `{a['endpoint']}` 결과를 요약했습니다.\n요약 미리보기: {data_preview}"
        else:
            answer = f"분석 API 호출에 실패했습니다. endpoint={a.get('endpoint')}, error={a.get('error')}"
    else:
        # ---- 제품 추천 ----
        ctrl = _control_from_text(req.message)
        top_k = ctrl.get("top_k", 1)

        try:
            recs = search_products(
                req.message,
                top_k=top_k,
                offset=ctrl.get("offset", 0),
                min_price=ctrl.get("min_price"),
                max_price=ctrl.get("max_price"),
                price_bias=ctrl.get("price_bias"),        # 'cheaper' | 'pricier' | None
                prefer=ctrl.get("prefer"),                # 'popular' | 'value' | None
                prefer_similar=ctrl.get("prefer_similar", False),
            )
        except Exception:
            recs = []

        if debug:
            used_contexts = recs

        if recs:
            top = recs[0]
            prompt = (
                f"사용자 질문: {req.message}\n\n"
                f"추천 후보(1개):\n"
                f"- 이름: {top.get('name')}\n- 가격: {top.get('price')}\n"
                f"- 카테고리: {top.get('category')}/{top.get('subcategory')}\n"
                f"- 링크: {top.get('link')}\n"
                f"- 근거 스니펫: \"{top.get('snippet','')}\"\n"
                f"- 평균 평점: {top.get('rating')}\n\n"
                f"요구사항: 2~4문장으로 짧게 추천 사유를 설명하고, 제품명/가격/핵심 장점 1-2개/링크를 자연스럽게 포함하라."
            )
            llm_messages = [{"role": "user", "content": prompt}]
            for h in history[-4:]:
                llm_messages.insert(0, {"role": h["role"], "content": h["content"]})
            answer = _call_llm(SYSTEM_PROMPT, llm_messages)
        else:
            answer = "요청과 맞는 제품을 찾기 어렵네요. 카테고리(예: 자켓/하의/신발)나 예산을 조금 더 알려줄래요?"

        for r in recs:
            r["evidence"] = [{
                "snippet": r.get("snippet"),
                "rating": r.get("rating"),
                "source": r.get("source", "reviews")
            }]

    _save_message(db, session_id, "user", req.message, meta={"route": route})
    _save_message(db, session_id, "assistant", answer, meta={"route": route})

    latency_ms = int((time.time() - t0) * 1000)
    return ChatOut(
        answer=answer,
        recommendations=recs or None,
        used_contexts=used_contexts,
        session_id=session_id,
        meta={"latency_ms": latency_ms, "route": route, "env": ENV},
    )