# routers/chatbot.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from services.embeddings import search_similar, rebuild_index
from services.utils import (
    get_product_summary,     # product_summary에서 메타 조회
    pick_positive_texts,     # 장점 2개 안전 추출
    pick_negative_texts,     # 단점 2개 안전 추출
)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str
    used_contexts: List[Dict[str, Any]]

# -------------------------------
# 간단 상태(세션 없는 버전): 직전 추천 기억
# -------------------------------
LAST: Dict[str, Optional[str]] = {
    "product_id": None,
    "category": None,      # 예: "러닝", "등산/하이킹"
    "subkw": None,         # 예: "자켓", "가방" ...
}

# -------------------------------
# 동의어/토큰 정규화
# -------------------------------
REPLACE = {
    # 서브 품목 계열
    "재킷": "자켓",
    "점퍼": "자켓",
    "바람막이": "자켓",
    "윈드재킷": "자켓",
    "윈드 재킷": "자켓",

    "백팩": "가방",
    "배낭": "가방",
    "데이팩": "가방",
    "daypack": "가방",
    "pack": "가방",

    "캡": "모자",
    "헤드밴드": "모자",

    "티셔츠": "티",
    "셔츠": "티",

    "반바지": "쇼츠",
}

CATEGORY_TOKENS = {
    "러닝": ["러닝", "런닝", "조깅", "running"],
    "등산/하이킹": ["등산", "하이킹", "트레킹", "hiking", "trekking"],
}

# 서브카테고리 동의어 풀(이 목록으로 이름/서브카테고리 매칭)
SUBTOKENS = {
    "자켓": ["자켓", "바람막이", "윈드재킷", "재킷", "wind"],
    "가방": ["가방", "백팩", "배낭", "데이팩", "daypack", "pack"],
    "모자": ["모자", "캡", "헤드밴드", "비니", "버킷햇", "hat", "cap"],
    "양말": ["양말", "삭스", "socks"],
    "티"  : ["티", "티셔츠", "top", "shirt"],
    "쇼츠": ["쇼츠", "숏츠", "반바지", "shorts"],
    "바지": ["바지", "팬츠", "하의", "pants"],
    "벨트": ["벨트", "belt"],
    "플라스크": ["플라스크", "물병", "보틀", "soft flask", "flask"],
    "신발": ["신발", "러닝화", "하이킹화", "트레일러닝화", "슈즈", "shoes"],
}

HELLO = ["안녕", "안녕하세요", "하이", "hello", "hi"]
THANKS = ["고마워", "감사", "덕분에"]

def norm(s: str) -> str:
    out = s
    for a, b in REPLACE.items():
        out = out.replace(a, b)
    return out

def detect_intent(msg: str) -> Dict[str, Optional[str]]:
    """
    msg에서 카테고리/서브키워드(품목) 추출
      - category: None 가능(품목만 있는 요청 허용: 예) "가방 추천")
      - subkw   : None 가능
    """
    m = norm(msg)
    cat: Optional[str] = None
    for k, toks in CATEGORY_TOKENS.items():
        if any(t in m for t in toks):
            cat = k
            break

    sub: Optional[str] = None
    # 가장 구체적인 토큰부터 찾음
    for sk, toks in SUBTOKENS.items():
        if any(t in m for t in toks):
            sub = sk
            break

    return {"category": cat, "subkw": sub}

# -------------------------------
# 포맷/메타/필터 도우미
# -------------------------------
def join_quotes(quotes: List[str]) -> str:
    return " / ".join([f"“{q}”" for q in quotes]) if quotes else "정보 없음"

def enrich_with_meta(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = get_product_summary(str(doc.get("product_id")))
    if not meta:
        return None
    out = dict(doc)
    out.update(meta)
    return out

def contexts_to_brief(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"product_id": str(d.get("product_id")), "score": float(d.get("score", 0.0))} for d in docs]

def _match_with_synonyms(text: str, subkw: str) -> bool:
    """SUBTOKENS 내 모든 별칭 기준으로 부분 매칭 (대소문자 무시)"""
    if not text or not subkw:
        return False
    text = text.lower()
    for alias in SUBTOKENS.get(subkw, [subkw]):
        if alias.lower() in text:
            return True
    return False

def filter_by_intent(enriched_docs: List[Dict[str, Any]],
                     category: Optional[str],
                     subkw: Optional[str]) -> List[Dict[str, Any]]:
    results = []
    for e in enriched_docs:
        # 카테고리 일치 우선
        if category and e.get("category") != category:
            continue

        # 서브키워드 동의어 매칭
        if subkw:
            name = e.get("product_name", "")
            subc = e.get("subcategory", "")
            if not (_match_with_synonyms(name, subkw) or _match_with_synonyms(subc, subkw)):
                continue

        results.append(e)
    return results

def format_rec(e: Dict[str, Any], show_link=True, only_pos=False, only_neg=False) -> str:
    pid = str(e["product_id"])
    pos = pick_positive_texts(pid, limit=2) if not only_neg else []
    neg = pick_negative_texts(pid, limit=2) if not only_pos and not only_neg else []
    parts = [
        f"{e['product_name']} ({pid})",
        f"가격: {e['price']}원 · 평점 {e['avg_rating']}",
    ]
    if only_neg:
        parts.append(f"단점: {join_quotes(neg)}")
    else:
        parts.append(f"장점: {join_quotes(pos)}")
        if neg and not only_pos:
            parts.append(f"단점: {join_quotes(neg)}")
    if show_link:
        parts.append(f"제품 링크 : {e['url']}")
    return "\n".join(parts)

# -------------------------------
# 엔드포인트
# -------------------------------
@router.post("/reindex")
def force_reindex():
    idx = rebuild_index()
    return {"ok": True, "docs": len(idx.get("docs", []))}

@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    user_msg_raw = (body.message or "").strip()
    user_msg = norm(user_msg_raw)

    def is_short_plain(text: str) -> bool:
        return (len(text) <= 15) and ("?" not in text) and ("추천" not in text) and ("요약" not in text) and ("단점" not in text)

    if any(t in user_msg for t in HELLO) and is_short_plain(user_msg_raw):
        return {"answer": "안녕하세요! 무엇을 도와드릴까요? (예: 제품 추천 / 리뷰 요약 / 단점 알려줘)", "used_contexts": []}
    if any(t in user_msg for t in THANKS) and is_short_plain(user_msg_raw):
        return {"answer": "도움이 되었다니 다행입니다. 필요하시면 언제든 말씀해 주세요.", "used_contexts": []}

    # 1) 의도 파악
    intent = detect_intent(user_msg)
    req_cat, req_sub = intent["category"], intent["subkw"]

    # 2) 검색 → 메타 합치기
    raw_contexts = search_similar(user_msg, top_k=7)
    enriched = [ed for d in raw_contexts if (ed := enrich_with_meta(d))]

    if not enriched:
        return {"answer": "죄송합니다. 관련 제품을 찾지 못했습니다.", "used_contexts": contexts_to_brief(raw_contexts)}

    if ("말고" in user_msg) or ("다른" in user_msg):
        base_cat = LAST["category"] or req_cat
        base_sub = LAST["subkw"] or req_sub
        exclude_id = LAST["product_id"]

        # 같은 category+subkw 우선
        pool = filter_by_intent(enriched, base_cat, base_sub)

        # 만약 품목이 있었는데 비면 그대로 리턴
        if base_sub and not pool:
            return {"answer": f"{base_sub} 쪽에서 다른 추천은 찾지 못했어요.", "used_contexts": []}

        if not pool and base_cat:
            pool = filter_by_intent(enriched, base_cat, None)
        if not pool:
            pool = enriched

        if exclude_id:
            pool = [e for e in pool if str(e["product_id"]) != str(exclude_id)]
        if not pool:
            return {"answer": "다른 추천 결과를 찾지 못했습니다.", "used_contexts": contexts_to_brief(enriched[:3])}

        best = pool[0]
        LAST.update(product_id=str(best["product_id"]), category=best.get("category"), subkw=base_sub)
        return {"answer": format_rec(best, show_link=True, only_pos=True), "used_contexts": contexts_to_brief(pool[:3])}

    # 4) 카테고리/품목 기반 추천(예: “러닝 자켓”, “가방 추천”)
    cand = filter_by_intent(enriched, req_cat, req_sub)

    # 카테고리만 말했는데 실패하면, 카테고리만으로 한 번 더 좁혀보기
    if not cand:
        if req_sub:
            return {"answer": f"요청하신 ‘{req_sub}’ 제품을 찾지 못했습니다. "
                            f"예: 등산 {req_sub}, 러닝 {req_sub} 처럼 말해보세요.",
                    "used_contexts": []}
        cand = enriched

    best = cand[0]

    # 5-a) 요약 요청
    if "요약" in user_msg:
        LAST.update(product_id=str(best["product_id"]), category=best.get("category"), subkw=req_sub)
        pos = pick_positive_texts(str(best["product_id"]), limit=2)
        neg = pick_negative_texts(str(best["product_id"]), limit=2)
        ans = (
            f"제품 요약: {best['product_name']} ({best['product_id']})\n"
            f"총 리뷰 {best['total_reviews']}개 / 평균 평점 {best['avg_rating']} / 가격 {best['price']}원\n"
            f"장점: {join_quotes(pos)}\n"
            f"단점: {join_quotes(neg)}\n"
            f"제품 링크 : {best['url']}"
        )
        return {"answer": ans, "used_contexts": contexts_to_brief([best])}

    # 5-b) 단점만 물어봄 → 직전 선택이 있으면 그걸, 없으면 이번 best
    if "단점" in user_msg:
        target = None
        if LAST["product_id"]:
            meta = get_product_summary(LAST["product_id"])
            if meta:
                target = meta
        if not target:
            target = best
        LAST.update(product_id=str(target["product_id"]), category=target.get("category"), subkw=req_sub or LAST["subkw"])
        return {
            "answer": format_rec(target, show_link=False, only_neg=True),
            "used_contexts": contexts_to_brief([target])
        }

    # 6) 기본 추천(요구: 기본은 장점만 노출, 링크 출력)
    LAST.update(product_id=str(best["product_id"]), category=best.get("category"), subkw=req_sub)
    return {
        "answer": format_rec(best, show_link=True, only_pos=True),
        "used_contexts": contexts_to_brief(cand[:3]),
    }