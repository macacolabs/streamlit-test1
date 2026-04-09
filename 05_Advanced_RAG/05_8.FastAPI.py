"""
FastAPI 백엔드 서버

엔드포인트:
  GET  /health        — 헬스 체크
  POST /api/chat      — 정보 수집 대화 (LangChain ChatOpenAI)
  POST /api/generate  — 커리큘럼 생성 (RAG + LangChain)
"""

import importlib.util
import os
from contextlib import asynccontextmanager
from textwrap import dedent

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 형제 모듈 동적 임포트
# ──────────────────────────────────────────────

def _load_sibling(filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_auth      = _load_sibling("05_3.Auth.py")
_indexing  = _load_sibling("05_4.Indexing.py")
_retrieval = _load_sibling("05_5.Retrieval.py")
_schemas   = _load_sibling("05_2.Schemas.py")

load_env_file      = _auth.load_env_file
require_api_key    = _auth.require_api_key
setup_vector_store = _indexing.setup_vector_store
build_chain        = _retrieval.build_chain
CollectedInfo      = _schemas.CollectedInfo

# ──────────────────────────────────────────────
# 수집 시스템 프롬프트
# ──────────────────────────────────────────────

COLLECTION_SYSTEM_PROMPT = dedent("""
    당신은 기업 AI 교육 커리큘럼 설계를 위한 정보 수집 어시스턴트다.
    아래 항목을 자연스러운 대화로 순서대로 수집해라.
    한 번에 1개씩 질문하고, 모든 항목이 수집되면 마지막에 "[정보 수집 완료]"를 출력해라.

    수집 항목:
    - 회사명 또는 팀 이름
    - 교육 목표
    - 교육 대상자
    - 현재 AI 활용 수준 (입문/초급/중급)
    - 총 교육 기간 (일수)
    - 하루 교육 시간 (시간)
    - 원하는 핵심 주제
    - 반영해야 할 조건 또는 제한사항
    - AX Compass 진단 결과: 6개 유형별 인원수 (균형형, 이해형, 과신형, 실행형, 판단형, 조심형)
""").strip()

# ──────────────────────────────────────────────
# 앱 상태 (lifespan에서 초기화)
# ──────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] 환경 변수 로드 중...")
    load_env_file()
    api_key = require_api_key()

    print("[Startup] LLM 초기화 중...")
    _state["api_key"]    = api_key
    _state["llm"]        = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

    print("[Startup] 벡터 DB 초기화 중...")
    _state["vectorstore"] = setup_vector_store(api_key)

    print("[Startup] 완료 — 서버 준비됨")
    yield
    _state.clear()


# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────

app = FastAPI(
    title="AX Compass 커리큘럼 백엔드",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Streamlit Cloud 도메인 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 요청 / 응답 스키마
# ──────────────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    content: str
    is_done: bool   # "[정보 수집 완료]" 감지 시 True


class GenerateRequest(BaseModel):
    messages: list[Message]


class GenerateResponse(BaseModel):
    curriculum: dict


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _to_lc(messages: list[Message]) -> list:
    """프론트 메시지 목록 → LangChain 메시지 목록 (시스템 프롬프트 자동 삽입)"""
    lc = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)]
    for m in messages:
        if m.role == "user":
            lc.append(HumanMessage(content=m.content))
        else:
            lc.append(AIMessage(content=m.content))
    return lc


def _calculate_groups(info: CollectedInfo) -> dict:
    return {
        "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],
                    "count": info.count_balanced + info.count_learner},
        "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],
                    "count": info.count_overconfident + info.count_doer},
        "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"],
                    "count": info.count_analyst + info.count_cautious},
    }

# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vectorstore_ready": "vectorstore" in _state,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """정보 수집 대화 — 사용자 메시지를 받아 다음 질문을 반환한다."""
    if "llm" not in _state:
        raise HTTPException(status_code=503, detail="LLM이 아직 초기화되지 않았습니다.")

    try:
        lc_msgs  = _to_lc(req.messages)
        response = _state["llm"].invoke(lc_msgs)
        is_done  = "[정보 수집 완료]" in response.content
        return ChatResponse(content=response.content, is_done=is_done)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """RAG 기반 커리큘럼 생성 — 수집된 대화 히스토리에서 정보를 추출 후 커리큘럼을 반환한다."""
    if "vectorstore" not in _state:
        raise HTTPException(status_code=503, detail="벡터 DB가 아직 초기화되지 않았습니다.")

    try:
        llm         = _state["llm"]
        vectorstore = _state["vectorstore"]
        api_key     = _state["api_key"]
        lc_msgs     = _to_lc(req.messages)

        # 정보 구조화 추출
        extract_llm = llm.with_structured_output(CollectedInfo)
        info = extract_llm.invoke(
            lc_msgs + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
        )

        groups      = _calculate_groups(info)
        total_hours = info.days * info.hours_per_day

        chain  = build_chain(vectorstore, api_key)
        result = chain.invoke({
            "conversation": lc_msgs,
            "groups":       groups,
            "total_hours":  total_hours,
            "topic":        info.topic,
            "level":        info.level,
        })

        return GenerateResponse(curriculum=result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# 직접 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
