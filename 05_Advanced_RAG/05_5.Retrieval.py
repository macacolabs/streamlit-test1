"""
RAG 검색 파이프라인 — Contextual Retrieval (하이브리드)

검색 전략:
  1. Semantic Search   : ChromaDB 유사도 검색 (Contextual Embedding 적용된 벡터)
  2. Contextual BM25   : rank_bm25 키워드 검색 (맥락 보강 텍스트 기반)
  3. RRF Fusion        : Reciprocal Rank Fusion으로 두 순위 목록을 병합
  → 의미 기반 + 키워드 기반 검색을 결합해 누락·오순위를 최소화
"""

import hashlib
import importlib.util
import os
import re
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# ──────────────────────────────────────────────
# 형제 모듈 동적 임포트
# ──────────────────────────────────────────────

def _load_sibling(filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_schemas       = _load_sibling("05_2.Schemas.py")
CurriculumPlan = _schemas.CurriculumPlan


# ──────────────────────────────────────────────
# BM25 토크나이저 (05_4.Indexing과 동일한 로직)
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+", text.lower())


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────

def _rrf_fuse(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """
    여러 순위 목록을 RRF로 병합한다.

    score(d) = Σ  1 / (rrf_k + rank_i(d))
               i

    같은 문서는 page_content 앞 200자의 MD5 해시로 식별한다.
    """
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            scores[doc_id]  = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = [doc_map[did] for did in sorted_ids]
    return fused[:top_n] if top_n else fused


# ──────────────────────────────────────────────
# BM25 검색
# ──────────────────────────────────────────────

def _bm25_search(
    bm25_data: dict,
    query: str,
    filter_fn: "callable[[dict], bool]",
    k: int,
) -> list[Document]:
    """
    BM25로 query와 관련된 상위 k개 문서를 반환한다.
    filter_fn(metadata) → True인 문서만 결과에 포함된다.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores        = bm25_data["bm25"].get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[Document] = []
    for idx in ranked_indices:
        if len(results) >= k:
            break
        meta = bm25_data["metadatas"][idx]
        if filter_fn(meta):
            results.append(Document(
                page_content=bm25_data["contents"][idx],
                metadata=meta,
            ))
    return results


# ──────────────────────────────────────────────
# 하이브리드 검색 (Semantic + BM25 + RRF)
# ──────────────────────────────────────────────

def _hybrid_search(
    vectorstore: Chroma,
    bm25_data: dict | None,
    query: str,
    chroma_filter: dict,
    bm25_filter_fn: "callable[[dict], bool]",
    k: int,
) -> list[Document]:
    """
    Semantic Search + Contextual BM25 결과를 RRF로 병합한다.
    bm25_data가 None이면 Semantic Search만 사용한다.
    """
    # 1) Semantic Search
    semantic_docs = vectorstore.similarity_search(query, k=k, filter=chroma_filter)

    if bm25_data is None:
        return semantic_docs

    # 2) BM25 Search
    bm25_docs = _bm25_search(bm25_data, query, bm25_filter_fn, k=k)

    if not bm25_docs:
        return semantic_docs

    # 3) RRF Fusion
    return _rrf_fuse([semantic_docs, bm25_docs], top_n=k)


# ──────────────────────────────────────────────
# 생성 프롬프트
# ──────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = dedent("""
    당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
    앞선 대화에서 수집한 기업 요구사항과 AX Compass 진단 결과를 바탕으로 맞춤형 교육 커리큘럼을 설계하라.

    커리큘럼 구조:
    - theory_sessions: 모든 참가자가 동일하게 수강하는 공통 이론 수업. 4개 이상 6개 이하.
    - group_sessions: 3개 그룹이 각각 다른 실습을 진행. 각 그룹당 2개 이상 3개 이하.

    규칙:
    1. 그룹별 실습은 해당 유형의 강점을 활용하고 보완 방향을 실습 활동에 녹인다.
    2. 각 회차는 title, duration_hours, goals, activities를 포함한다.
    3. duration_hours는 시간 단위(소수점 가능)이며, 아래 시간 배분 규칙을 따른다:
       - theory_sessions 합계 + 그룹 실습 합계(1개 그룹 기준) = 총 교육 시간
       - group_sessions는 3개 그룹이 동시에 진행되므로 각 그룹의 duration_hours 합산은 모두 동일해야 한다.
    4. 기업 교육답게 실무 적용 중심으로 구성한다.
    5. notes에는 유형별 특성과 기업 제한사항을 반영한 주의사항을 작성한다.
    6. [커리큘럼 예시 참고 자료]가 제공되는 경우 세션 구성 방식·활동 유형·강의 수준을 참고하되 그대로 복사하지 마라.
""").strip()


# ──────────────────────────────────────────────
# RAG 3단계: 유형별 컨텍스트 검색 (하이브리드)
# ──────────────────────────────────────────────

def retrieve_group_context(
    vectorstore: Chroma,
    type_names: list[str],
    bm25_data: dict | None = None,
) -> str:
    """
    AX Compass 유형별 특성 문서를 하이브리드 검색으로 가져온다.
    - Semantic: type_name 메타데이터 필터 + 유사도 검색
    - BM25: 동일 필터 조건의 키워드 검색
    - RRF: 두 결과 병합 후 유형별 그룹화 출력
    """
    query = (
        f"{', '.join(type_names)} 유형의 "
        "AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    )
    k = max(len(type_names) * 2, 4)

    chroma_filter  = {"$and": [
        {"doc_type":  {"$eq": "ax_compass"}},
        {"type_name": {"$in": type_names}},
    ]}
    bm25_filter_fn = lambda m: (
        m.get("doc_type") == "ax_compass" and m.get("type_name") in type_names
    )

    docs = _hybrid_search(
        vectorstore, bm25_data, query,
        chroma_filter, bm25_filter_fn, k=k,
    )

    # 유형별로 그룹화하여 출력
    grouped: dict[str, list[str]] = {t: [] for t in type_names}
    for d in docs:
        tname = d.metadata.get("type_name", "")
        # Contextual Embedding이 적용된 경우 original_content 우선 사용
        content = d.metadata.get("original_content") or d.page_content
        if tname in grouped:
            grouped[tname].append(content)

    parts = []
    for tname, contents in grouped.items():
        if contents:
            parts.append(f"[{tname}]\n" + "\n\n".join(contents))

    return "\n\n".join(parts) if parts else "\n\n".join(
        d.metadata.get("original_content") or d.page_content for d in docs
    )


# ──────────────────────────────────────────────
# RAG 3단계: 커리큘럼 예시 검색 (하이브리드)
# ──────────────────────────────────────────────

def retrieve_curriculum_examples(
    vectorstore: Chroma,
    query: str,
    bm25_data: dict | None = None,
    k: int = 3,
) -> str:
    """
    커리큘럼 예시를 하이브리드 검색으로 가져온다.
    출처(파일명/섹션)를 헤더로 추가해 LLM이 구조를 파악하기 쉽게 한다.
    """
    chroma_filter  = {"doc_type": {"$eq": "curriculum_example"}}
    bm25_filter_fn = lambda m: m.get("doc_type") == "curriculum_example"

    docs = _hybrid_search(
        vectorstore, bm25_data, query,
        chroma_filter, bm25_filter_fn, k=k,
    )

    parts = []
    for d in docs:
        course  = d.metadata.get("course_name", "")
        section = d.metadata.get("section", "")
        header  = f"[출처: {course}" + (f" / {section}" if section else "") + "]"
        content = d.metadata.get("original_content") or d.page_content
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────
# RAG 4단계: LCEL 체인
# ──────────────────────────────────────────────

def build_chain(vectorstore: Chroma, api_key: str, bm25_data: dict | None = None):
    """
    커리큘럼 생성 체인을 반환한다.

    Parameters
    ----------
    vectorstore : ChromaDB 인스턴스
    api_key     : OpenAI API 키
    bm25_data   : Contextual BM25 인덱스 (None이면 Semantic Search만 사용)
    """
    llm            = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation = input_dict["conversation"]
        groups       = input_dict["groups"]
        total_hours  = input_dict["total_hours"]
        topic        = input_dict.get("topic", "")
        level        = input_dict.get("level", "")
        ga, gb, gc   = groups["group_a"], groups["group_b"], groups["group_c"]

        print("[RAG] 유형별 컨텍스트 검색 중... (하이브리드)")
        ctx_a = retrieve_group_context(vectorstore, ga["types"], bm25_data)
        ctx_b = retrieve_group_context(vectorstore, gb["types"], bm25_data)
        ctx_c = retrieve_group_context(vectorstore, gc["types"], bm25_data)

        print("[RAG] 커리큘럼 예시 검색 중... (하이브리드)")
        curriculum_examples = retrieve_curriculum_examples(
            vectorstore,
            f"{topic} {level} 기업 AI 교육 커리큘럼",
            bm25_data,
            k=3,
        )

        chat_history = [m for m in conversation if not isinstance(m, SystemMessage)]
        theory_hours = round(total_hours * 0.65)
        group_hours  = total_hours - theory_hours

        rag_content = dedent(f"""
            위 대화에서 수집한 요구사항을 바탕으로 맞춤형 교육 커리큘럼을 설계해줘.

            [시간 배분 기준]
            총 교육 시간: {total_hours}시간
            - 이론 수업(theory_sessions) duration_hours 합계: 정확히 {theory_hours}시간
            - 그룹 실습(group_sessions) duration_hours 합계 (1개 그룹 기준): 정확히 {group_hours}시간

            [그룹 구성]
            - {ga['name']} ({' · '.join(ga['types'])}): {ga['count']}명
            - {gb['name']} ({' · '.join(gb['types'])}): {gb['count']}명
            - {gc['name']} ({' · '.join(gc['types'])}): {gc['count']}명

            [AX Compass 유형별 특성 — 하이브리드 검색 결과]
            === 그룹 A ({' · '.join(ga['types'])}) ===
            {ctx_a}

            === 그룹 B ({' · '.join(gb['types'])}) ===
            {ctx_b}

            === 그룹 C ({' · '.join(gc['types'])}) ===
            {ctx_c}

            [커리큘럼 예시 참고 자료]
            {curriculum_examples}
        """).strip()

        return (
            [SystemMessage(content=GENERATION_SYSTEM_PROMPT)]
            + chat_history
            + [HumanMessage(content=rag_content)]
        )

    return RunnableLambda(retrieve_and_build_messages) | structured_llm
