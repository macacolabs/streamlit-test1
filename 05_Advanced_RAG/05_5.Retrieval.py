"""
RAG 검색 파이프라인 — Hybrid Search (Full Pipeline)

검색 전략 (5단계):
  Stage 1. Multi-Query Expansion
           LLM이 원본 쿼리를 2개 변형 생성 → 총 3개 쿼리로 리콜 확장

  Stage 2. Per-Query Hybrid Retrieval
           각 쿼리별로 다음 두 검색을 병렬 실행:
             a) Semantic Search  — ChromaDB (Contextual Embedding 적용 벡터)
             b) Contextual BM25  — rank_bm25 (맥락 보강 텍스트 역인덱스)

  Stage 3. RRF Fusion
           모든 쿼리의 Semantic + BM25 결과를 Reciprocal Rank Fusion으로 통합
           → 의미·키워드 모두에서 높은 순위를 가진 문서가 상위에 집중

  Stage 4. FlashRank Re-ranking
           후보 풀을 크로스인코더 모델로 1차 정밀 재정렬
           → k_final × 2 중간 후보 풀 생성

  Stage 5. LLM Listwise Re-ranking
           FlashRank 후보 풀을 LLM이 쿼리-문서 의미를 직접 이해하여 최종 재정렬
           → 원본 쿼리에 가장 관련도 높은 상위 k_final개만 반환

흐름 요약:
  원본 쿼리
    → [q_orig, q1, q2]         (Multi-Query Expansion)
    → 각 qi: Semantic(k) + BM25(k)
    → 전체 결과 RRF             (후보 풀: k_final × 3)
    → FlashRank                 (중간 풀: k_final × 2)
    → LLM Listwise Reranking    (최종 top-k)
"""

import hashlib
import importlib.util
import os
import re
from functools import lru_cache
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


# ══════════════════════════════════════════════
# Stage 1: Multi-Query Expansion
# ══════════════════════════════════════════════

_EXPAND_SYSTEM = (
    "당신은 RAG 검색 전문가다. "
    "주어진 검색 쿼리를 의미가 유사하지만 다른 표현으로 변형해 검색 리콜을 높여라. "
    "변형 쿼리만 줄바꿈으로 구분해 출력하라. 번호·기호·설명을 붙이지 마라."
)


def _expand_query(llm: ChatOpenAI, query: str, n: int = 2) -> list[str]:
    """
    원본 쿼리에서 n개의 변형 쿼리를 생성한다.
    원본 포함 최대 n+1개 쿼리를 반환하며, LLM 실패 시 원본만 반환한다.

    Examples
    --------
    원본: "균형형, 이해형 유형의 AI 활용 특성"
    변형: ["균형형 이해형의 인공지능 도구 활용 방식",
           "balanced learner 유형 AI 교육 접근법"]
    """
    prompt = (
        f"아래 검색 쿼리를 의미는 같되 표현이 다른 변형 {n}개를 생성하라.\n\n"
        f"원본 쿼리: {query}"
    )
    try:
        resp  = llm.invoke([
            SystemMessage(content=_EXPAND_SYSTEM),
            HumanMessage(content=prompt),
        ])
        lines = [ln.strip() for ln in resp.content.strip().splitlines() if ln.strip()]
        variants = lines[:n]
    except Exception as e:
        print(f"[MultiQuery] 쿼리 확장 실패 (원본만 사용): {e}")
        variants = []

    queries = [query] + variants
    print(f"[MultiQuery] 쿼리 {len(queries)}개 생성: {queries}")
    return queries


# ══════════════════════════════════════════════
# Stage 2-a: Semantic Search
# ══════════════════════════════════════════════

def _semantic_search(
    vectorstore: Chroma,
    query: str,
    chroma_filter: dict,
    k: int,
) -> list[Document]:
    return vectorstore.similarity_search(query, k=k, filter=chroma_filter)


# ══════════════════════════════════════════════
# Stage 2-b: BM25 Search
# ══════════════════════════════════════════════

def _bm25_search(
    bm25_data: dict,
    query: str,
    filter_fn: "callable[[dict], bool]",
    k: int,
) -> list[Document]:
    """
    Contextual BM25로 query와 관련된 상위 k개 문서를 반환한다.
    filter_fn(metadata) → True인 문서만 결과에 포함된다.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores         = bm25_data["bm25"].get_scores(tokens)
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


# ══════════════════════════════════════════════
# Stage 3: RRF Fusion
# ══════════════════════════════════════════════

def _rrf_fuse(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[Document]:
    """
    여러 순위 목록을 Reciprocal Rank Fusion으로 병합한다.

    score(d) = Σ  1 / (rrf_k + rank_i(d))
               i

    문서 식별: page_content 앞 200자의 MD5 해시
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


# ══════════════════════════════════════════════
# Stage 4: FlashRank Re-ranking
# ══════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_flashrank_ranker():
    """FlashRank Ranker를 싱글턴으로 캐싱한다 (모델 1회만 로드)."""
    from flashrank import Ranker
    print("[FlashRank] 모델 로드 중 (최초 1회)...")
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")


def _rerank_with_flashrank(
    query: str,
    docs: list[Document],
    top_k: int,
) -> list[Document]:
    """
    FlashRank 크로스인코더 모델로 문서를 재정렬한다.

    - 검색 쿼리와 각 문서를 쌍으로 평가해 관련도 점수를 계산
    - 임베딩 기반 유사도보다 정밀한 의미 매칭 수행
    - 실패 시 RRF 순위 그대로 반환 (graceful degradation)
    """
    if not docs:
        return docs

    try:
        from flashrank import RerankRequest

        ranker   = _get_flashrank_ranker()
        passages = [
            {"id": i, "text": d.metadata.get("original_content") or d.page_content}
            for i, d in enumerate(docs)
        ]
        request  = RerankRequest(query=query, passages=passages)
        results  = ranker.rerank(request)

        # FlashRank 결과는 score 내림차순 정렬된 passage 목록
        reranked = [docs[r["id"]] for r in results[:top_k]]
        print(f"[FlashRank] {len(docs)}개 → {len(reranked)}개 재정렬 완료")
        return reranked

    except Exception as e:
        print(f"[FlashRank] 재정렬 실패 (RRF 순위 사용): {e}")
        return docs[:top_k]


# ══════════════════════════════════════════════
# Stage 5: LLM Listwise Re-ranking
# ══════════════════════════════════════════════

_LLM_RERANK_SYSTEM = (
    "당신은 검색 관련도 평가 전문가다. "
    "주어진 문서 목록을 검색 쿼리와의 관련도가 높은 순서로 재정렬하라. "
    "관련도가 높은 순서대로 문서 번호를 쉼표로 구분해 출력하라. "
    "번호 외 다른 텍스트는 절대 출력하지 마라. 예시: 3,1,4,2,5"
)


def _rerank_with_llm(
    llm: ChatOpenAI,
    query: str,
    docs: list[Document],
    top_k: int,
) -> list[Document]:
    """
    LLM 리스트와이즈 재정렬 — 후보 문서를 한 번의 LLM 호출로 최종 정렬한다.

    - FlashRank(크로스인코더)가 처리하지 못하는 심층 의미 이해를 LLM이 수행
    - 모든 후보를 한 번에 제공해 문서 간 상대 관련도를 비교
    - 실패 시 FlashRank 순위 그대로 반환 (graceful degradation)
    """
    if not docs:
        return docs

    try:
        doc_texts = []
        for i, doc in enumerate(docs):
            content = (doc.metadata.get("original_content") or doc.page_content)[:600]
            doc_texts.append(f"[{i + 1}] {content}")

        prompt = (
            f"검색 쿼리: {query}\n\n"
            "문서 목록:\n"
            + "\n\n".join(doc_texts)
            + "\n\n관련도 순서 (번호만, 쉼표 구분):"
        )

        resp  = llm.invoke([
            SystemMessage(content=_LLM_RERANK_SYSTEM),
            HumanMessage(content=prompt),
        ])

        raw_order = [
            int(x.strip()) - 1
            for x in resp.content.strip().split(",")
            if x.strip().isdigit()
        ]

        seen:     set[int]      = set()
        reranked: list[Document] = []
        for idx in raw_order:
            if 0 <= idx < len(docs) and idx not in seen:
                reranked.append(docs[idx])
                seen.add(idx)
        # LLM이 누락한 문서는 원래 순서로 추가
        for i, doc in enumerate(docs):
            if i not in seen:
                reranked.append(doc)

        result = reranked[:top_k]
        print(f"[LLM Rerank] {len(docs)}개 → {len(result)}개 최종 재정렬 완료")
        return result

    except Exception as e:
        print(f"[LLM Rerank] 재정렬 실패 (FlashRank 순위 사용): {e}")
        return docs[:top_k]


# ══════════════════════════════════════════════
# Full Hybrid Search Pipeline (Stage 1~5 통합)
# ══════════════════════════════════════════════

def _full_hybrid_search(
    llm: ChatOpenAI,
    vectorstore: Chroma,
    bm25_data: dict | None,
    original_query: str,
    chroma_filter: dict,
    bm25_filter_fn: "callable[[dict], bool]",
    k_final: int,
    k_per_query: int = 10,
    n_queries: int = 2,
    use_flashrank: bool = True,
    use_llm_rerank: bool = True,
) -> list[Document]:
    """
    완전한 5단계 Hybrid Search 파이프라인.

    Parameters
    ----------
    k_final        : 최종 반환 문서 수
    k_per_query    : 각 쿼리의 Semantic/BM25 검색 수 (과다 수집 후 압축)
    n_queries      : 원본 외 추가 쿼리 수 (multi-query expansion)
    use_flashrank  : Stage 4 FlashRank 크로스인코더 활성화 여부
    use_llm_rerank : Stage 5 LLM Listwise Reranking 활성화 여부
    """
    # Stage 1: Multi-Query Expansion
    queries = _expand_query(llm, original_query, n=n_queries)

    # Stage 2 + 3: 각 쿼리의 Semantic + BM25 → 전체 RRF
    all_ranked_lists: list[list[Document]] = []

    for q in queries:
        sem_docs = _semantic_search(vectorstore, q, chroma_filter, k=k_per_query)
        all_ranked_lists.append(sem_docs)

        if bm25_data:
            bm25_docs = _bm25_search(bm25_data, q, bm25_filter_fn, k=k_per_query)
            if bm25_docs:
                all_ranked_lists.append(bm25_docs)

    # 후보 풀: 최종 반환 수의 3배 확보
    candidate_pool = _rrf_fuse(all_ranked_lists, top_n=k_final * 3)

    # Stage 4: FlashRank Re-ranking → LLM 입력용 중간 풀 (k_final × 2)
    flashrank_top_k = k_final * 2 if use_llm_rerank else k_final
    if use_flashrank and candidate_pool:
        intermediate = _rerank_with_flashrank(original_query, candidate_pool, top_k=flashrank_top_k)
    else:
        intermediate = candidate_pool[:flashrank_top_k]

    # Stage 5: LLM Listwise Re-ranking → 최종 top-k
    if use_llm_rerank and intermediate:
        return _rerank_with_llm(llm, original_query, intermediate, top_k=k_final)

    return intermediate[:k_final]


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
# RAG 3단계: 유형별 컨텍스트 검색
# ──────────────────────────────────────────────

def retrieve_group_context(
    vectorstore: Chroma,
    type_names: list[str],
    llm: ChatOpenAI,
    bm25_data: dict | None = None,
) -> str:
    """
    AX Compass 유형별 특성 문서를 Full Hybrid Search로 가져온다.

    Pipeline: Multi-Query → (Semantic + BM25) × n → RRF → FlashRank
    """
    query = (
        f"{', '.join(type_names)} 유형의 "
        "AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    )
    k_final = max(len(type_names) * 2, 4)

    chroma_filter  = {"$and": [
        {"doc_type":  {"$eq": "ax_compass"}},
        {"type_name": {"$in": type_names}},
    ]}
    bm25_filter_fn = lambda m: (
        m.get("doc_type") == "ax_compass" and m.get("type_name") in type_names
    )

    docs = _full_hybrid_search(
        llm, vectorstore, bm25_data,
        original_query=query,
        chroma_filter=chroma_filter,
        bm25_filter_fn=bm25_filter_fn,
        k_final=k_final,
        k_per_query=k_final + 4,
    )

    # 유형별 그룹화 출력 (LLM 프롬프트 구조화)
    grouped: dict[str, list[str]] = {t: [] for t in type_names}
    for d in docs:
        tname   = d.metadata.get("type_name", "")
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
# RAG 3단계: 커리큘럼 예시 검색
# ──────────────────────────────────────────────

def retrieve_curriculum_examples(
    vectorstore: Chroma,
    query: str,
    llm: ChatOpenAI,
    bm25_data: dict | None = None,
    k: int = 3,
) -> str:
    """
    커리큘럼 예시를 Full Hybrid Search로 가져온다.
    출처(파일명/섹션)를 헤더로 추가해 LLM이 구조를 파악하기 쉽게 한다.
    """
    chroma_filter  = {"doc_type": {"$eq": "curriculum_example"}}
    bm25_filter_fn = lambda m: m.get("doc_type") == "curriculum_example"

    docs = _full_hybrid_search(
        llm, vectorstore, bm25_data,
        original_query=query,
        chroma_filter=chroma_filter,
        bm25_filter_fn=bm25_filter_fn,
        k_final=k,
        k_per_query=k + 4,
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
    커리큘럼 생성 LCEL 체인을 반환한다.
    LLM을 Multi-Query Expansion과 커리큘럼 생성에 공통으로 사용한다.

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

        print("[RAG] 유형별 컨텍스트 검색 중... (Full Hybrid: Multi-Query + RRF + FlashRank)")
        ctx_a = retrieve_group_context(vectorstore, ga["types"], llm, bm25_data)
        ctx_b = retrieve_group_context(vectorstore, gb["types"], llm, bm25_data)
        ctx_c = retrieve_group_context(vectorstore, gc["types"], llm, bm25_data)

        print("[RAG] 커리큘럼 예시 검색 중... (Full Hybrid)")
        curriculum_examples = retrieve_curriculum_examples(
            vectorstore,
            f"{topic} {level} 기업 AI 교육 커리큘럼",
            llm,
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

            [AX Compass 유형별 특성 — Hybrid Search 결과]
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
