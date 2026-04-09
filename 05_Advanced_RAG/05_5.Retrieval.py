import importlib.util
import os
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
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
# RAG 3단계: 시맨틱 검색
# ──────────────────────────────────────────────

def retrieve_group_context(vectorstore: Chroma, type_names: list[str]) -> str:
    """
    유형별 AX Compass 문서를 검색한다.
    - type_names에 포함된 유형 섹션 청크만 필터링 (section 메타데이터 활용)
    - 유형당 최소 2개 청크를 확보하기 위해 k = len(type_names) * 2
    """
    query = (
        f"{', '.join(type_names)} 유형의 "
        "AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    )
    k = max(len(type_names) * 2, 4)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"$and": [
                {"doc_type":  {"$eq": "ax_compass"}},
                {"type_name": {"$in": type_names}},
            ]},
        },
    )
    docs = retriever.invoke(query)

    # 유형별로 그룹화하여 출력 (section 메타데이터 활용)
    grouped: dict[str, list[str]] = {t: [] for t in type_names}
    for d in docs:
        tname = d.metadata.get("type_name", "")
        if tname in grouped:
            grouped[tname].append(d.page_content)

    parts = []
    for tname, contents in grouped.items():
        if contents:
            parts.append(f"[{tname}]\n" + "\n\n".join(contents))

    return "\n\n".join(parts) if parts else "\n\n".join(d.page_content for d in docs)


def retrieve_curriculum_examples(vectorstore: Chroma, query: str, k: int = 3) -> str:
    """
    커리큘럼 예시를 검색한다.
    - section 메타데이터를 출처 표시에 활용하여 LLM이 구조를 파악하기 쉽게 함
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": {"$eq": "curriculum_example"}},
        },
    )
    docs = retriever.invoke(query)

    parts = []
    for d in docs:
        course  = d.metadata.get("course_name", "")
        section = d.metadata.get("section", "")
        header  = f"[출처: {course}" + (f" / {section}" if section else "") + "]"
        parts.append(f"{header}\n{d.page_content}")

    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────
# RAG 4단계: LCEL 체인
# ──────────────────────────────────────────────

def build_chain(vectorstore: Chroma, api_key: str):
    llm            = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(CurriculumPlan)

    def retrieve_and_build_messages(input_dict: dict) -> list:
        conversation = input_dict["conversation"]
        groups       = input_dict["groups"]
        total_hours  = input_dict["total_hours"]
        topic        = input_dict.get("topic", "")
        level        = input_dict.get("level", "")
        ga, gb, gc   = groups["group_a"], groups["group_b"], groups["group_c"]

        print("[RAG] 유형별 컨텍스트 검색 중...")
        ctx_a = retrieve_group_context(vectorstore, ga["types"])
        ctx_b = retrieve_group_context(vectorstore, gb["types"])
        ctx_c = retrieve_group_context(vectorstore, gc["types"])

        print("[RAG] 커리큘럼 예시 검색 중...")
        curriculum_examples = retrieve_curriculum_examples(
            vectorstore, f"{topic} {level} 기업 AI 교육 커리큘럼"
        )

        chat_history  = [m for m in conversation if not isinstance(m, SystemMessage)]
        theory_hours  = round(total_hours * 0.65)
        group_hours   = total_hours - theory_hours

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

            [AX Compass 유형별 특성 — 벡터 DB 검색 결과]
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
