import json
import os
from textwrap import dedent

import streamlit as st
import requests

# ──────────────────────────────────────────────
# 환경 설정
# ──────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_env_file(env_path=ENV_PATH):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# ──────────────────────────────────────────────
# 페이지 설정 & 스타일
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI 커리큘럼 설계 챗봇",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ── 기본 배경 ── */
    .stApp { background-color: #ffffff; color: #111111; }

    /* ── 사이드바 ── */
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }

    /* ── 입력창 ── */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: #fafafa;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        color: #111111;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #111111;
        box-shadow: 0 0 0 2px rgba(0,0,0,0.08);
    }

    /* ── 버튼 ── */
    .stButton > button {
        background-color: #111111;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background-color: #333333;
        color: #ffffff;
    }
    .stButton > button:disabled {
        background-color: #cccccc;
        color: #888888;
    }

    /* ── 채팅 메시지 ── */
    .chat-bubble-user {
        background: #111111;
        color: #ffffff;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-bubble-bot {
        background: #f2f2f2;
        color: #111111;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        max-width: 80%;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #e0e0e0;
    }

    /* ── 커리큘럼 카드 ── */
    .curriculum-header {
        background: #111111;
        color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .curriculum-header h2 { margin: 0; font-size: 1.3rem; font-weight: 700; }
    .curriculum-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.75; }

    .session-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #111111;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .session-card h4 { margin: 0 0 0.5rem; font-size: 1rem; font-weight: 700; color: #111111; }

    .tag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.4rem; }
    .tag {
        background: #111111;
        color: #ffffff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .tag-outline {
        background: #ffffff;
        color: #111111;
        border: 1px solid #cccccc;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
    }

    .outcome-box {
        background: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .note-box {
        background: #fffbe6;
        border: 1px solid #ffe58f;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #555;
    }

    /* ── 진행 단계 표시기 ── */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 1.2rem;
    }
    .step-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #dddddd;
    }
    .step-dot.active { background: #111111; }
    .step-dot.done   { background: #555555; }

    /* ── 구분선 ── */
    hr { border: none; border-top: 1px solid #e8e8e8; margin: 1rem 0; }

    /* ── 숨기기 ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────

QUESTIONS = [
    ("company_name", "회사명 또는 팀 이름을 알려주세요."),
    ("goal",         "이번 교육의 목표는 무엇인가요?"),
    ("audience",     "교육 대상자는 누구인가요?"),
    ("level",        "현재 AI 활용 수준을 알려주세요. (예: 입문, 초급, 중급)"),
    ("duration",     "교육 기간 또는 총 시간은 어떻게 되나요?"),
    ("topic",        "원하는 핵심 주제는 무엇인가요?"),
    ("constraints",  "꼭 반영해야 할 조건이나 제한사항이 있나요?"),
]

def _reset_state():
    st.session_state.messages   = []
    st.session_state.step       = 0
    st.session_state.answers    = {}
    st.session_state.curriculum = None
    st.session_state.generating = False
    st.session_state.greeted    = False

if "greeted" not in st.session_state:
    _reset_state()

# ──────────────────────────────────────────────
# 백엔드 API 통신 헬퍼
# ──────────────────────────────────────────────

def generate_curriculum(messages):
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
    
    # 프론트엔드의 "bot" 역할을 백엔드가 기대하는 "assistant"로 변경
    formatted_messages = []
    for m in messages:
        role = "assistant" if m["role"] == "bot" else m["role"]
        formatted_messages.append({"role": role, "content": m["content"]})
        
    try:
        response = requests.post(
            f"{backend_url}/api/generate",
            json={"messages": formatted_messages},
            timeout=180
        )
        response.raise_for_status()
        
        # FastAPI 서버는 {"curriculum": {...}} 형태의 JSON을 반환함
        data = response.json()
        if "curriculum" in data:
            return data["curriculum"]
        else:
            raise ValueError("백엔드 응답에 커리큘럼 데이터가 없습니다.")
            
    except Exception as e:
        raise ValueError(f"백엔드 연결/생성 오류: {e}")

# ──────────────────────────────────────────────
# 채팅 메시지 렌더러
# ──────────────────────────────────────────────

def render_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────
# 커리큘럼 시각화
# ──────────────────────────────────────────────

def render_curriculum(curriculum):
    st.markdown(
        f"""
        <div class="curriculum-header">
            <h2>📚 {curriculum['program_title']}</h2>
            <p>{curriculum['target_summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📘 공통 이론 세션")
    for i, session in enumerate(curriculum.get("theory_sessions", []), start=1):
        goals_html = "".join(f'<span class="tag">{g}</span>' for g in session.get("goals", []))
        activities_html = "".join(f'<span class="tag-outline">{a}</span>' for a in session.get("activities", []))
        st.markdown(
            f"""
            <div class="session-card">
                <h4>{i}회차 &nbsp;·&nbsp; {session.get('title', '')} <span style="font-size:0.85rem;color:#888;">({session.get('duration_hours', 0)}시간)</span></h4>
                <p style="font-size:0.82rem;color:#666;margin:0 0 0.4rem;">목표</p>
                <div class="tag-list">{goals_html}</div>
                <p style="font-size:0.82rem;color:#666;margin:0.7rem 0 0.4rem;">활동</p>
                <div class="tag-list">{activities_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 맞춤형 실습 세션 (그룹별)")
    for group in curriculum.get("group_sessions", []):
        st.markdown(f"#### 👥 {group.get('group_name', '')} <span style=\"font-size:1rem;color:#555;\">({group.get('target_types', '')} / {group.get('participant_count', 0)}명)</span>", unsafe_allow_html=True)
        st.markdown(f"<p style=\"font-size:0.95rem;color:#444;margin-bottom:1rem;\">{group.get('focus_description', '')}</p>", unsafe_allow_html=True)
        
        for i, session in enumerate(group.get("sessions", []), start=1):
            goals_html = "".join(f'<span class="tag">{g}</span>' for g in session.get("goals", []))
            activities_html = "".join(f'<span class="tag-outline">{a}</span>' for a in session.get("activities", []))
            st.markdown(
                f"""
                <div class="session-card" style="margin-left: 1rem; border-left-color: #555;">
                    <h4>{i}회차 &nbsp;·&nbsp; {session.get('title', '')} <span style="font-size:0.85rem;color:#888;">({session.get('duration_hours', 0)}시간)</span></h4>
                    <p style="font-size:0.82rem;color:#666;margin:0 0 0.4rem;">목표</p>
                    <div class="tag-list">{goals_html}</div>
                    <p style="font-size:0.82rem;color:#666;margin:0.7rem 0 0.4rem;">활동</p>
                    <div class="tag-list">{activities_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 예상 결과")
        for outcome in curriculum["expected_outcomes"]:
            st.markdown(
                f'<div class="outcome-box">✅ {outcome}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### 참고 사항")
        for note in curriculum["notes"]:
            st.markdown(
                f'<div class="note-box">⚠️ {note}</div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 AI 커리큘럼 챗봇")
    st.markdown("기업 요구사항을 대화로 입력하면\nOpenAI API가 커리큘럼 초안을 자동 생성합니다.")
    st.markdown("---")

    # 진행 단계 표시
    total = len(QUESTIONS)
    current_step = st.session_state.step

    st.markdown("**진행 단계**")
    for i, (_, q) in enumerate(QUESTIONS):
        label_step = i + 1
        if current_step > label_step:
            icon = "✅"
        elif current_step == label_step:
            icon = "▶️"
        else:
            icon = "○"
        short = q[:18] + "…" if len(q) > 18 else q
        st.markdown(f"{icon} **{label_step}.** {short}")

    st.markdown("---")

    if st.button("대화 초기화", use_container_width=True):
        _reset_state()
        st.rerun()

    # 수집된 요구사항 미리보기
    if st.session_state.answers:
        st.markdown("**수집된 요구사항**")
        label_map = {
            "company_name": "회사/팀",
            "goal":         "교육 목표",
            "audience":     "교육 대상자",
            "level":        "현재 수준",
            "duration":     "교육 기간",
            "topic":        "핵심 주제",
            "constraints":  "제한사항",
        }
        for k, v in st.session_state.answers.items():
            st.markdown(f"**{label_map.get(k, k)}**: {v}")

# ──────────────────────────────────────────────
# 메인 레이아웃
# ──────────────────────────────────────────────

st.markdown("# AI 커리큘럼 설계 챗봇")
st.markdown("기업 맞춤형 교육 커리큘럼을 대화로 만들어보세요.")
st.markdown("---")

# ── 첫 인사 (최초 1회만) ──────────────────────────
if not st.session_state.greeted:
    st.session_state.messages = [{
        "role": "bot",
        "content": (
            "안녕하세요! 기업 교육용 AI 커리큘럼 설계 챗봇입니다.\n\n"
            "몇 가지 질문에 답해주시면 맞춤형 커리큘럼 초안을 바로 생성해 드릴게요.\n\n"
            f"**{QUESTIONS[0][1]}**"
        ),
    }]
    st.session_state.step    = 1
    st.session_state.greeted = True

render_messages()

# ── 커리큘럼 결과 ─────────────────────────────────
if st.session_state.curriculum:
    st.markdown("---")
    render_curriculum(st.session_state.curriculum)
    st.markdown("---")
    if st.button("새 커리큘럼 만들기"):
        _reset_state()
        st.rerun()

# ── 커리큘럼 생성 중 ──────────────────────────────
elif st.session_state.generating:
    with st.spinner("커리큘럼을 생성하고 있습니다..."):
        try:
            curriculum = generate_curriculum(st.session_state.messages)
            st.session_state.curriculum = curriculum
            st.session_state.generating = False
            st.session_state.messages = st.session_state.messages + [{
                "role": "bot",
                "content": "✅ 커리큘럼 초안이 완성되었습니다! 아래에서 확인해보세요.",
            }]
        except Exception as e:
            st.session_state.generating = False
            st.session_state.messages = st.session_state.messages + [{
                "role": "bot",
                "content": f"❌ 오류가 발생했습니다: {e}",
            }]
    st.rerun()

# ── 질문 진행 중 → st.chat_input ─────────────────
elif st.session_state.step > 0 and st.session_state.step <= len(QUESTIONS):
    user_input = st.chat_input("답변을 입력하세요...")

    if user_input and user_input.strip():
        q_key, _ = QUESTIONS[st.session_state.step - 1]
        st.session_state.answers[q_key] = user_input.strip()

        # 리스트 재할당으로 Streamlit 변경 감지 보장
        st.session_state.messages = st.session_state.messages + [
            {"role": "user", "content": user_input.strip()}
        ]

        next_step = st.session_state.step + 1

        if next_step <= len(QUESTIONS):
            _, next_q = QUESTIONS[next_step - 1]
            st.session_state.messages = st.session_state.messages + [
                {"role": "bot", "content": next_q}
            ]
            st.session_state.step = next_step
        else:
            summary_lines = [
                f"**{label}**: {st.session_state.answers.get(k, '')}"
                for k, label in [
                    ("company_name", "회사/팀"),
                    ("goal",         "교육 목표"),
                    ("audience",     "교육 대상자"),
                    ("level",        "현재 수준"),
                    ("duration",     "교육 기간"),
                    ("topic",        "핵심 주제"),
                    ("constraints",  "제한사항"),
                ]
            ]
            st.session_state.messages = st.session_state.messages + [{
                "role": "bot",
                "content": "감사합니다! 입력하신 내용을 바탕으로 커리큘럼을 생성하겠습니다.\n\n" + "\n".join(summary_lines),
            }]
            st.session_state.step       = next_step
            st.session_state.generating = True

        st.rerun()   # 성공/실패 모두 rerun → 새 메시지 화면에 반영
