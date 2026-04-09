"""
Streamlit 프론트엔드 — FastAPI 백엔드 API 클라이언트

백엔드 URL은 Streamlit Cloud 시크릿에서 읽어옵니다:
  [secrets.toml]
  BACKEND_URL = "https://xxxx.ngrok-free.app"
"""

import requests
import streamlit as st

# ──────────────────────────────────────────────
# 백엔드 URL
# ──────────────────────────────────────────────

def _backend_url() -> str:
    try:
        return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        return st.session_state.get("backend_url", "").rstrip("/")


# ──────────────────────────────────────────────
# 페이지 설정 & 스타일
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AX Compass 커리큘럼 챗봇 (RAG)",
    page_icon="🧭",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #ffffff; color: #111111; }

section[data-testid="stSidebar"] {
    background: #f5f5f5;
    border-right: 1px solid #e0e0e0;
}

.stTextInput > div > div > input,
.stTextArea textarea {
    background: #fafafa;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    color: #111111;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #111111;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.08);
}

.stButton > button {
    background: #111111;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    transition: background 0.15s;
}
.stButton > button:hover { background: #333333; color: #ffffff; }

/* 채팅 버블 */
.bubble-user {
    background: #111111;
    color: #ffffff;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.25rem 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.94rem;
    line-height: 1.55;
}
.bubble-bot {
    background: #f2f2f2;
    color: #111111;
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.25rem 0;
    max-width: 78%;
    font-size: 0.94rem;
    line-height: 1.55;
    border: 1px solid #e0e0e0;
}

/* 커리큘럼 헤더 */
.cv-header {
    background: #111111;
    color: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.cv-header h2 { margin: 0; font-size: 1.35rem; font-weight: 700; }
.cv-header p  { margin: 0.3rem 0 0; font-size: 0.88rem; opacity: 0.72; }

/* 세션 카드 */
.session-card {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #111111;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.session-card h4 { margin: 0 0 0.5rem; font-size: 1rem; font-weight: 700; color: #111111; }

/* 그룹 헤더 */
.group-header {
    background: #222222;
    color: #ffffff;
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    margin: 1rem 0 0.6rem;
    font-size: 0.95rem;
    font-weight: 600;
}

/* 태그 */
.tag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.4rem; }
.tag {
    background: #111111; color: #ffffff;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 500;
}
.tag-outline {
    background: #fff; color: #111111;
    border: 1px solid #cccccc;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.78rem;
}

/* 결과 / 참고 박스 */
.outcome-box {
    background: #f9f9f9; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
}
.note-box {
    background: #fffbe6; border: 1px solid #ffe58f;
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    font-size: 0.9rem; color: #555;
}

/* 시간 배지 */
.hour-badge {
    background: #333; color: #fff;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600;
    margin-left: 8px;
}

hr { border: none; border-top: 1px solid #e8e8e8; margin: 1rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────

def _reset_state():
    st.session_state.ui_messages = []   # [{"role": "user"|"bot", "content": "..."}]
    st.session_state.api_messages = []  # [{"role": "user"|"assistant", "content": "..."}]
    st.session_state.phase        = "init"
    st.session_state.curriculum   = None
    st.session_state.error        = None


if "phase" not in st.session_state:
    _reset_state()

# ──────────────────────────────────────────────
# API 호출 헬퍼
# ──────────────────────────────────────────────

def _api_chat(messages: list[dict]) -> dict:
    """POST /api/chat → {"content": str, "is_done": bool}"""
    url = f"{_backend_url()}/api/chat"
    resp = requests.post(url, json={"messages": messages}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _api_generate(messages: list[dict]) -> dict:
    """POST /api/generate → {"curriculum": dict}"""
    url = f"{_backend_url()}/api/generate"
    resp = requests.post(url, json={"messages": messages}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _api_health() -> bool:
    try:
        resp = requests.get(f"{_backend_url()}/health", timeout=5)
        return resp.ok
    except Exception:
        return False

# ──────────────────────────────────────────────
# 커리큘럼 시각화
# ──────────────────────────────────────────────

def _tags(items: list[str], cls: str = "tag") -> str:
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


def render_curriculum(c: dict):
    st.markdown(
        f"""<div class="cv-header">
               <h2>🧭 {c['program_title']}</h2>
               <p>{c['target_summary']}</p>
            </div>""",
        unsafe_allow_html=True,
    )

    # ── 공통 이론 수업 ──
    st.markdown("### 📖 공통 이론 수업")
    theory_total = sum(s["duration_hours"] for s in c["theory_sessions"])
    st.caption(f"총 {theory_total}시간 · {len(c['theory_sessions'])}개 회차")

    for i, s in enumerate(c["theory_sessions"], 1):
        goals_html = _tags(s["goals"])
        acts_html  = _tags(s["activities"], "tag-outline")
        st.markdown(
            f"""<div class="session-card">
                  <h4>{i}회차 &nbsp;·&nbsp; {s['title']}
                      <span class="hour-badge">{s['duration_hours']}h</span></h4>
                  <p style="font-size:0.82rem;color:#666;margin:0 0 0.35rem;">목표</p>
                  <div class="tag-list">{goals_html}</div>
                  <p style="font-size:0.82rem;color:#666;margin:0.65rem 0 0.35rem;">활동</p>
                  <div class="tag-list">{acts_html}</div>
               </div>""",
            unsafe_allow_html=True,
        )

    # ── 그룹별 실습 ──
    st.markdown("### 👥 그룹별 맞춤 실습")
    group_cols = st.columns(len(c["group_sessions"]))

    for col, grp in zip(group_cols, c["group_sessions"]):
        with col:
            st.markdown(
                f"""<div class="group-header">
                      {grp['group_name']} &nbsp;·&nbsp; {grp['target_types']}
                      &nbsp;<span style="opacity:0.7;font-size:0.85rem;">{grp['participant_count']}명</span>
                    </div>
                    <p style="font-size:0.85rem;color:#555;margin:0 0 0.6rem;">{grp['focus_description']}</p>""",
                unsafe_allow_html=True,
            )
            for i, s in enumerate(grp["sessions"], 1):
                goals_html = _tags(s["goals"])
                acts_html  = _tags(s["activities"], "tag-outline")
                st.markdown(
                    f"""<div class="session-card">
                          <h4>{i}회차 &nbsp;·&nbsp; {s['title']}
                              <span class="hour-badge">{s['duration_hours']}h</span></h4>
                          <div class="tag-list" style="margin-bottom:0.5rem;">{goals_html}</div>
                          <div class="tag-list">{acts_html}</div>
                       </div>""",
                    unsafe_allow_html=True,
                )

    # ── 결과 / 참고 ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ 예상 결과")
        for o in c["expected_outcomes"]:
            st.markdown(f'<div class="outcome-box">{o}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### ⚠️ 참고 사항")
        for n in c["notes"]:
            st.markdown(f'<div class="note-box">{n}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 채팅 메시지 렌더
# ──────────────────────────────────────────────

def render_messages():
    for msg in st.session_state.ui_messages:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        st.markdown(
            f'<div class="{cls}">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧭 AX 커리큘럼 챗봇")
    st.markdown("AX Compass 진단 결과를 반영한\n**그룹 맞춤형** 교육 커리큘럼을 생성합니다.")
    st.markdown("---")

    # 백엔드 URL 직접 입력 (secrets가 없을 때 대비)
    try:
        _ = st.secrets["BACKEND_URL"]
        st.caption("✅ 백엔드 URL: secrets에서 로드됨")
    except Exception:
        manual_url = st.text_input(
            "백엔드 URL (ngrok)",
            value=st.session_state.get("backend_url", ""),
            placeholder="https://xxxx.ngrok-free.app",
        )
        if manual_url:
            st.session_state.backend_url = manual_url

    st.markdown("---")

    phase = st.session_state.phase
    steps = [
        ("정보 수집", "collecting"),
        ("커리큘럼 생성", "generating"),
        ("완료", "done"),
    ]
    step_keys = [s[1] for s in steps]
    for label, p in steps:
        if phase == p:
            icon = "▶️"
        elif p in step_keys and phase in step_keys and step_keys.index(phase) > step_keys.index(p):
            icon = "✅"
        else:
            icon = "○"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        _reset_state()
        st.rerun()

# ──────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────

st.markdown("# 🧭 AX Compass 커리큘럼 챗봇")
st.markdown("AX Compass 유형 진단 결과를 반영해 **그룹별 맞춤형** AI 교육 커리큘럼을 설계합니다.")
st.markdown("---")

# ── 초기 인사 (최초 1회만) ──────────────────────
if st.session_state.phase == "init":
    backend = _backend_url()
    if not backend:
        st.warning("사이드바에서 백엔드 URL을 입력해주세요.")
        st.stop()

    with st.spinner("백엔드에 연결 중..."):
        try:
            result = _api_chat([])   # 빈 메시지 → 첫 질문 요청
            first_msg = result["content"]
            st.session_state.api_messages = [{"role": "assistant", "content": first_msg}]
            st.session_state.ui_messages  = [{"role": "bot",       "content": first_msg}]
            st.session_state.phase = "collecting"
        except Exception as e:
            st.error(f"백엔드 연결 실패: {e}")
            st.stop()

# ── 메시지 렌더 ──────────────────────────────────
render_messages()

# ── 에러 표시 ────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

# ── 커리큘럼 결과 ────────────────────────────────
if st.session_state.phase == "done" and st.session_state.curriculum:
    st.markdown("---")
    render_curriculum(st.session_state.curriculum)
    st.markdown("---")
    if st.button("새 커리큘럼 만들기"):
        _reset_state()
        st.rerun()

# ── 커리큘럼 생성 중 ─────────────────────────────
elif st.session_state.phase == "generating":
    with st.spinner("커리큘럼을 생성하고 있습니다... (약 30초~1분 소요)"):
        try:
            data = _api_generate(st.session_state.api_messages)
            st.session_state.curriculum = data["curriculum"]
            st.session_state.ui_messages = st.session_state.ui_messages + [{
                "role": "bot",
                "content": "✅ 커리큘럼 초안이 완성되었습니다! 아래에서 확인하세요.",
            }]
            st.session_state.phase = "done"
        except Exception as e:
            st.session_state.error = f"생성 중 오류: {e}"
            st.session_state.phase = "collecting"
    st.rerun()

# ── 정보 수집 입력창 ─────────────────────────────
elif st.session_state.phase == "collecting":
    user_input = st.chat_input("답변을 입력하세요...")

    if user_input and user_input.strip():
        text = user_input.strip()

        st.session_state.api_messages = st.session_state.api_messages + [{"role": "user", "content": text}]
        st.session_state.ui_messages  = st.session_state.ui_messages  + [{"role": "user", "content": text}]

        try:
            result = _api_chat(st.session_state.api_messages)
            bot_content = result["content"]
            is_done     = result["is_done"]

            st.session_state.api_messages = st.session_state.api_messages + [{"role": "assistant", "content": bot_content}]
            st.session_state.ui_messages  = st.session_state.ui_messages  + [{"role": "bot", "content": bot_content}]

            if is_done:
                st.session_state.ui_messages = st.session_state.ui_messages + [{
                    "role": "bot",
                    "content": "정보 수집이 완료되었습니다. 커리큘럼을 생성하고 있습니다...",
                }]
                st.session_state.phase = "generating"
        except Exception as e:
            st.session_state.error = f"응답 오류: {e}"
        finally:
            st.rerun()
