import os

# ──────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH        = os.path.join(BASE_DIR, ".env")
DATA_DIR        = os.path.join(BASE_DIR, "Data")
PDF_PATH        = os.path.join(DATA_DIR, "AXCompass.pdf")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_v2"


# ──────────────────────────────────────────────
# 환경 변수 로드
# ──────────────────────────────────────────────

def load_env_file() -> None:
    """프로젝트 루트의 .env 파일을 읽어 환경 변수로 등록한다."""
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def require_api_key() -> str:
    """OPENAI_API_KEY를 반환한다. 없으면 ValueError를 발생시킨다."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            f"OPENAI_API_KEY를 찾을 수 없습니다. {ENV_PATH} 파일을 확인해주세요."
        )
    return key
