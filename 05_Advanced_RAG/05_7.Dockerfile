# ──────────────────────────────────────────────────────────────────────────────
# AX Compass 커리큘럼 챗봇 — Docker 배포
#
# 빌드:
#   docker build -f 05_Advanced_RAG/05_7.Dockerfile -t ax-curriculum .
#
# 실행:
#   docker run -p 8501:8501 \
#     -e OPENAI_API_KEY=sk-... \
#     -v $(pwd)/Data:/app/Data \
#     -v $(pwd)/vectorDB:/app/vectorDB \
#     ax-curriculum
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# 시스템 의존성 (unstructured[xlsx] 에 필요한 라이브러리 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 의존성 설치 (레이어 캐시 활용) ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 애플리케이션 복사 ─────────────────────────────────────────────────────────
# Data/ 와 vectorDB/ 는 런타임에 볼륨으로 마운트하므로 복사하지 않음
COPY 05_Advanced_RAG/ ./05_Advanced_RAG/

# ── Streamlit 설정 ────────────────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
CMD ["python", "05_Advanced_RAG/05_8.FastAPI.py"]
