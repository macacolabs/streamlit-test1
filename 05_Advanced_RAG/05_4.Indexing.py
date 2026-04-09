"""
RAG 인덱싱 파이프라인 (고도화)

개선 사항:
  1. 문서 종류별 인덱싱 전략 분리
       - AXCompass.pdf  : 유형 섹션 경계 기반 사전 분리 → 청크가 두 유형에 걸치지 않음
       - 커리큘럼 PDF   : Day/모듈/챕터 헤딩 구조 보존 후 청킹
       - Excel          : 헤더+행 쌍으로 재구성 후 청킹 (테이블 의미 보존)
  2. 청킹 전 문서 구조 보존 전처리
       - 빈 줄 정규화, 하이픈 줄바꿈 복원, 페이지 헤더/푸터 제거
  3. 검색에 유리한 메타데이터 확장
       - file_hash, chunk_index, total_chunks, section,
         page_start, char_count, keywords, indexed_at
  4. 증분 인덱싱 지원
       - 파일별 SHA-256 해시 기반으로 변경된 파일만 재인덱싱
       - setup_vector_store(force_reindex=True) 로 전체 재구축 가능
"""

import hashlib
import importlib.util
import os
import re
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────
# 형제 모듈 동적 임포트
# ──────────────────────────────────────────────

def _load_sibling(filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_auth    = _load_sibling("05_3.Auth.py")
_schemas = _load_sibling("05_2.Schemas.py")

DATA_DIR        = _auth.DATA_DIR
PDF_PATH        = _auth.PDF_PATH
VECTOR_DB_PATH  = _auth.VECTOR_DB_PATH
COLLECTION_NAME = _auth.COLLECTION_NAME
TYPE_INFO       = _schemas.TYPE_INFO
TYPE_MARKERS    = _schemas.TYPE_MARKERS


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """파일의 SHA-256 해시를 반환한다 (증분 인덱싱 키)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]   # 앞 16자리로 축약


# 한국어 불용어 (키워드 추출용)
_STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "및", "을", "를", "가", "은",
    "는", "의", "에", "로", "으로", "하다", "있다", "되다", "하는", "있는",
    "되는", "한", "그리고", "또한", "통해", "위해", "대한", "따라", "기반",
    "위한", "통한", "대해", "관련", "경우", "때문", "이후", "이전", "이상",
}


def _extract_keywords(text: str, n: int = 6) -> str:
    """청크 내 빈도 상위 n개 한국어 명사(2글자 이상)를 추출한다."""
    tokens = re.findall(r"[가-힣]{2,}", text)
    freq: dict[str, int] = {}
    for t in tokens:
        if t not in _STOPWORDS:
            freq[t] = freq.get(t, 0) + 1
    top = sorted(freq, key=lambda x: freq[x], reverse=True)[:n]
    return ",".join(top)


def _clean_pdf_text(text: str) -> str:
    """
    PDF 추출 텍스트 공통 노이즈 제거.
      - 하이픈 줄바꿈 복원 (영단어/숫자)
      - 연속 공백·빈 줄 정규화
      - 페이지 번호만 있는 줄 제거
    """
    # 하이픈 줄바꿈 (예: "stra-\ntegy" → "strategy")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 페이지 번호만 있는 줄 제거 ("  3  " 형태)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # 3개 이상 빈 줄 → 2개
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 줄 선두/말미 공백 정리
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def _tag_chunks(chunks: list[Document], source_file: str,
                file_hash: str, base_meta: dict) -> list[Document]:
    """
    청크 리스트에 공통 메타데이터를 주입한다.
      - chunk_index, total_chunks, char_count, keywords, indexed_at
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        kw = _extract_keywords(chunk.page_content)
        chunk.metadata.update({
            **base_meta,
            "source_file":  source_file,
            "file_hash":    file_hash,
            "chunk_index":  i,
            "total_chunks": total,
            "char_count":   len(chunk.page_content),
            "keywords":     kw,
            "indexed_at":   now,
        })
    return chunks


# ──────────────────────────────────────────────
# 전략 1: AXCompass.pdf — 유형 섹션 경계 기반 분리
# ──────────────────────────────────────────────

# AX Compass 전용 splitter: 섹션 내부에서만 분할
_AX_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", ".", " "],
)


def _load_ax_compass(pdf_path: str, fhash: str) -> list[Document]:
    """
    1) 전체 텍스트 병합
    2) TYPE_MARKERS로 유형 섹션 경계 감지 (사전 분리)
    3) 각 섹션 내부에서만 청킹 → 청크가 두 유형에 걸치지 않음
    4) 유형별 메타데이터 사전 부착
    """
    pages = PyPDFLoader(pdf_path).load()
    full_text = _clean_pdf_text("\n".join(p.page_content for p in pages))

    # 섹션 경계 위치 탐색
    marker_positions: list[tuple[int, str]] = []
    for type_name, marker in TYPE_MARKERS.items():
        idx = full_text.find(marker)
        if idx >= 0:
            marker_positions.append((idx, type_name))

    marker_positions.sort(key=lambda x: x[0])

    # 유형 섹션이 전혀 없으면 단순 청킹으로 폴백
    if not marker_positions:
        print("[RAG][AX] 섹션 마커를 찾지 못했습니다. 기본 청킹으로 전환.")
        raw_chunks = _AX_SPLITTER.split_documents(pages)
        for c in raw_chunks:
            c.metadata["doc_type"] = "ax_compass"
        return _tag_chunks(raw_chunks, "AXCompass.pdf", fhash,
                           {"doc_type": "ax_compass"})

    # 섹션별 Document 생성
    section_docs: list[Document] = []
    for i, (start, type_name) in enumerate(marker_positions):
        end  = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(full_text)
        text = full_text[start:end].strip()
        if not text:
            continue
        info = TYPE_INFO.get(type_name, {})
        section_docs.append(Document(
            page_content=text,
            metadata={
                "doc_type":  "ax_compass",
                "type_name": type_name,
                "group":     info.get("group", ""),
                "english":   info.get("english", ""),
                "section":   type_name,
            },
        ))

    # 섹션 내부 청킹
    all_chunks: list[Document] = []
    for sec_doc in section_docs:
        chunks = _AX_SPLITTER.split_documents([sec_doc])
        all_chunks.extend(chunks)

    fname = os.path.basename(pdf_path)
    print(f"[RAG][AX] {fname}: {len(marker_positions)}개 섹션 → {len(all_chunks)}개 청크")
    return _tag_chunks(all_chunks, fname, fhash, {})


# ──────────────────────────────────────────────
# 전략 2: 커리큘럼 PDF — Day/모듈 헤딩 구조 보존
# ──────────────────────────────────────────────

# 커리큘럼 PDF 전용 splitter: 조금 더 큰 청크, 높은 overlap
_CURRICULUM_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", ".", " "],
)

# 커리큘럼 구조 헤딩 패턴 (Day X, 모듈 X, Chapter X, 숫자+점 제목)
_SECTION_PATTERN = re.compile(
    r"(?m)^(?:"
    r"Day\s*\d+|일차\s*\d+|"
    r"모듈\s*\d+|Module\s*\d+|"
    r"Chapter\s*\d+|챕터\s*\d+|"
    r"\d+\.\s+[^\n]{5,}"          # "1. 제목" 형태
    r")",
    re.IGNORECASE,
)


def _load_curriculum_pdf(pdf_path: str, fhash: str, course_name: str) -> list[Document]:
    """
    1) 페이지 단위로 로드 후 전체 텍스트 병합
    2) 섹션 헤딩 패턴으로 경계 감지 → 섹션별 Document 구성
    3) 섹션 내부 청킹 + 페이지 범위 메타데이터 부착
    """
    pages = PyPDFLoader(pdf_path).load()
    page_texts = [_clean_pdf_text(p.page_content) for p in pages]
    full_text  = "\n\n".join(page_texts)

    # 섹션 경계 탐색
    matches = list(_SECTION_PATTERN.finditer(full_text))

    all_chunks: list[Document] = []
    base_meta = {"doc_type": "curriculum_example", "course_name": course_name}

    if not matches:
        # 헤딩을 찾지 못하면 페이지 단위 청킹
        for p_idx, page in enumerate(pages):
            page.metadata.update({**base_meta, "page_start": p_idx + 1, "section": ""})
            chunks = _CURRICULUM_SPLITTER.split_documents([page])
            all_chunks.extend(chunks)
    else:
        # 섹션별 Document 생성
        boundaries = [(m.start(), m.group().strip()) for m in matches]
        for i, (start, heading) in enumerate(boundaries):
            end  = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            text = full_text[start:end].strip()
            if not text:
                continue
            # 해당 섹션이 속하는 대략적인 페이지 번호 추정
            char_offset = sum(len(t) + 2 for t in page_texts)
            page_num    = _estimate_page(full_text, start, page_texts)
            sec_doc = Document(
                page_content=text,
                metadata={**base_meta, "section": heading, "page_start": page_num},
            )
            chunks = _CURRICULUM_SPLITTER.split_documents([sec_doc])
            all_chunks.extend(chunks)

    fname = os.path.basename(pdf_path)
    print(f"[RAG][PDF] {fname}: {len(matches) or len(pages)}개 섹션 → {len(all_chunks)}개 청크")
    return _tag_chunks(all_chunks, fname, fhash, {})


def _estimate_page(full_text: str, char_pos: int, page_texts: list[str]) -> int:
    """문자 위치로부터 대략적인 페이지 번호를 추정한다 (1-based)."""
    cumulative = 0
    for i, pt in enumerate(page_texts):
        cumulative += len(pt) + 2   # "\n\n" join
        if char_pos < cumulative:
            return i + 1
    return len(page_texts)


# ──────────────────────────────────────────────
# 전략 3: 커리큘럼 Excel — 헤더+행 쌍 재구성
# ──────────────────────────────────────────────

_EXCEL_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    separators=["\n\n", "\n", " "],
)


def _load_curriculum_excel(xlsx_path: str, fhash: str, course_name: str) -> list[Document]:
    """
    openpyxl로 직접 읽어 헤더+행 쌍을 자연어로 재구성.
    예) 주제: AI 개요\n내용: LLM 기초\n시간: 2h
    테이블 의미(열 이름과 값의 관계)를 청크 내에 보존한다.
    """
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError:
        # fallback: UnstructuredExcelLoader
        from langchain_community.document_loaders import UnstructuredExcelLoader
        docs = UnstructuredExcelLoader(xlsx_path).load()
        fname = os.path.basename(xlsx_path)
        base  = {"doc_type": "curriculum_example", "course_name": course_name,
                 "section": "", "page_start": 1}
        for doc in docs:
            doc.metadata.update(base)
        chunks = _EXCEL_SPLITTER.split_documents(docs)
        print(f"[RAG][XLS] {fname}: fallback → {len(chunks)}개 청크")
        return _tag_chunks(chunks, fname, fhash, {})

    wb    = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    fname = os.path.basename(xlsx_path)
    base  = {"doc_type": "curriculum_example", "course_name": course_name}
    all_chunks: list[Document] = []

    for sheet_name in wb.sheetnames:
        ws   = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # 첫 행을 헤더로 사용 (None이면 열 번호로 대체)
        headers = [str(h).strip() if h is not None else f"열{i+1}"
                   for i, h in enumerate(rows[0])]

        # 행별로 "헤더: 값" 텍스트 구성
        row_texts: list[str] = []
        for row in rows[1:]:
            pairs = []
            for h, v in zip(headers, row):
                if v is not None and str(v).strip():
                    pairs.append(f"{h}: {str(v).strip()}")
            if pairs:
                row_texts.append("\n".join(pairs))

        if not row_texts:
            continue

        # 연속된 행을 묶어 하나의 Document로 구성 (chunk_size 범위 내)
        bucket: list[str] = []
        bucket_len = 0
        BUCKET_LIMIT = 1200   # 행 묶음 최대 문자 수

        for rt in row_texts:
            if bucket and bucket_len + len(rt) > BUCKET_LIMIT:
                text = "\n\n".join(bucket)
                all_chunks.append(Document(
                    page_content=text,
                    metadata={**base, "section": sheet_name, "page_start": 1},
                ))
                bucket, bucket_len = [], 0
            bucket.append(rt)
            bucket_len += len(rt)

        if bucket:
            text = "\n\n".join(bucket)
            all_chunks.append(Document(
                page_content=text,
                metadata={**base, "section": sheet_name, "page_start": 1},
            ))

    wb.close()
    print(f"[RAG][XLS] {fname}: {len(wb.sheetnames)}개 시트 → {len(all_chunks)}개 청크")
    return _tag_chunks(all_chunks, fname, fhash, {})


# ──────────────────────────────────────────────
# 문서 수집 — 파일별 (hash, chunks) 딕셔너리 반환
# ──────────────────────────────────────────────

def _collect_documents() -> dict[str, tuple[str, list[Document]]]:
    """
    Data/ 디렉터리를 순회하여 파일별로 (hash, chunks)를 반환.
    반환 형식: {filename: (file_hash, chunk_list)}

    증분 인덱싱에서 hash가 변경된 파일만 재처리하는 데 사용한다.
    """
    result: dict[str, tuple[str, list[Document]]] = {}

    # ── AXCompass.pdf ──
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"AXCompass.pdf 파일을 찾을 수 없습니다: {PDF_PATH}")

    fhash = _file_hash(PDF_PATH)
    fname = os.path.basename(PDF_PATH)
    result[fname] = (fhash, _load_ax_compass(PDF_PATH, fhash))

    # ── 커리큘럼 PDF ──
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith(".pdf") and fn != "AXCompass.pdf":
            fpath       = os.path.join(DATA_DIR, fn)
            fhash       = _file_hash(fpath)
            course_name = os.path.splitext(fn)[0]
            result[fn]  = (fhash, _load_curriculum_pdf(fpath, fhash, course_name))

    # ── 커리큘럼 Excel ──
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith(".xlsx"):
            fpath       = os.path.join(DATA_DIR, fn)
            fhash       = _file_hash(fpath)
            course_name = os.path.splitext(fn)[0]
            result[fn]  = (fhash, _load_curriculum_excel(fpath, fhash, course_name))

    return result


# ──────────────────────────────────────────────
# 증분 인덱싱 헬퍼
# ──────────────────────────────────────────────

def _get_indexed_hashes(vectorstore: Chroma) -> dict[str, str]:
    """
    벡터 DB에 현재 인덱싱된 파일별 해시를 반환한다.
    반환 형식: {source_file: file_hash}
    """
    if vectorstore._collection.count() == 0:
        return {}
    results = vectorstore._collection.get(include=["metadatas"])
    hashes: dict[str, str] = {}
    for meta in results["metadatas"]:
        fn = meta.get("source_file")
        fh = meta.get("file_hash")
        if fn and fh:
            hashes[fn] = fh     # 같은 파일의 모든 청크는 동일 해시
    return hashes


def _delete_file_chunks(vectorstore: Chroma, source_file: str) -> None:
    """특정 파일에서 생성된 청크를 벡터 DB에서 삭제한다."""
    vectorstore._collection.delete(
        where={"source_file": {"$eq": source_file}}
    )


# ──────────────────────────────────────────────
# RAG 2단계: 벡터 DB 구축 (증분 인덱싱)
# ──────────────────────────────────────────────

def setup_vector_store(api_key: str, force_reindex: bool = False) -> Chroma:
    """
    벡터 DB를 구축하거나 증분 업데이트한다.

    Parameters
    ----------
    api_key       : OpenAI API 키
    force_reindex : True이면 기존 컬렉션을 전부 삭제하고 재구축한다.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    # 강제 재인덱싱: 기존 컬렉션 전체 삭제
    if force_reindex and vectorstore._collection.count() > 0:
        print("[VectorDB] 강제 재인덱싱: 기존 컬렉션 전체 삭제...")
        vectorstore._collection.delete(where={})

    # 현재 인덱싱 상태 파악
    indexed_hashes = _get_indexed_hashes(vectorstore)

    # 파일별 처리
    print("[VectorDB] 문서 수집 및 증분 인덱싱 시작...")
    file_docs = _collect_documents()

    added   = 0
    updated = 0
    skipped = 0

    for fname, (fhash, chunks) in file_docs.items():
        existing_hash = indexed_hashes.get(fname)

        if existing_hash == fhash:
            # 변경 없음 → 스킵
            skipped += 1
            print(f"[VectorDB] 스킵 (변경 없음): {fname}")
            continue

        if existing_hash is not None:
            # 파일 변경 → 기존 청크 삭제 후 재추가
            print(f"[VectorDB] 업데이트: {fname} (해시 변경 감지)")
            _delete_file_chunks(vectorstore, fname)
            updated += 1
        else:
            print(f"[VectorDB] 신규 추가: {fname}")
            added += 1

        if chunks:
            vectorstore.add_documents(chunks)

    total = vectorstore._collection.count()
    print(
        f"[VectorDB] 완료 — "
        f"신규: {added}개 파일, 업데이트: {updated}개 파일, 스킵: {skipped}개 파일 | "
        f"총 {total}개 청크"
    )
    return vectorstore
