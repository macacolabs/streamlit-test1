import importlib.util
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────
# 형제 모듈 동적 임포트 (숫자 시작 파일명 대응)
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


# ──────────────────────────────────────────────
# RAG 1단계: 문서 로드 & 청크 분할
# ──────────────────────────────────────────────

def load_and_split_documents() -> list:
    all_docs = []

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    print("[RAG] AXCompass PDF 로드 중...")
    ax_pages = PyPDFLoader(PDF_PATH).load()
    for page in ax_pages:
        page.metadata["doc_type"] = "ax_compass"
    all_docs.extend(ax_pages)

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".pdf") and fname != "AXCompass.pdf":
            fpath = os.path.join(DATA_DIR, fname)
            print(f"[RAG] 커리큘럼 PDF 로드: {fname}")
            pages = PyPDFLoader(fpath).load()
            course_name = os.path.splitext(fname)[0]
            for page in pages:
                page.metadata.update({"doc_type": "curriculum_example", "course_name": course_name})
            all_docs.extend(pages)

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".xlsx"):
            fpath = os.path.join(DATA_DIR, fname)
            print(f"[RAG] 커리큘럼 Excel 로드: {fname}")
            docs = UnstructuredExcelLoader(fpath).load()
            course_name = os.path.splitext(fname)[0]
            for doc in docs:
                doc.metadata.update({"doc_type": "curriculum_example", "course_name": course_name})
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)

    for chunk in chunks:
        if chunk.metadata.get("doc_type") == "ax_compass":
            for type_name, info in TYPE_INFO.items():
                if type_name in chunk.page_content:
                    chunk.metadata.update({
                        "type_name": type_name,
                        "group":     info["group"],
                        "english":   info["english"],
                    })
                    break

    ax_count = sum(1 for c in chunks if c.metadata.get("doc_type") == "ax_compass")
    print(f"[RAG] 총 {len(chunks)}개 청크 (AX Compass: {ax_count}, 커리큘럼 예시: {len(chunks) - ax_count})")
    return chunks


# ──────────────────────────────────────────────
# RAG 2단계: 벡터 DB 구축
# ──────────────────────────────────────────────

def setup_vector_store(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    if vectorstore._collection.count() == 0:
        print("[VectorDB] 임베딩 생성 중...")
        chunks = load_and_split_documents()
        vectorstore.add_documents(chunks)
        print(f"[VectorDB] {len(chunks)}개 청크 저장 완료 → {VECTOR_DB_PATH}")
    else:
        print(f"[VectorDB] 기존 컬렉션 로드 완료 ({vectorstore._collection.count()}개 청크)")

    return vectorstore
