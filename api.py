# api.py
import os
import time
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query as QParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Dict, Optional
from uuid import uuid5, NAMESPACE_URL
from fastapi import HTTPException, Query  # ✅ dùng Query, không phải QParam
from fastapi.responses import FileResponse
from app.ingest import ingest_pdf  # noqa: E402
from app import ingest as ingest_mod  # dùng PG*, collection từ ingest.py  # noqa: E402
from app.query import QueryEngine  # noqa: E402

from sqlalchemy import create_engine, text as sql_text  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402


try:
    from dotenv import load_dotenv

    ENV_CANDIDATES = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for p in ENV_CANDIDATES:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
except Exception:
    pass
os.environ.setdefault("PGHOST", "localhost")
os.environ.setdefault("PGPORT", os.getenv("PGPORT", "55432"))

os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")

# Tên collection (dùng cả các biến thường gặp trong code)
os.environ.setdefault("QDRANT_COLLECTION", "books_rag")
os.environ.setdefault("BOOKS_COLLECTION", os.getenv("QDRANT_COLLECTION", "books_rag"))
os.environ.setdefault("BOOKS_SEED_COLLECTION", "book_rag_seed")

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---------------------------------------------------------
app = FastAPI(title="Books RAG API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


@app.middleware("http")
async def timing_logger(request, call_next):
    start = time.perf_counter()
    resp = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, resp.status_code, dur_ms)
    return resp


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------
# 5) Engines / Clients (singleton)
# ---------------------------------------------------------
RAG = QueryEngine()
# Band-aid nếu query.py cũ không set thuộc tính này:
if not hasattr(RAG, "has_summary"):
    RAG.has_summary = False

ENGINE = create_engine(
    f"postgresql+psycopg2://{ingest_mod.PGUSER}:{ingest_mod.PGPASSWORD}"
    f"@{os.getenv('PGHOST', 'localhost')}:{ingest_mod.PGPORT}/{ingest_mod.PGDATABASE}"
)

QDRANT = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

COLLECTION = os.getenv("QDRANT_COLLECTION", getattr(ingest_mod, "BOOKS_COLLECTION", "books_rag"))


# ---------------------------------------------------------
# 6) Utils
# ---------------------------------------------------------

# === FS library utils ===
DATA_DIR: Path = getattr(
    ingest_mod, "DATA_DIR",
    Path(__file__).resolve().parents[1] / "data" / "books"
)
# Đọc toàn bộ file theo từng chunk (mặc định 1MB)
def file_iter(path: Path, chunk_size: int = 1_048_576):
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Đọc theo dải byte (phục vụ Range)
def range_stream(path: Path, start: int, end: int, chunk_size: int = 1_048_576):
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_len = min(chunk_size, remaining)
            data = f.read(read_len)
            if not data:
                break
            yield data
            remaining -= len(data)


def _stable_book_id_for_path(p: Path) -> str:
    size = p.stat().st_size if p.exists() else 0
    return str(uuid5(NAMESPACE_URL, f"book::{p.name}::{size}"))

def _guess_meta_from_filename(p: Path) -> Dict:
    try:
        bm = ingest_mod.guess_meta_from_filename(p)
        return {"title": bm.title, "author": bm.author, "year": bm.year}
    except Exception:
        stem = p.stem
        return {"title": stem, "author": None, "year": None}

def _scan_fs_books(q: Optional[str] = None) -> List[Dict]:
    items: List[Dict] = []
    if not DATA_DIR.exists():
        return items
    ql = (q or "").strip().lower()
    for p in sorted(DATA_DIR.glob("*.pdf")):
        if ql and ql not in p.stem.lower():
            continue
        meta = _guess_meta_from_filename(p)
        items.append({
            "file": p.name,
            "path": str(p),
            "title": meta["title"],
            "author": meta["author"],
            "year": meta["year"],
            "size_bytes": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
            "book_id": _stable_book_id_for_path(p),
        })
    return items

def _safe_pdf_path(filename: str) -> Path:
    # chặn path traversal
    fn = filename.strip().replace("\\", "/").split("/")[-1]
    p = (DATA_DIR / fn).resolve()
    if (not p.exists()) or (p.suffix.lower() != ".pdf") or (DATA_DIR.resolve() not in p.parents):
        raise HTTPException(status_code=404, detail="PDF not found")
    return p


def _ensure_event_loop():
    """Đảm bảo worker thread có event loop (fix lỗi AnyIO worker thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


# Pydantic v1/v2 compatibility cho extra fields
try:
    from pydantic import ConfigDict

    _PYD_V2 = True
except Exception:
    _PYD_V2 = False


class QueryIn(BaseModel):
    question: str = Field(..., description="Câu hỏi cho RAG")
    book_id: str | None = Field(None, description="ID sách (để ghim vào 1 cuốn cụ thể)")
    k: int | None = Field(20, description="Top-K retrieve (nếu query.py hỗ trợ)")
    target_chars: int | None = Field(1200, description="Độ dài mục tiêu (nếu hỗ trợ)")
    dry_run: bool | None = Field(False, description="Chỉ retrieve, không generate (nếu hỗ trợ)")

    if _PYD_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


# ---------------------------------------------------------
# 7) Routes
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest/pdf")
def api_ingest_pdf(file: UploadFile = File(...)):
    """
    Upload -> LƯU vào thư mục data/books -> ingest -> trả về book_id
    """
    _ensure_event_loop()

    name = (file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf is accepted")

    # Đảm bảo thư mục tồn tại
    ingest_mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = ingest_mod.DATA_DIR / name

    try:
        # 1) LƯU file vào thư mục thư viện (để Streamlit/FS cùng thấy)
        dest.write_bytes(file.file.read())

        # 2) Tạo book_id theo đúng thuật toán của ingest.py (uuid5(name, size))
        book_id = _stable_book_id_for_path(dest)

        # 3) Ingest từ file trong DATA_DIR (DB + Qdrant)
        ingest_pdf(dest)

        # 4) Trả về cả book_id để FE đồng bộ
        return {
            "filename": name,
            "path": str(dest),
            "ingested": True,
            "book_id": book_id
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            file.file.close()
        except Exception:
            pass


PDF_BASE_URL = os.getenv("PDF_BASE_URL", "http://172.16.1.112:4000/pdf")

@app.get("/books")
def list_books():
    sql = """ 
    WITH stats AS (
        SELECT book_id,
               MIN(page_no)                    AS min_page,
               MAX(page_no)                    AS max_page,
               COUNT(DISTINCT page_id)         AS total_pages
        FROM pages
        GROUP BY book_id
    )
    SELECT 
        b.id::text AS book_id,
        b.title, b.author, b.year,
        b.isbn, b.publisher, b.contentType, b.genres,
        b.digitalPrice, b.digitalQuantity, 
        b.physicalPrice, b.physicalQuantity,
        b.allowPhoneAccess, b.allowPhysicalAccess,
        b.description,
        b.source_pdf,   -- 👈 thêm source_pdf để build url
        COALESCE(s.min_page, 1)                         AS min_page,
        COALESCE(s.max_page, 0)                         AS max_page,
        COALESCE(b.pages, s.total_pages, 0)             AS total_pages
    FROM books b
    LEFT JOIN stats s ON s.book_id = b.id
    ORDER BY b.title ASC;
    """
    with ENGINE.begin() as conn:
        rows = conn.execute(sql_text(sql)).mappings().all()

    out = []
    for r in rows:
        d = dict(r)
        d["pdf_url"] = f"{PDF_BASE_URL}/{r['source_pdf']}" if r.get("source_pdf") else None
        out.append(d)
    return out
# ---- thay thế định nghĩa BookEdit ----
from typing import Optional, List

class BookEdit(BaseModel):
    book_id: str = Field(..., description="UUID của sách đã ingest")

    # core
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = Field(None, description="Năm xuất bản (ví dụ 2019)")

    # extra fields trong bảng books
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    pages: Optional[int] = None
    contentType: Optional[str] = None
    description: Optional[str] = None
    digitalPrice: Optional[str] = None
    digitalQuantity: Optional[str] = None
    physicalPrice: Optional[str] = None
    physicalQuantity: Optional[str] = None
    genres: Optional[List[str]] = None
    allowPhoneAccess: Optional[bool] = None
    allowPhysicalAccess: Optional[bool] = None

# ---- thay thế toàn bộ endpoint POST /books ----
import json

@app.post("/booksUpload")
def edit_book(body: BookEdit):
    # whitelist các cột được phép sửa
    allowed = {
        "title", "author", "year",
        "isbn", "publisher", "pages", "contentType", "description",
        "digitalPrice", "digitalQuantity", "physicalPrice", "physicalQuantity",
        "allowPhoneAccess", "allowPhysicalAccess", "genres"
    }

    # gom thay đổi từ payload (bỏ None)
    raw = body.model_dump(exclude_unset=True) if hasattr(body, "model_dump") else body.dict(exclude_unset=True)
    changes = {k: v for k, v in raw.items() if k in allowed and v is not None}

    # validate nhẹ
    if "year" in changes:

        y = int(changes["year"])
        if y < 0 or y > 2100:
            raise HTTPException(status_code=422, detail="year không hợp lệ")
        changes["year"] = y
    if "pages" in changes:
        p = int(changes["pages"])
        if p < 0:
            raise HTTPException(status_code=422, detail="pages không hợp lệ")
        changes["pages"] = p

    # genres là JSONB -> nhận list và cast sang ::jsonb
    genres_is_jsonb = False
    if "genres" in changes:
        g = changes["genres"]
        if not isinstance(g, list):
            raise HTTPException(status_code=422, detail="genres phải là mảng chuỗi")
        changes["genres"] = json.dumps(g, ensure_ascii=False)  # param text
        genres_is_jsonb = True

    if not changes:
        raise HTTPException(status_code=422, detail="Không có trường nào để cập nhật")

    # build SET clause; riêng genres cần CAST sang JSONB
    sets = []
    for k in changes.keys():
        if k == "genres" and genres_is_jsonb:
            sets.append(f"{k} = CAST(:{k} AS JSONB)")
        else:
            sets.append(f"{k} = :{k}")
    set_clause = ", ".join(sets)

    with ENGINE.begin() as conn:
        res = conn.execute(
            sql_text(f"UPDATE books SET {set_clause} WHERE id::text = :id"),
            {**changes, "id": body.book_id}
        )
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Không tìm thấy book_id")

    row = conn.execute(sql_text("""
          WITH stats AS (
              SELECT book_id,
                     MIN(page_no)            AS min_page,
                     MAX(page_no)            AS max_page,
                     COUNT(DISTINCT page_id) AS total_pages
              FROM pages
              WHERE book_id::text = :id
              GROUP BY book_id
          )
          SELECT 
              b.id::text AS book_id,
              b.title, b.author, b.year,
              b.isbn, b.publisher, b.contentType, b.genres,
              b.digitalPrice, b.digitalQuantity, 
              b.physicalPrice, b.physicalQuantity,
              b.allowPhoneAccess, b.allowPhysicalAccess,
              b.description,
              b.source_pdf,   -- 👈 thêm vào
              COALESCE(s.min_page, 1)             AS min_page,
              COALESCE(s.max_page, 0)             AS max_page,
              COALESCE(b.pages, s.total_pages, 0) AS total_pages
          FROM books b
          LEFT JOIN stats s ON s.book_id = b.id
          WHERE b.id::text = :id
      """), {"id": body.book_id}).mappings().first()

    result = dict(row)
    result["pdf_url"] = f"{PDF_BASE_URL}/{row['source_pdf']}" if row.get("source_pdf") else None
    return result


import inspect
from fastapi import HTTPException, Request

@app.post("/query")
def api_query(body: QueryIn, request: Request):
    try:
        # Pydantic v2: model_dump ; v1: dict
        incoming = body.model_dump(exclude_none=True) if hasattr(body, "model_dump") \
                   else body.dict(exclude_none=True)

        question = (incoming.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="Missing 'question'")

        # Lấy RAG engine: ưu tiên app.state nếu có, fallback biến toàn cục RAG
        rag = getattr(request.app.state, "RAG_ENGINE", None) if "request" in api_query.__code__.co_varnames else None
        rag = rag or RAG

        # Chuẩn hoá book_id -> str (nếu có)
        if "book_id" in incoming and incoming["book_id"] is not None:
            incoming["book_id"] = str(incoming["book_id"])

        # Lọc theo chữ ký thật của run_query (để truyền được 'debug')
        sig = inspect.signature(rag.run_query)
        safe_kwargs = {k: v for k, v in incoming.items() if k in sig.parameters}

        # Gọi run_query; nếu body có "debug": true -> response sẽ có "trace"
        result = rag.run_query(**safe_kwargs)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 8) Debug endpoints (giúp soi nhanh cấu hình & dữ liệu)
import os
from fastapi import Request, HTTPException

@app.get("/debug/state")
def debug_state(request: Request):
    # Lấy RAG engine: ưu tiên app.state, fallback biến toàn cục RAG
    rag = getattr(request.app.state, "RAG_ENGINE", None)
    rag = rag or globals().get("RAG")
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    qc = rag.qc

    main_coll = os.getenv("QDRANT_COLLECTION", "books_rag")
    seed_coll = os.getenv("QDRANT_SEED_COLLECTION", "book_rag_seed")

    # danh sách collection hiện có trong Qdrant
    try:
        coll_list = [c.name for c in qc.get_collections().collections]
    except Exception:
        coll_list = []

    def _count(coll_name: str):
        try:
            return int(qc.count(collection_name=coll_name, count_filter=None).count or 0)
        except Exception:
            return None  # collection có thể không tồn tại

    return {
        "ok": True,
        "db": {
            # nếu bạn có các hàm đếm trong engine thì trả, không thì None
            "books": getattr(rag, "count_books", lambda: None)(),
            "pages": getattr(rag, "count_pages", lambda: None)(),
        },
        "qdrant": {
            "collections": coll_list,  # để bạn nhìn được toàn bộ tên collections
            "main": {
                "name": main_coll,
                "exists": main_coll in coll_list,
                "points": _count(main_coll),
            },
            "seed": {
                "name": seed_coll,
                "exists": seed_coll in coll_list,
                "points": _count(seed_coll),
            },
            "has_seed": getattr(rag, "has_seed", None),  # flag bên trong QueryEngine
        },
    }

# === FS library endpoints ===
@app.get("/fs/books")
def list_fs_books(
    q: Optional[str] = Query(None, description="lọc theo cụm từ trong tên/tựa"),
    with_db: bool = Query(False, description="đếm trang đã ingest trong DB"),
):
    items = _scan_fs_books(q)
    if with_db and items:
        try:
            with ENGINE.begin() as conn:
                for it in items:
                    cnt = conn.execute(
                        sql_text("SELECT COUNT(*) FROM pages WHERE book_id = :b"),
                        {"b": it["book_id"]}
                    ).scalar()
                    it["db_pages"] = int(cnt or 0)
        except Exception:
            # DB không bắt buộc cho FS mode
            pass
    return {"root": str(DATA_DIR), "items": items, "count": len(items)}


@app.get("/fs/books/{filename}/pdf")
def get_fs_pdf(filename: str):
    """
    Trả về PDF file trực tiếp (dùng FileResponse).
    Browser sẽ hiển thị inline hoặc cho tải về.
    """
    p = _safe_pdf_path(filename)

    if not p.exists() or p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=str(p),
        media_type="application/pdf",
        filename=p.name,
        headers={
            "Content-Disposition": f'inline; filename="{p.name}"',
            "Accept-Ranges": "bytes",  # để browser có thể tua trang
        },
    )
# api.py
from qdrant_client.http import models as qmodels

def _or_book_id(book_id: str):
    return [
        qmodels.FieldCondition(key="book_id",            match=qmodels.MatchValue(value=str(book_id))),
        qmodels.FieldCondition(key="metadata.book_id",   match=qmodels.MatchValue(value=str(book_id))),
    ]

@app.get("/debug/qdrant/check_book")
def debug_check_book(book_id: str, request: Request):
    rag = getattr(request.app.state, "RAG_ENGINE", None) or globals().get("RAG")
    qc = rag.qc
    main_coll = os.getenv("QDRANT_COLLECTION","books_rag")
    seed_coll = os.getenv("QDRANT_SEED_COLLECTION","book_rag_seed")

    flt = qmodels.Filter(should=_or_book_id(book_id))  # OR giữa 2 khóa

    chunks = int(qc.count(collection_name=main_coll, count_filter=flt).count or 0)
    seeds  = int(qc.count(collection_name=seed_coll, count_filter=flt).count or 0)

    # lấy vài section_id (cũng OR giữa 2 key-path)
    pts, _ = qc.scroll(collection_name=seed_coll, scroll_filter=flt, with_payload=True, limit=5)
    secs = []
    for p in pts:
        pl = p.payload or {}
        sid = pl.get("section_id") or (pl.get("metadata", {}) or {}).get("section_id")
        if sid: secs.append(sid)

    return {"book_id": str(book_id), "counts": {"chunks": chunks, "seeds": seeds}, "sample_section_ids": secs}


@app.get("/debug/qdrant/check_section")
def debug_check_section(book_id: str, section_id: str):
    qc = (getattr(app.state, "RAG_ENGINE", None) or RAG).qc
    coll = os.getenv("QDRANT_COLLECTION", "books_rag")

    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="book_id",   match=qmodels.MatchValue(value=str(book_id))),
        qmodels.FieldCondition(key="section_id",match=qmodels.MatchValue(value=str(section_id))),
    ])
    cnt = int(qc.count(collection_name=coll, count_filter=flt).count or 0)
    return {"book_id": book_id, "section_id": section_id, "chunks_in_section": cnt}
# api.py
from fastapi import Request, HTTPException
import os

@app.get("/debug/qdrant/list_books")
def list_books(request: Request, limit: int = 100):
    rag = getattr(request.app.state, "RAG_ENGINE", None) or globals().get("RAG")
    if not rag:
        raise HTTPException(500, "RAG engine not initialized")
    qc = rag.qc

    main_coll = os.getenv("QDRANT_COLLECTION", "books_rag")
    seed_coll = os.getenv("QDRANT_SEED_COLLECTION", "book_rag_seed")

    found = set()

    def _collect(coll_name: str, need: int):
        offset = None
        while len(found) < need:
            pts, offset = qc.scroll(
                collection_name=coll_name,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            if not pts:
                break
            for p in pts:
                pl = p.payload or {}
                # book_id có thể ở root hoặc lồng trong metadata (hiếm)
                bid = pl.get("book_id") or (pl.get("metadata", {}) or {}).get("book_id")
                if bid:
                    found.add(str(bid))
                if len(found) >= need:
                    break
            if offset is None:  # hết trang
                break

    # quét cả 2 collection
    _collect(main_coll, limit)
    if len(found) < limit:
        _collect(seed_coll, limit)

    return {
        "books_in_qdrant_sample": sorted(found),
        "scanned_collections": [main_coll, seed_coll],
        "total_found": len(found)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",  # Tên file (api.py) : app instance
        host="0.0.0.0",  # Cho phép truy cập từ LAN
        port=8000,  # Có thể đổi port nếu muốn
        reload=True  # Tự reload khi code thay đổi (chỉ nên dùng dev)
    )