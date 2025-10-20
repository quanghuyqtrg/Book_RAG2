from __future__ import annotations

import asyncio
import math
import mimetypes
import os
import re
import sys
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv
from google import genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import create_engine, text as sql_text
from tqdm import tqdm

# ------------------------- Config -------------------------
load_dotenv()

PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "husky")
PGDATABASE = os.getenv("PGDATABASE", "postgres")
PGPORT = int(os.getenv("PGPORT", "55432"))
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION = os.getenv("QDRANT_COLLECTION", "books_rag")
SUMMARY_COLLECTION = os.getenv("QDRANT_SUMMARY_COLLECTION", "books_rag_summary")

SEED_COLLECTION = os.getenv("QDRANT_SEED_COLLECTION", "book_rag_seed")
RECREATE_SEED = str(os.getenv("RECREATE_SEED", "0")).lower() in {"1", "true", "yes", "y"}

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
EMBED_DIM = 768  # text-embedding-004
GENAI_SUMMARY_MODEL = os.getenv("GENAI_SUMMARY_MODEL", "gemini-2.5-flash")
GENAI_HYDE_MODEL = os.getenv("GENAI_HYDE_MODEL", GENAI_SUMMARY_MODEL)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "320"))
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "books"

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# ---- thêm gần các biến ENV khác ----
USE_CONTENT_META = str(os.getenv("USE_CONTENT_META", "1")).lower() in {"1", "true", "yes", "y"}
FORCE_REINDEX = str(os.getenv("FORCE_REINDEX", "0")).lower() in {"1", "true", "yes", "y"}

engine = create_engine(
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@localhost:{PGPORT}/{PGDATABASE}"
)

# -------------------- Heuristics / Regex --------------------
HEADING_RE = re.compile(
    r"^(?:\s*(?:Chapter|Chương|Section|Mục)\s+\d+\b|\s*\d+(?:\.\d+){0,3}\s+.+)$",
    flags=re.IGNORECASE,
)
BULLET_RE = re.compile(r"^\s*(?:[-•*]\s+|\d+\.|[a-z]\))\s+.+")
TABLE_FIG_RE = re.compile(r"\b(?:Table|Figure|Fig\.)\s*\d+\b", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^\s*```")

MIN_SECTION_LEN = 400  # merge tiny sections into neighbors

VI_CHARS = "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộơớờởỡợúùủũụứừửữựýỳỷỹỵ"


def _est_tokens(text: str) -> int:
    if not text:
        return 0
    # xấp xỉ ~4 ký tự / token
    return max(1, math.ceil(len(text) / 4))


def _est_tokens_batch(texts) -> int:
    return sum(_est_tokens(t) for t in (texts or []))


def guess_lang(s: str) -> str:
    return "vi" if any(ch in s for ch in VI_CHARS) else "en"


# === Debug helpers: in ra metadata + số điểm trong Qdrant ===
def _or_book_id(book_id: str) -> list[qmodels.FieldCondition]:
    bid = str(book_id)
    return [
        qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=bid)),
        qmodels.FieldCondition(key="metadata.book_id", match=qmodels.MatchValue(value=bid)),
    ]


def _count_qdrant_points(qc: QdrantClient, collection: str, book_id: str) -> int:
    try:
        flt = qmodels.Filter(should=_or_book_id(book_id))
        return int(qc.count(collection_name=collection, count_filter=flt).count or 0)
    except Exception:
        return 0


def _print_ingest_summary(book_meta, book_id: str, qc: QdrantClient,
                          main_coll: str, seed_coll: str | None = None):
    """In ra terminal: metadata + số chunks/seed trong Qdrant."""
    title = getattr(book_meta, "title", None)
    author = getattr(book_meta, "author", None)
    year = getattr(book_meta, "year", None)
    source = getattr(book_meta, "source_pdf", None)
    pages = getattr(book_meta, "pages", None)  # nếu bạn có set

    chunks = _count_qdrant_points(qc, main_coll, book_id)
    seeds = _count_qdrant_points(qc, seed_coll, book_id) if seed_coll else 0

    print("\n========== INGEST SUMMARY ==========")
    print(f"Book ID     : {book_id}")
    print(f"Title       : {title}")
    print(f"Author      : {author}")
    print(f"Year        : {year}")
    print(f"Source PDF  : {source}")
    if pages is not None:
        print(f"Pages       : {pages}")
    print("------------------------------------")
    print(f"Qdrant [{main_coll}] chunks : {chunks}")
    if seed_coll:
        print(f"Qdrant [{seed_coll}] seed   : {seeds}")
    print("====================================\n")


def numeric_heading_path(path: List[str]) -> List[int]:
    out: List[int] = []
    for h in path or []:
        m = re.match(r"\s*(\d+(?:\.\d+)*)\b", str(h))
        if m: out += [int(x) for x in m.group(1).split(".")]
    return out


def _ensure_event_loop() -> None:
    """Đảm bảo thread hiện tại có asyncio event loop (cần cho grpfc.aio)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Trên Windows, gRPC thường ổn hơn với Selector policy
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        asyncio.set_event_loop(asyncio.new_event_loop())


@lru_cache(maxsize=1)
def get_embeddings():
    """Factory có cache để tránh khởi tạo client nhiều lần."""
    _ensure_event_loop()
    return GeminiEmb(model=EMBED_MODEL, google_api_key=API_KEY)


def make_flags(text: str) -> Dict:
    return {
        "has_table": bool(TABLE_FIG_RE.search(text)),
        "has_code": bool(CODE_FENCE_RE.search(text)),
        "lang": guess_lang(text),
        "num_chars": len(text),
    }


def build_seed(text: str, limit: int = 400) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    core = " ".join((sents[:1] + sents[-1:]) or sents[:1])
    return core[:limit]


# ------------------------- Models -------------------------
@dataclass
class BookMeta:
    title: str
    author: Optional[str]
    year: Optional[int]
    source_pdf: str


# ===== Content-based metadata extraction (no filename focus) =====
import re
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import fitz  # PyMuPDF (optional, better title detection by font size)

    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

try:
    from pypdf import PdfReader  # PyPDF2 mới
except Exception:
    PdfReader = None


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _stable_book_id(pdf_path: Path) -> uuid.UUID:
    size = pdf_path.stat().st_size if pdf_path.exists() else 0
    return uuid.uuid5(uuid.NAMESPACE_URL, f"book::{pdf_path.name}::{size}")


def heading_level(h: str) -> int:
    m = re.match(r"\s*(\d+(?:\.\d+)*)\b", h)
    if not m:
        return 2
    parts = m.group(1).split(".")
    return min(1 + len(parts) - 1, 4)


def update_heading_path(path: List[str], new_h: str) -> List[str]:
    lvl = heading_level(new_h)
    return path[: max(0, lvl - 1)] + [new_h]


def apply_no_break_zones(text: str) -> str:
    """Keep bullets, code fences, and caption blocks intact during splitting."""
    lines = text.splitlines()
    out: List[str] = []
    buf: List[str] = []
    in_code = False

    def flush():
        nonlocal buf
        if buf:
            out.append("\n".join(buf))
            buf = []

    for ln in lines:
        if CODE_FENCE_RE.match(ln):
            in_code = not in_code
            buf.append(ln)
            if not in_code:
                flush()
            continue
        if in_code:
            buf.append(ln)
            continue
        if BULLET_RE.match(ln) or TABLE_FIG_RE.search(ln):
            buf.append(ln)
            continue
        if HEADING_RE.match(ln):
            flush()
            out.append(ln)
            continue
        buf.append(ln)
    flush()
    return "\n".join(out)


def split_headings(page_text: str) -> List[Tuple[List[str], str]]:
    """Return list of (heading_path, section_text) for a single page."""
    lines = page_text.splitlines()
    sections: List[Tuple[List[str], List[str]]] = []
    cur_path: List[str] = []
    cur_lines: List[str] = []

    for ln in lines:
        if HEADING_RE.match(ln):
            # flush current section
            if cur_lines:
                body = "\n".join(cur_lines).strip()
                if body:
                    body = apply_no_break_zones(body)
                    sections.append((cur_path.copy(), [body]))
                cur_lines = []
            cur_path = update_heading_path(cur_path, ln.strip())
        else:
            cur_lines.append(ln)

    if cur_lines:
        body = "\n".join(cur_lines).strip()
        if body:
            body = apply_no_break_zones(body)
            sections.append((cur_path.copy(), [body]))

    # merge short sections into previous one
    merged: List[Tuple[List[str], str]] = []
    for hp, bodies in sections:
        text_block = "\n".join(bodies).strip()
        if len(text_block) < MIN_SECTION_LEN and merged:
            prev_hp, prev_txt = merged[-1]
            merged[-1] = (prev_hp, (prev_txt + "\n" + text_block).strip())
        else:
            merged.append((hp, text_block))
    return merged


# -------------------------- DB helpers --------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS books (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT,
  year INT,
  source_pdf TEXT UNIQUE,
  isbn                 TEXT,
  publisher            TEXT,
  pages                INT,                  -- tổng số trang (nếu ingest trích được)
  allowPhoneAccess   BOOLEAN,
  allowPhysicalAccess BOOLEAN,
  contentType         TEXT,                 -- vd: "application/pdf"
  description          TEXT,
  digitalPrice        TEXT,
  digitalQuantity     TEXT,
  physicalPrice       TEXT,
  physicalQuantity    TEXT,
  genres               JSONB NOT NULL DEFAULT '[]'::jsonb  
);

CREATE TABLE IF NOT EXISTS pages (
  page_id UUID PRIMARY KEY,
  book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  page_no INT NOT NULL,
  text TEXT NOT NULL,
  UNIQUE (book_id, page_no)
);
"""


def ensure_schema():
    with engine.begin() as conn:
        conn.execute(sql_text(SCHEMA_SQL))


def upsert_book(meta: BookMeta, book_id: uuid.UUID, **extras) -> uuid.UUID:
    """
    Lưu (INSERT/UPDATE) 1 cuốn sách vào bảng `books`.

    - 5 cột lõi lấy từ `meta`: title, author, year, source_pdf (id là khóa chính).
    - Các cột mở rộng (isbn, publisher, pages, ... genres) truyền thêm bằng **extras.
      Chỉ những cột bạn TRUYỀN VÀO (và khác None) mới được ghi vào DB.

    Ví dụ gọi:
        upsert_book(meta, book_id)  # chỉ cột lõi
        upsert_book(meta, book_id, pages=339, isbn="978-...", content_type="application/pdf")
        upsert_book(meta, book_id, genres=["ML", "Khoa học"])
    """
    import json as _json

    # 1) Cột mở rộng được phép ghi (phải tồn tại trong schema mới)
    allowed_extras = {
        "isbn",
        "publisher",
        "pages",
        "allowPhoneAccess",
        "allowPhysicalAccess",
        "contentType",
        "description",
        "digitalPrice",
        "digitalQuantity",
        "physicalPrice",
        "physicalQuantity",
        "genres",  # JSONB
    }

    # 2) Payload lõi (đặt tên khóa dễ hiểu -> map vào SQL)
    payload = {
        "id": str(book_id),
        "title": meta.title,
        "author": meta.author,
        "year": meta.year,
        "source_pdf": meta.source_pdf,
    }

    # 3) Lọc extras: chỉ giữ key hợp lệ & khác None
    extras_filtered = {}
    for k in allowed_extras:
        if k in extras and extras[k] is not None:
            if k == "genres":
                # genres có thể là list/dict -> chuyển sang chuỗi JSON để cast ::jsonb trong SQL
                v = extras[k]
                extras_filtered[k] = v if isinstance(v, str) else _json.dumps(v)
            else:
                extras_filtered[k] = extras[k]

    # 4) Ghép danh sách cột/giá trị cho phần INSERT
    insert_cols = ["id", "title", "author", "year", "source_pdf"] + list(extras_filtered.keys())
    # với genres, ta cast ::jsonb trong SQL cho rõ ràng
    insert_vals = [
        ":id", ":title", ":author", ":year", ":source_pdf",
        *[(":genres::jsonb" if c == "genres" else f":{c}") for c in extras_filtered.keys()]
    ]

    # 5) Ghép phần UPDATE (ON CONFLICT) — chỉ update cột mình có dữ liệu
    update_sets = [
                      "title = EXCLUDED.title",
                      "author = EXCLUDED.author",
                      "year = EXCLUDED.year",
                      "source_pdf = EXCLUDED.source_pdf",
                  ] + [f"{c} = EXCLUDED.{c}" for c in extras_filtered.keys()]

    # 6) Gộp payload: cột lõi + extras đã lọc
    payload.update(extras_filtered)

    # 7) Tạo SQL rõ ràng, dễ đọc
    sql = f"""
        INSERT INTO books ({", ".join(insert_cols)})
        VALUES ({", ".join(insert_vals)})
        ON CONFLICT (id) DO UPDATE SET
          {", ".join(update_sets)}
    """

    # 8) Thực thi
    with engine.begin() as conn:
        conn.execute(sql_text(sql), payload)

    return book_id


def upsert_page(book_id: uuid.UUID, page_no: int, text_: str) -> uuid.UUID:
    pid = uuid.uuid5(book_id, f"page::{page_no}")
    with engine.begin() as conn:
        conn.execute(sql_text(
            """
            INSERT INTO pages (page_id, book_id, page_no, text)
            VALUES (:id, :b, :p, :t)
            ON CONFLICT (book_id, page_no) DO UPDATE SET text = EXCLUDED.text
            """
        ), {"id": str(pid), "b": str(book_id), "p": page_no, "t": text_})
    return pid


# ------------------------- Ingest core -------------------------

def guess_meta_from_filename(pdf_path: Path) -> BookMeta:
    name = pdf_path.stem
    m = re.search(r"(19|20)\d{2}", name)
    year = int(m.group(0)) if m else None
    parts = [p.strip() for p in re.split(r"[-_]", name) if p.strip()]
    title = parts[0] if parts else name
    author = parts[1] if len(parts) > 1 and not re.match(r"^\d{4}$", parts[1]) else None
    return BookMeta(title=title, author=author, year=year, source_pdf=pdf_path.name)


class GeminiEmb(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts=texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text=text, task_type="RETRIEVAL_QUERY")


ENABLE_VISION_COVER = bool(int(os.getenv("ENABLE_VISION_COVER", "1")))
VISION_MODEL = os.getenv("GENAI_VISION_MODEL", os.getenv("GENAI_SUMMARY_MODEL", "gemini-2.5-flash"))
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "10"))


def _gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _read_image_file(path: Path) -> Tuple[bytes, str]:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        data = f.read()
    return data, mime


def _gemini_image_infer(prompt: str, image_bytes: bytes, mime: str, model_name: str) -> str:
    client = _gemini_client()
    part = genai.types.Part.from_bytes(data=image_bytes, mime_type=mime)
    resp = client.models.generate_content(model=model_name, contents=[prompt, part])
    return (getattr(resp, "text", None) or "").strip()


def _vision_extract_meta_from_image(img_path: Path) -> dict:
    """
    Đọc 1 ảnh (bìa/scans) bằng Gemini Image (google-genai).
    Trả về: {title, author, year} (có thể None).
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[VISION-IMG] missing GOOGLE_API_KEY/GEMINI_API_KEY")
            return {}

        model_name = os.getenv("GENAI_VISION_MODEL", "gemini-2.5-flash")
        prompt = (
            "You are reading ONLY the provided image(s) of a book cover or front-matter. "
            "Extract bibliographic metadata if visible: title, author(s), publication year. "
            "Return STRICT JSON with keys: title, author, year. Values may be null; "
            "year must be 4 digits if present."
        )

        data, mime = _read_image_file(img_path)
        raw = _gemini_image_infer(prompt, data, mime, model_name)

        import json, re
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}

        title = data.get("title") or None
        author = data.get("author") or None
        year = data.get("year") or None
        try:
            year = int(year) if year is not None else None
            if year and not (1000 <= year <= 2100):
                year = None
        except Exception:
            year = None

        print(f"[VISION-IMG] parsed -> title={title!r} author={author!r} year={year!r}")
        return {"title": title, "author": author, "year": year}
    except Exception as e:
        print(f"[VISION-IMG] error: {e}")
        return {}


def guess_meta_from_pdf_content(pdf_path: Path) -> BookMeta:
    data = _vision_extract_meta_from_image(pdf_path) or {}
    title = (data.get("title") or "").strip() or pdf_path.stem
    auth = data.get("author")
    # tác giả có thể là list -> join
    if isinstance(auth, (list, tuple)):
        auth = ", ".join([str(a).strip() for a in auth if str(a).strip()])
    elif isinstance(auth, str):
        auth = auth.strip() or None
    else:
        auth = None

    year = data.get("year")
    try:
        year = int(year) if year not in (None, "") else None
        if year and not (1000 <= year <= 2100):
            year = None
    except Exception:
        year = None

    return BookMeta(title=title, author=auth, year=year, source_pdf=pdf_path.name)


def ensure_collections(qc: QdrantClient):
    # Lấy danh sách collection hiện có
    cols = {c.name for c in qc.get_collections().collections}

    if RECREATE_SEED and SEED_COLLECTION in cols:
        qc.delete_collection(SEED_COLLECTION)
        cols.remove(SEED_COLLECTION)

    # Tạo collection chính (chunks)
    if COLLECTION not in cols:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBED_DIM, distance=qmodels.Distance.COSINE
            ),
        )

    # Tạo collection seed (Small-to-Big)
    if SEED_COLLECTION not in cols:
        qc.create_collection(
            collection_name=SEED_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBED_DIM, distance=qmodels.Distance.COSINE
            ),
        )

    # ----- Payload indexes (lọc nhanh theo metadata) -----
    idx_defs = [
        ("book_id", qmodels.PayloadSchemaType.KEYWORD),
        ("section_id", qmodels.PayloadSchemaType.KEYWORD),
        ("heading_level", qmodels.PayloadSchemaType.INTEGER),
        ("type", qmodels.PayloadSchemaType.KEYWORD),  # text | seed | object_caption
        ("page_number", qmodels.PayloadSchemaType.INTEGER),
        ("has_table", qmodels.PayloadSchemaType.BOOL),
        ("has_code", qmodels.PayloadSchemaType.BOOL),
        ("lang", qmodels.PayloadSchemaType.KEYWORD),  # vi | en (heuristic)
        ("author", qmodels.PayloadSchemaType.KEYWORD),
        ("year", qmodels.PayloadSchemaType.INTEGER),
        ("printed_label", qmodels.PayloadSchemaType.KEYWORD),

        # bạn có thể thêm "year", "author" nếu cần lọc thêm
    ]

    # Tạo index cho cả COLLECTION & SEED_COLLECTION (SUMMARY tuỳ bạn có dùng hay không)
    for field, schema in idx_defs:
        for cname in (COLLECTION, SEED_COLLECTION):
            try:
                qc.create_payload_index(
                    collection_name=cname, field_name=field, field_schema=schema
                )
            except Exception:
                # đã có index hoặc version Qdrant không hỗ trợ schema này -> bỏ qua
                pass


from qdrant_client.http import models as qmodels


def _count_points(qc: QdrantClient, collection: str, book_id: str) -> int:
    try:
        flt = qmodels.Filter(must=[
            qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=book_id))
        ])
        return int(qc.count(collection_name=collection, count_filter=flt).count or 0)
    except Exception:
        return 0


def _already_indexed(qc: QdrantClient, book_id: str) -> dict:
    cols = {c.name for c in qc.get_collections().collections}
    n_chunks = _count_points(qc, COLLECTION, book_id) if COLLECTION in cols else 0
    n_seeds = _count_points(qc, SEED_COLLECTION, book_id) if SEED_COLLECTION in cols else 0
    n_summ = _count_points(qc, SUMMARY_COLLECTION, book_id) if SUMMARY_COLLECTION in cols else 0
    return {"chunks": n_chunks, "seeds": n_seeds, "summaries": n_summ, "total": n_chunks + n_seeds + n_summ}


def ingest_pdf(pdf_path: Path, *, skip_if_exists: bool = True, force: bool = False, pages=None) -> dict:
    global cap
    assert pdf_path.exists(), f"Not found: {pdf_path}"
    ensure_schema()

    book_id = _stable_book_id(pdf_path)
    usage = {
        "embedding_docs_tokens": 0,
        "embedding_seed_tokens": 0,
        "vision_extract_tokens": 0,
        "pages": 0,
    }
    try:
        usage["pages"] = len(pages)
    except Exception:
        pass

    # Cho phép ép reindex qua ENV (dùng ở bước 4)
    force = force or FORCE_REINDEX

    if USE_CONTENT_META:
        try:
            meta = guess_meta_from_pdf_content(pdf_path)  # dùng nội dung PDF
            # Phòng hờ tiêu đề rỗng → fallback tên file
            if not (meta and meta.title):
                raise ValueError("empty title from content extractor")
        except Exception:
            meta = guess_meta_from_filename(pdf_path)  # fallback tên file
    else:
        meta = guess_meta_from_filename(pdf_path)

    upsert_book(meta, book_id)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    # --- footer page labels --

    qc = QdrantClient(url=f"http://localhost:{QDRANT_PORT}", prefer_grpc=False)
    ensure_collections(qc)

    if skip_if_exists and not force:
        stats = _already_indexed(qc, str(book_id))
        if stats["total"] > 0:
            print(f"⚠️  Skip vectorize (already indexed): {pdf_path.name} -> {stats}")
            return {
                "book_id": str(book_id),
                "file": pdf_path.name,
                "skipped": True,
                "points": stats,
                "pages": None,
            }

    embeddings = get_embeddings()
    docs_texts: List[str] = []
    docs_metas: List[Dict] = []
    seed_texts: List[str] = []
    seed_metas: List[Dict] = []
    global_chunk_index = 0

    for i, p in enumerate(tqdm(pages, desc=f"{pdf_path.name}")):
        page_text = (p.page_content or "").strip()
        upsert_page(book_id, i + 1, page_text)
        if not page_text:
            continue

        sections = split_headings(page_text)
        for sec_order, (heading_path, sec_text) in enumerate(sections):
            # compute heading level from last heading token
            heading_level_val = 2
            if heading_path:
                last_h = str(heading_path[-1])
                m = re.match(r"\s*(\d+(?:\.\d+)*)\b", last_h)
                if m:
                    heading_level_val = min(1 + len(m.group(1).split(".")) - 1, 4)

            section_id = uuid.uuid5(book_id, f"{i + 1}::{' > '.join(map(str, heading_path))}")
            # --- Embedded object captions (Table/Figure) ---
            # Tách các dòng caption kiểu "Table 2.1 ..." / "Figure 3.4 ..." / "Fig. 1 ..."
            cap_lines = [ln.strip() for ln in sec_text.splitlines() if TABLE_FIG_RE.search(ln)]

            # (tuỳ chọn) hạn chế số caption/section để tránh noise quá nhiều
            for cap in cap_lines[:8]:
                # Thêm văn bản caption vào docs_texts để embed như một node độc lập
                docs_texts.append(cap)
                usage["embedding_docs_tokens"] += _est_tokens(cap)  # hoặc _est_tokens(chunk)

                # Gắn metadata cho node loại "object_caption"
                payload_obj = {
                    "book_id": str(book_id),
                    "page_number": i + 1,
                    "page_span": (i + 1, i + 1),
                    "section_id": str(uuid.uuid5(book_id, f"{i + 1}::{' > '.join(map(str, heading_path))}")),
                    "section_order": int(sec_order),
                    "heading_level": int(heading_level(heading_path[-1]) if heading_path else 2),
                    "heading_path": heading_path,
                    "global_chunk_index": global_chunk_index,
                    "title": meta.title,
                    "author": meta.author,
                    "year": meta.year,
                    "source_pdf": pdf_path.name,
                    "text": cap, "type": "object_caption",
                    "object_kind": "figure" if cap.lower().startswith(("figure", "fig.")) else "table",

                }
                docs_metas.append(payload_obj)
                global_chunk_index += 1

            # dynamic chunk size by level
            if heading_level_val <= 1:
                cs, ov = int(CHUNK_SIZE * 1.3), int(CHUNK_OVERLAP * 1.25)
            elif heading_level_val == 2:
                cs, ov = CHUNK_SIZE, CHUNK_OVERLAP
            else:
                cs = max(int(CHUNK_SIZE * 0.75), 600)
                ov = max(int(CHUNK_OVERLAP * 0.85), 180)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cs,
                chunk_overlap=ov,
                separators=["\n\n", "\n", ". ", ".", "! ", "? ", "?", ", ", ",", " "]
            )
            chunks = splitter.split_text(sec_text)
            # additional pass to merge tiny trailing shards
            merged: List[str] = []
            buf = ""
            for ch in chunks:
                if len(ch) < max(120, int(0.2 * cs)):
                    buf = (buf + "\n" + ch).strip()
                else:
                    if buf:
                        merged.append(buf)
                        buf = ""
                    merged.append(ch)
            if buf:
                merged.append(buf)
            # HyDE pseudo‑metadata (optional)
            hyde_meta: Optional[Dict] = None
            for j, chunk in enumerate(merged):
                docs_texts.append(chunk)
                usage["embedding_docs_tokens"] += _est_tokens(chunk)  # hoặc _est_tokens(chunk)
                payload = {
                    "book_id": str(book_id),
                    "page_number": i + 1,
                    "page_span": (i + 1, i + 1),
                    "section_id": str(section_id),
                    "section_order": int(sec_order),
                    "heading_level": int(heading_level_val),
                    "heading_path": heading_path,
                    "heading_numeric_path": numeric_heading_path(heading_path),
                    "chunk_index": j,
                    "global_chunk_index": global_chunk_index,
                    "title": meta.title,
                    "author": meta.author,
                    "year": meta.year,
                    "source_pdf": pdf_path.name,
                    "text": chunk,
                    "type": "text",
                }
                payload.update(make_flags(chunk))
                docs_metas.append(payload)
                global_chunk_index += 1

                # --- Small-to-Big seed (1 seed/section, không dùng LLM) ---
            seed_text = build_seed(sec_text)
            if seed_text:
                seed_texts.append(seed_text)
                usage["embedding_seed_tokens"] += _est_tokens(seed_text)
                seed_metas.append({
                    "book_id": str(book_id),
                    "page_number": i + 1,
                    "page_span": (i + 1, i + 1),
                    "section_id": str(uuid.uuid5(book_id, f"{i + 1}::{' > '.join(map(str, heading_path))}")),
                    "section_order": int(sec_order),
                    "heading_level": int(heading_level(heading_path[-1]) if heading_path else 2),
                    "heading_path": heading_path,
                    "heading_numeric_path": numeric_heading_path(heading_path),
                    "title": meta.title,
                    "author": meta.author,
                    "year": meta.year,
                    "source_pdf": pdf_path.name,
                    "text": seed_text, "type": "seed",
                    **make_flags(seed_text),
                })
            # Section summary → summary collection
    # Upsert to Qdrant (chunks)
    if docs_texts:
        Qdrant.from_texts(
            texts=docs_texts,
            embedding=embeddings,
            url=f"http://localhost:{QDRANT_PORT}",
            prefer_grpc=False,
            collection_name=COLLECTION,
            metadatas=docs_metas,
        )
    # Upsert to Qdrant (seeds)
    if seed_texts:
        Qdrant.from_texts(
            texts=seed_texts,
            embedding=embeddings,
            url=f"http://localhost:{QDRANT_PORT}",
            prefer_grpc=False,
            collection_name=SEED_COLLECTION,
            metadatas=seed_metas,
        )

    # ======= PRINT INGEST SUMMARY (metadata + qdrant counts) =======
    try:
        # Lọc theo book_id ở cả hai key-path: root và metadata.*
        flt = qmodels.Filter(should=[
            qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=str(book_id))),
            qmodels.FieldCondition(key="metadata.book_id", match=qmodels.MatchValue(value=str(book_id))),
        ])

        # Đếm số điểm trong từng collection
        chunks_cnt = int(qc.count(collection_name=COLLECTION, count_filter=flt).count or 0)
        seeds_cnt = int(qc.count(collection_name=SEED_COLLECTION, count_filter=flt).count or 0)

        print("\n========== INGEST SUMMARY ==========")
        print(f"Book ID     : {book_id}")
        print(f"Title       : {meta.title}")
        print(f"Author      : {meta.author}")
        print(f"Year        : {meta.year}")
        print(f"Source PDF  : {pdf_path.name}")
        print("------------------------------------")
        print(f"Qdrant [{COLLECTION}] chunks   : {chunks_cnt}")
        print(f"Qdrant [{SEED_COLLECTION}] seed: {seeds_cnt}")
        print("---------- INGEST USAGE (tokens ~ xấp xỉ) ----------")
        print(f"Embed docs : {usage['embedding_docs_tokens']}")
        print(f"Embed seeds: {usage['embedding_seed_tokens']}")
        print(f"Vision meta: {usage['vision_extract_tokens']}")
        print("----------------------------------------------------")

        # (tuỳ chọn) lấy vài section_id mẫu từ seed để bạn kiểm tra nhanh
        try:
            pts, _ = qc.scroll(
                collection_name=SEED_COLLECTION,
                scroll_filter=flt,
                with_payload=True,
                limit=5,
            )
            sample_secs = []
            for p in pts:
                pl = p.payload or {}
                sid = pl.get("section_id") or (pl.get("metadata", {}) or {}).get("section_id")
                if sid:
                    sample_secs.append(str(sid))
            if sample_secs:
                print(f"Seed sample section_id (≤5): {sample_secs}")
        except Exception:
            pass

        print("====================================\n")
    except Exception as _e:
        print(f"[warn] print ingest summary failed: {_e}")
    print(
        f"✅ Ingested: {pdf_path.name}  pages={len(pages)}  chunks={len(docs_texts)}  seeds={len(seed_texts)}")


# ----------------------------- main -----------------------------

def main():
    _ensure_event_loop()
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: python -m app.ingest <path/to/file.pdf> [more.pdf]\n(Upload & nạp được gọi từ Streamlit; ./data/books chỉ là nơi LƯU.)")
        return
    for a in args:
        p = Path(a)
        if p.is_file() and p.suffix.lower() == ".pdf":
            ingest_pdf(p)
        else:
            print(f"Skip (not a PDF): {a}")


if __name__ == "__main__":
    main()
