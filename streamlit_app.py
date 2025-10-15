# streamlit_app.py
import base64
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Local modules
from app import ingest as ingest_mod
from app import assistant as ra
from app.query import run_query

# ---------------------- Init & Config ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
# Use the same DATA_DIR convention as ingest.py so both point to one folder
DATA_DIR: Path = (PROJECT_ROOT / "data" / "books")
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
load_dotenv()
PGUSER = "postgres"
PGPASSWORD = "husky"
PGDATABASE = "postgres"
PGPORT = 55432
st.set_page_config(page_title="AI Reading Assistant", page_icon="📖", layout="wide")

# DB engine (tận dụng config từ ingest_mod)
ENGINE = create_engine(
    f"postgresql+psycopg2://{ingest_mod.PGUSER}:{ingest_mod.PGPASSWORD}"
    f"@localhost:{ingest_mod.PGPORT}/{ingest_mod.PGDATABASE}"
)

# Ensure schema exists on first load (idempotent)
try:
    ingest_mod.ensure_schema()
except Exception:
    pass

# ---------------------- DB helpers ----------------------

def list_books() -> List[Dict]:
    sql = """
    SELECT b.id::text AS book_id, b.title,
           COALESCE(MIN(p.page_no), 1) AS min_page,
           COALESCE(MAX(p.page_no), 0) AS max_page,
           COUNT(p.page_id) AS total_pages
    FROM books b
    LEFT JOIN pages p ON p.book_id = b.id
    GROUP BY b.id, b.title
    ORDER BY b.title ASC;
    """
    with ENGINE.begin() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return list(rows)


def get_reading_output(book_id: str, type_: str, page_range: Optional[Tuple[int, int]] = None) -> Optional[Dict]:
    if page_range:
        sql = """
        SELECT content
        FROM reading_outputs
        WHERE book_id = :b AND type = :t AND page_span = int4range(:s, :e, '[]')
        ORDER BY created_at DESC
        LIMIT 1;
        """
        params = {"b": book_id, "t": type_, "s": page_range[0], "e": page_range[1]}
    else:
        sql = """
        SELECT content
        FROM reading_outputs
        WHERE book_id = :b AND type = :t AND page_span IS NULL
        ORDER BY created_at DESC
        LIMIT 1;
        """
        params = {"b": book_id, "t": type_}
    with ENGINE.begin() as conn:
        row = conn.execute(text(sql), params).fetchone()
    return row[0] if row else None


# ---------------------- UI state ----------------------

def _ensure_session_defaults():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("selected_book_id", None)
    ss.setdefault("selected_range", {"start": None, "end": None})
    ss.setdefault("outline_cache", {})  # {book_id: outline_json}


_ensure_session_defaults()


# ---------------------- Utilities ----------------------

def _pretty_json(obj: Dict) -> str:
    try:
        import json as _json
        return _json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _find_pdf_for_book(title: str) -> Optional[Path]:
    # 1) khớp chính xác "<title>.pdf"
    direct = DATA_DIR / f"{title}.pdf"
    if direct.exists():
        return direct
    # 2) tìm gần đúng theo stem
    candidates = list(DATA_DIR.glob("*.pdf"))
    # ưu tiên tên chứa nguyên cụm title (không phân biệt hoa thường)
    title_l = title.lower()
    scored = sorted(
        candidates,
        key=lambda p: (0 if title_l in p.stem.lower() else 1, abs(len(p.stem) - len(title))),
    )
    return scored[0] if scored else None


def _embed_pdf(pdf_path: Path, height: int = 900):
    data = pdf_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    html = f"""
    <iframe src="data:application/pdf;base64,{b64}#toolbar=1&navpanes=1&scrollbar=1"
            style="width:100%;height:{height}px;border:none;"></iframe>
    """
    st.components.v1.html(html, height=height, scrolling=False)


def _chip(label: str, key: str) -> bool:
    return st.button(label, key=key, use_container_width=True)


def _ask_and_render(question: str):
    # Lưu tin nhắn người dùng
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    # Trả lời: chỉ những gì query.py sinh ra (không filter theo book_id)
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ…"):
            try:
                # Luôn tìm trên toàn bộ thư viện (bỏ book_id)
                res = run_query(question, k=20, book_id=None)

                # run_query được kỳ vọng trả về dict; phòng hờ nếu khác
                if not isinstance(res, dict):
                    res = {"answer": str(res)}

                # Hiển thị nguyên văn answer do query.py trả về
                answer_text = res.get("answer") or ""
                st.markdown(answer_text)

                # Hiển thị toàn bộ "raw output" đúng như query.py trả về
                with st.expander("Chi tiết (raw output từ query.py)"):
                    st.code(json.dumps(res, ensure_ascii=False, indent=2), language="json")

                # Lưu vào session
                st.session_state.messages.append(("assistant", answer_text))

            except Exception as e:
                st.error("❌ Lỗi khi xử lý yêu cầu.")
                st.exception(e)
                st.session_state.messages.append(("assistant", f"❌ {e}"))


# match PDF file for title

def find_pdf_for_title(title: str) -> Path | None:
    # exact match
    p = DATA_DIR / f"{title}.pdf"
    if p.exists():
        return p
    # fuzzy contains
    cands = sorted(DATA_DIR.glob("*.pdf"), key=lambda x: (title.lower() not in x.stem.lower(), abs(len(x.stem)-len(title))))
    return cands[0] if cands else None


# --- Markdown render helpers ---

def _md_summary(content: Dict) -> str:
    title = content.get("title", "Summary")
    abstract = content.get("abstract", "").strip()
    bullets = content.get("bullets", [])[:10]
    takeaways = content.get("takeaways", [])[:6]
    bl = "\n".join(f"- {b}" for b in bullets) or "- (không có)"
    tk = "\n".join(f"• {t}" for t in takeaways) or "• (không có)"
    return f"""**{title}**

{abstract}

**Key points**
{bl}

**Takeaways**
{tk}
"""


def _md_keypoints(content: Dict) -> str:
    kps = content.get("key_points", [])[:10]
    cons = content.get("conclusions", [])[:3]
    kp_md = "\n".join(f"- {k}" for k in kps) or "- (không có)"
    c_md = "\n".join(f"- {c}" for c in cons) or "- (không có)"
    return f"""**Key points**
{kp_md}

**Conclusions**
{c_md}
"""


def _md_quotes(content: Dict) -> str:
    qts = content.get("quotes", [])[:6]
    if not qts:
        return "> (không trích dẫn nào)"
    lines = []
    for q in qts:
        qtext = q.get("quote", "").strip()
        p = q.get("approx_page", "?")
        th = q.get("theme", "")
        lines.append(f"> {qtext}\n\n— p.{p} · {th}")
    return "\n\n".join(lines)


# ---------------------- Layout ----------------------
left, right = st.columns([7, 3], gap="large")

# ===== RIGHT: AI Reading Assistant (chips + chat) =====
with right:
    st.markdown("### 🤖 AI Reading Assistant")
    st.caption("Hỏi đáp về nội dung sách")

    st.markdown("**💡 Câu hỏi gợi ý – Click để hỏi ngay:**")
    c1, c2 = st.columns(2)
    with c1:
        if _chip("Tóm tắt chương 1", "sg_s1"):
            _ask_and_render("Tóm tắt chương 1 của cuốn sách này (nêu các điểm chính và số trang liên quan).")
        if _chip("Điểm chính cuốn sách", "sg_p1"):
            _ask_and_render("Liệt kê các điểm chính (key points) của cuốn sách, kèm số trang tham chiếu.")
        if _chip("Các ví dụ thực tế", "sg_ex"):
            _ask_and_render("Tìm các ví dụ/thực nghiệm thực tế tiêu biểu trong sách, tóm lược ngắn gọn.")
    with c2:
        if _chip("Gợi ý phần nên đọc", "sg_rec"):
            _ask_and_render("Gợi ý những phần nên đọc trước (vì dễ tiếp cận/quan trọng), kèm trang.")
        if _chip("Tìm phát biểu quan trọng", "sg_quote"):
            _ask_and_render("Tìm các phát biểu/định nghĩa quan trọng (trích dẫn nguyên văn) và trang.")
        if _chip("Kết luận chính", "sg_conc"):
            _ask_and_render("Tóm tắt các kết luận chính của sách, kèm trang nếu có.")

    st.markdown("---")

    # Lịch sử + input chat (chỉ 1 khung chat)
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_q = st.chat_input("Hỏi về nội dung sách…")
    if user_q:
        _ask_and_render(user_q)

# ===== LEFT: Viewer + ingest panel =====
with left:
    st.markdown("### 📚 Thư viện & Trình đọc")

    # --- Upload & Ingest panel ---
    with st.expander("📥 Thêm PDF vào thư viện", expanded=False):
        files = st.file_uploader("Chọn PDF", type=["pdf"], accept_multiple_files=True)
        force = st.checkbox("Force re-index (ghi đè nếu đã có)", value=False,
                            help="Bỏ qua kiểm tra đã index. Chỉ dùng khi bạn chắc muốn làm mới vectors.")

        if files and st.button("🚀 Nạp & index", type="primary"):
            saved_paths: List[Path] = []
            for f in files:
                dest = DATA_DIR / f.name
                dest.write_bytes(f.getbuffer())
                saved_paths.append(dest)

            with st.spinner("Đang ingest từng file vừa upload…"):
                for p in saved_paths:
                    try:
                        res = ingest_mod.ingest_pdf(p, skip_if_exists=True, force=force)
                        if res.get("skipped"):
                            st.info(f"⏭️ Bỏ qua (đã có): {res.get('file')} — {res.get('points')}")
                        else:
                            pts = res.get("points", {})
                            st.success(
                                f"✅ Đã nạp: {res.get('file')} — pages={res.get('pages')} chunks={pts.get('chunks')} seeds={pts.get('seeds')}")
                    except Exception as e:
                        st.error(f"❌ Lỗi ingest {p.name}: {e}")
            st.rerun()

    # 1) Lấy danh sách sách
    books = list_books()
    if not books:
        st.info("Chưa có sách nào trong thư viện. Hãy upload PDF ở trên và bấm Ingest.")
        st.stop()

    titles = [f"{b['title']} (pp. {b['min_page']}–{b['max_page']}, {b['total_pages']} trang)" for b in books]
    sel = st.selectbox(""
                       "", options=list(range(len(books))), format_func=lambda i: titles[i], key="book_select")
    book = books[sel]
    # GÁN book_id TRƯỚC khi vẽ RIGHT
    st.session_state["selected_book_id"] = book["book_id"]

    st.markdown(f"#### **{book['title']}**")
    st.caption(f"Số trang: {book['total_pages']} (pp. {book['min_page']}–{book['max_page']})")

    # ---- PDF.js viewer via streamlit-pdf-viewer ----
    pdf_path = find_pdf_for_title(book["title"]) or _find_pdf_for_book(book["title"])  # fallback to fuzzy finder
    if not pdf_path:
        st.error(f"Không tìm thấy PDF cho “{book['title']}” trong `{DATA_DIR}`.")
    else:
        try:
            from streamlit_pdf_viewer import pdf_viewer
            pdf_bytes = pdf_path.read_bytes()
            pdf_viewer(pdf_bytes, height=900, width=0, key=f"pdfv_{pdf_path.stem}")
            st.caption(f"Đang hiển thị: {pdf_path.name} • {pdf_path.stat().st_size/1e6:.1f} MB")
        except Exception:
            _embed_pdf(pdf_path, height=900)

st.markdown("---")
st.caption("© AI Reading Assistant — Streamlit + Qdrant + Gemini")
