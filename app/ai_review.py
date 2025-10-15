# app/ai_content_review.py
"""
AI Content Review (one-file version)

Phụ thuộc sẵn có trong project:
- Postgres tables: books(book_id, title, ...), pages(page_id, book_id, page_number, text)
- app.ingest: PGUSER, PGPASSWORD, PGPORT, PGDATABASE
- app.query: _llm()  -> LLM (Gemini) để gọi hiệu đính & moderation

Chức năng chính:
- review_book(book_id, start_page=None, end_page=None)
- review_range(book_id, start_page, end_page)
- review_page(book_id, page_number, text_override=None)
- get_issues(book_id, type=None, severity=None, status=None, limit=1000)
- export_issues_json(book_id), export_issues_csv(book_id)
- mark_issue_resolved(issue_id, by="user"), mark_issue_ignored(issue_id, by="user")
- get_issue_metrics(book_id)  # đếm theo type/severity/status

Có thể gọi trực tiếp từ Streamlit UI.
"""


from __future__ import annotations
import uuid, re, json, csv, io
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from sqlalchemy import create_engine, text
import json
from typing import Any, Dict

# ---- reuse project config & llm ----
try:
    from app.ingest import PGUSER, PGPASSWORD, PGPORT, PGDATABASE
except Exception:
    # Fallback khi chạy tách rời
    PGUSER = "postgres"; PGPASSWORD = "husky"; PGPORT = 55432; PGDATABASE = "postgres"

try:
    from app.query import _llm
except Exception as e:
    raise RuntimeError("Thiếu app.query._llm(). Hãy đảm bảo file query.py có hàm _llm().") from e

engine = create_engine(f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@localhost:{PGPORT}/{PGDATABASE}")

# ----------------------------- SCHEMA ---------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS content_issues (
  issue_id UUID PRIMARY KEY,
  book_id UUID NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
  page_number INT,                          -- NULL nếu áp dụng toàn sách
  issue_type TEXT NOT NULL,                 -- 'spelling'|'grammar'|'policy'|'sensitive'|'toxicity'|...
  span INT4RANGE,                           -- vị trí [start,end] trong text (NULL = không xác định)
  snippet TEXT,                             -- vài câu quanh vị trí lỗi / mô tả ngắn
  suggestion TEXT,                          -- gợi ý sửa (nếu có)
  severity TEXT,                            -- 'low'|'med'|'high'|'critical'
  status TEXT NOT NULL DEFAULT 'open',      -- 'open'|'resolved'|'ignored'
  details JSONB,                            -- điểm số model, nhãn phụ, nguồn detector, ...
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,
  resolved_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_ci_book ON content_issues(book_id);
CREATE INDEX IF NOT EXISTS idx_ci_type ON content_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_ci_status ON content_issues(status);
"""

def _ensure_schema():
    with engine.begin() as conn:
        for s in SCHEMA_SQL.strip().split(";"):
            ss = s.strip()
            if ss:
                conn.execute(text(ss))

# gọi sớm để chắc chắn có bảng
_ensure_schema()

# ----------------------------- CONFIG ---------------------------------

@dataclass
class ReviewConfig:
    # LLM
    llm_temp_spelling: float = 0.1
    llm_temp_moderation: float = 0.1

    # page batching (nếu tự chia lô ở UI thì không cần)
    max_chars_per_llm_call: int = 12000

    # moderation thresholds
    high_threshold: float = 0.70
    med_threshold: float = 0.40

    # bật/tắt detector
    enable_spell_grammar: bool = True
    enable_pii_regex: bool = True
    enable_moderation: bool = True

DEFAULT_CFG = ReviewConfig()

# ----------------------------- HELPERS --------------------------------

def _json_loads_safe(txt: str) -> Dict[str, Any]:
    """
    Cố gắng parse JSON trực tiếp; nếu fail, tìm object JSON hợp lệ đầu tiên
    bằng cách đếm dấu ngoặc { } (có xử lý chuỗi/escape).
    Trả về dict rỗng nếu không tách được JSON hợp lệ.
    """
    # 1) thử parse trực tiếp
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 2) fallback: quét từ dấu '{' đầu tiên, đếm ngoặc
    n = len(txt)
    start = txt.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, n):
            ch = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = txt[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            # nếu đoạn vừa tách vẫn không phải JSON hợp lệ,
                            # tiếp tục tìm dấu '{' kế tiếp
                            break
        # tìm dấu '{' tiếp theo sau vị trí start hiện tại
        start = txt.find("{", start + 1)

    # 3) không tìm được JSON hợp lệ
    return {}

def _pages(book_id: str, page_range: Optional[Tuple[int,int]] = None) -> List[Dict[str, Any]]:
    sql = "SELECT page_number, text FROM pages WHERE book_id=:b"
    params: Dict[str, Any] = {"b": book_id}
    if page_range:
        sql += " AND page_number BETWEEN :s AND :e"
        params.update({"s": page_range[0], "e": page_range[1]})
    sql += " ORDER BY page_number ASC"
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return list(rows)

def _insert_issue(
    *, book_id: str, page_number: Optional[int], issue_type: str,
    span: Optional[Tuple[Optional[int], Optional[int]]], snippet: str,
    suggestion: Optional[str], severity: str, details: Dict[str, Any],
    status: str = "open"
) -> str:
    _ensure_schema()
    with engine.begin() as conn:
        iid = str(uuid.uuid4())
        conn.execute(text("""
            INSERT INTO content_issues (issue_id, book_id, page_number, issue_type, span, snippet, suggestion, severity, status, details)
            VALUES (:id, :b, :p, :t, int4range(:s, :e, '[]'), :snip, :sugg, :sev, :st, :det)
        """), {
            "id": iid,
            "b": book_id,
            "p": page_number,
            "t": issue_type,
            "s": span[0] if span else None,
            "e": span[1] if span else None,
            "snip": (snippet or "")[:1000],
            "sugg": (suggestion or "")[:1000],
            "sev": severity,
            "st": status,
            "det": json.dumps(details or {}, ensure_ascii=False),
        })
    return iid

# ----------------------------- DETECTORS -------------------------------

# 1) LLM-based spelling/grammar
_SPELL_SYS = """Bạn là trợ lý hiệu đính. Trả về JSON dạng:
{
  "issues":[
    { "span":[start,end], "snippet":"...", "suggestion":"...", "type":"spelling|grammar" }
  ]
}
- Chỉ liệt kê lỗi rõ ràng (chính tả/ngữ pháp), không chỉnh phong cách.
- Giữ nguyên ngôn ngữ gốc (Việt/Anh).
- span đếm theo ký tự trong đoạn đưa vào.
"""

def _detect_spell_grammar(text_in: str, *, cfg: ReviewConfig) -> List[Dict[str, Any]]:
    if not text_in.strip():
        return []
    llm = _llm(cfg.llm_temp_spelling)
    resp = llm.invoke(_SPELL_SYS + "\n\n[TEXT]\n" + text_in[:cfg.max_chars_per_llm_call])
    data = _json_loads_safe(getattr(resp, "content", str(resp)))
    out = []
    for it in data.get("issues", []) or []:
        itype = (it.get("type") or "").strip().lower()
        if itype not in ("spelling", "grammar"):
            continue
        span = it.get("span") or [None, None]
        s, e = (span + [None, None])[:2]
        out.append({
            "issue_type": itype,
            "span": (s, e) if s is not None and e is not None else None,
            "snippet": it.get("snippet", "")[:500],
            "suggestion": it.get("suggestion", "")[:500],
            "severity": "low",
            "details": {"source":"llm_spell"}
        })
    return out

# 2) PII regex (email, điện thoại, CCCD, thẻ)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Điện thoại VN cơ bản: dãy 9-11 số (có thể có dấu cách/gạch) — để đơn giản dùng chuỗi số >=9
_PHONE_RE = re.compile(r"(?<!\d)\d[\d\-\s]{8,16}\d(?!\d)")
# CCCD 12 số, CMND 9 số
_CCCD_RE  = re.compile(r"(?<!\d)\d{12}(?!\d)")
_CMND_RE  = re.compile(r"(?<!\d)\d{9}(?!\d)")
# Thẻ tín dụng (rất thô): 13-19 số (cho cảnh báo)
_CARD_RE  = re.compile(r"(?<!\d)\d[\d\-\s]{11,23}\d(?!\d)")

def _detect_pii_regex(text_in: str, *, cfg: ReviewConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not text_in:
        return out

    def _add(m: re.Match, label: str, sev: str):
        s, e = m.start(), m.end()
        snip = text_in[max(0, s-40): e+40]
        out.append({
            "issue_type": "sensitive",
            "span": (s, e),
            "snippet": snip,
            "suggestion": None,
            "severity": sev,
            "details": {"source":"regex_pii", "label": label}
        })

    for m in _EMAIL_RE.finditer(text_in):
        _add(m, "email", "med")
    for m in _PHONE_RE.finditer(text_in):
        # lọc bớt chuỗi quá nhiều dấu cách/gạch
        digits = re.sub(r"\D", "", m.group(0))
        if 9 <= len(digits) <= 13:
            _add(m, "phone", "med")
    for m in _CCCD_RE.finditer(text_in):
        _add(m, "cccd", "high")
    for m in _CMND_RE.finditer(text_in):
        _add(m, "cmnd", "med")
    for m in _CARD_RE.finditer(text_in):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19:
            _add(m, "card_number_like", "high")

    return out

# 3) Moderation LLM
_MOD_SYS = """Bạn là bộ lọc kiểm duyệt nội dung. Hãy phân loại đoạn văn theo nhãn:
["safe","sexual","violence","self-harm","hate","toxicity","politics","drugs","pii","other-sensitive"].
Trả về JSON:
{
  "labels":[{"label":"...", "score":0..1, "reason":"..."}],
  "overall":"<label>"
}
- Chỉ gắn nhãn khi có dấu hiệu rõ ràng; nếu không, 'overall' = 'safe'.
"""

def _detect_moderation(text_in: str, *, cfg: ReviewConfig) -> List[Dict[str, Any]]:
    if not text_in.strip():
        return []
    llm = _llm(cfg.llm_temp_moderation)
    resp = llm.invoke(_MOD_SYS + "\n\n[TEXT]\n" + text_in[:cfg.max_chars_per_llm_call])
    data = _json_loads_safe(getattr(resp, "content", str(resp)))

    labels = { (d.get("label") or "").lower(): float(d.get("score") or 0.0)
               for d in (data.get("labels") or []) }
    overall = (data.get("overall") or "safe").lower()

    if overall == "safe":
        return []

    # severity rule-of-thumb
    sev = "med"
    if any(labels.get(k, 0.0) >= cfg.high_threshold for k in ["sexual","violence","hate","self-harm"]):
        sev = "high"
    if any(labels.get(k, 0.0) >= 0.9 for k in labels):  # rất tự tin
        sev = "critical"

    return [{
        "issue_type": "policy",
        "span": None,
        "snippet": text_in[:400],
        "suggestion": None,
        "severity": sev,
        "details": {"source":"llm_moderation", "labels": labels, "overall": overall}
    }]

# ----------------------------- PIPELINE --------------------------------

def review_page(
    book_id: str,
    page_number: int,
    text_override: Optional[str] = None,
    cfg: ReviewConfig = DEFAULT_CFG
) -> List[str]:
    """
    Rà soát 1 trang. Trả về danh sách issue_id đã ghi.
    """
    if text_override is None:
        rows = _pages(book_id, (page_number, page_number))
        if not rows:
            return []
        txt = rows[0]["text"] or ""
    else:
        txt = text_override or ""

    created_ids: List[str] = []

    if cfg.enable_spell_grammar:
        for it in _detect_spell_grammar(txt, cfg=cfg):
            created_ids.append(_insert_issue(
                book_id=book_id, page_number=page_number,
                issue_type=it["issue_type"], span=it["span"],
                snippet=it["snippet"], suggestion=it.get("suggestion"),
                severity=it["severity"], details=it["details"]
            ))

    if cfg.enable_pii_regex:
        for it in _detect_pii_regex(txt, cfg=cfg):
            created_ids.append(_insert_issue(
                book_id=book_id, page_number=page_number,
                issue_type=it["issue_type"], span=it["span"],
                snippet=it["snippet"], suggestion=None,
                severity=it["severity"], details=it["details"]
            ))

    if cfg.enable_moderation:
        for it in _detect_moderation(txt, cfg=cfg):
            created_ids.append(_insert_issue(
                book_id=book_id, page_number=page_number,
                issue_type=it["issue_type"], span=None,
                snippet=it["snippet"], suggestion=None,
                severity=it["severity"], details=it["details"]
            ))

    return created_ids

def review_range(
    book_id: str, start_page: int, end_page: int, cfg: ReviewConfig = DEFAULT_CFG
) -> Dict[str, Any]:
    """
    Rà soát một khoảng trang. Trả về thống kê cơ bản.
    """
    total_pages = 0
    total_issues = 0
    for row in _pages(book_id, (start_page, end_page)):
        total_pages += 1
        ids = review_page(book_id, row["page_number"], text_override=row["text"] or "", cfg=cfg)
        total_issues += len(ids)
    return {"pages": total_pages, "issues": total_issues, "book_id": book_id, "range": [start_page, end_page]}

def review_book(book_id: str, cfg: ReviewConfig = DEFAULT_CFG) -> Dict[str, Any]:
    """
    Rà soát toàn bộ sách. (Nếu sách lớn, cân nhắc gọi theo batch range ở UI.)
    """
    rows = _pages(book_id, None)
    total_pages = len(rows)
    total_issues = 0
    for row in rows:
        ids = review_page(book_id, row["page_number"], text_override=row["text"] or "", cfg=cfg)
        total_issues += len(ids)
    return {"pages": total_pages, "issues": total_issues, "book_id": book_id}

# ----------------------------- QUERIES ---------------------------------

def get_issues(
    book_id: str,
    *,
    type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    sql = "SELECT issue_id, book_id, page_number, issue_type, span, snippet, suggestion, severity, status, details, created_at, resolved_at, resolved_by FROM content_issues WHERE book_id=:b"
    params: Dict[str, Any] = {"b": book_id}
    if type:
        sql += " AND issue_type=:t"; params["t"] = type
    if severity:
        sql += " AND severity=:sv"; params["sv"] = severity
    if status:
        sql += " AND status=:st"; params["st"] = status
    sql += " ORDER BY created_at DESC LIMIT :lim"
    params["lim"] = int(limit)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    # cast span text -> tuple (đơn giản hoá hiển thị)
    def _span_to_tuple(v: Any) -> Optional[Tuple[int,int]]:
        if v is None: return None
        # expect like "[12,34]"
        m = re.match(r"[\[\(](\d+),(\d+)[\]\)]", str(v))
        if not m: return None
        return (int(m.group(1)), int(m.group(2)))
    out = []
    for r in rows:
        d = dict(r)
        d["span"] = _span_to_tuple(r["span"])
        out.append(d)
    return out

def mark_issue_resolved(issue_id: str, *, by: str = "user") -> bool:
    with engine.begin() as conn:
        n = conn.execute(text("""
            UPDATE content_issues
               SET status='resolved', resolved_at=NOW(), resolved_by=:by
             WHERE issue_id=:id
        """), {"id": issue_id, "by": by}).rowcount
    return n > 0

def mark_issue_ignored(issue_id: str, *, by: str = "user") -> bool:
    with engine.begin() as conn:
        n = conn.execute(text("""
            UPDATE content_issues
               SET status='ignored', resolved_at=NOW(), resolved_by=:by
             WHERE issue_id=:id
        """), {"id": issue_id, "by": by}).rowcount
    return n > 0

def get_issue_metrics(book_id: str) -> Dict[str, Any]:
    sql = """
    SELECT issue_type, severity, status, COUNT(*) AS c
      FROM content_issues
     WHERE book_id=:b
     GROUP BY issue_type, severity, status
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"b": book_id}).mappings().all()
    # gộp thành dict lồng nhau: metrics[type][severity][status] = count
    metrics: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in rows:
        t = r["issue_type"]; sv = r["severity"]; st = r["status"]; c = int(r["c"])
        metrics.setdefault(t, {}).setdefault(sv, {})[st] = c
    return {"book_id": book_id, "metrics": metrics}

# ----------------------------- EXPORT ----------------------------------

def export_issues_json(book_id: str, *, type: Optional[str]=None, severity: Optional[str]=None, status: Optional[str]=None, limit: int=10000) -> str:
    issues = get_issues(book_id, type=type, severity=severity, status=status, limit=limit)
    return json.dumps({"book_id": book_id, "issues": issues}, ensure_ascii=False, indent=2)

def export_issues_csv(book_id: str, *, type: Optional[str]=None, severity: Optional[str]=None, status: Optional[str]=None, limit: int=10000) -> str:
    issues = get_issues(book_id, type=type, severity=severity, status=status, limit=limit)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["issue_id","book_id","page_number","issue_type","span_start","span_end","severity","status","snippet","suggestion","details","created_at","resolved_at","resolved_by"])
    for it in issues:
        s = it.get("span") or (None, None)
        w.writerow([
            it.get("issue_id"), it.get("book_id"), it.get("page_number"), it.get("issue_type"),
            s[0], s[1], it.get("severity"), it.get("status"),
            (it.get("snippet") or "").replace("\n"," ")[:300],
            (it.get("suggestion") or "").replace("\n"," ")[:300],
            json.dumps(it.get("details") or {}, ensure_ascii=False)[:500],
            it.get("created_at"), it.get("resolved_at"), it.get("resolved_by")
        ])
    return buf.getvalue()
