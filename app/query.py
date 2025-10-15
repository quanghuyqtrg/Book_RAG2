from __future__ import annotations
import os
import json
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# -------------------- config & logging --------------------
load_dotenv()
RAG_MIN_BEST_SCORE = float(os.getenv("RAG_MIN_BEST_SCORE", "0.15"))   # was 0.25
RAG_MIN_CONTEXT_CHARS = int(os.getenv("RAG_MIN_CONTEXT_CHARS", "300"))  # was 600
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "books_rag")
SEED_COLLECTION = os.getenv("QDRANT_SEED_COLLECTION", "book_rag_seed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GENAI_QUERY_MODEL = (
    os.getenv("GENAI_QUERY_MODEL")
    or os.getenv("GENAI_SUMMARY_MODEL")
    or "gemini-2.5-flash"
)

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | query | %(message)s",
)
log = logging.getLogger("query")

# query.py (đầu file hoặc trong class)

BOOK_ID_KEYS       = ["book_id", "metadata.book_id"]
SECTION_ID_KEYS    = ["section_id", "metadata.section_id"]
PAGE_NUMBER_KEYS   = ["page_number", "metadata.page_number"]
SECTION_ORDER_KEYS = ["section_order", "metadata.section_order"]



def _or_range(self, keys: list[str], gte=None, lte=None):
    rng = qmodels.Range(gte=gte, lte=lte)
    return [qmodels.FieldCondition(key=k, range=rng) for k in keys]

# -------------------- datatypes --------------------
@dataclass
class Hit:
    score: float
    payload: Dict[str, Any]

    @property
    def section_id(self) -> str:
        pl = self.payload or {}
        return pl.get("section_id") or (pl.get("metadata", {}) or {}).get("section_id") or ""

    @property
    def text(self) -> str:
        pl = self.payload or {}
        return pl.get("text") or pl.get("page_content") or ""

    @property
    def section_order(self) -> Optional[int]:
        so = self.payload.get("section_order")
        try:
            return int(so) if so is not None else None
        except Exception:
            return None

    @property
    def page(self) -> Optional[int]:
        p = self.payload.get("page_number") or self.payload.get("page_no")
        try:
            return int(p) if p is not None else None
        except Exception:
            return None

    @property
    def heading(self) -> Optional[str]:
        hp = self.payload.get("heading_path")
        if isinstance(hp, (list, tuple)) and hp:
            return str(hp[-1])
        return None

    @property
    def heading_level(self) -> int:
        hl = self.payload.get("heading_level")
        try:
            return int(hl) if hl is not None else 2
        except Exception:
            return 2

    @property
    def node_type(self) -> str:
        return str(self.payload.get("type") or "text")


# -------------------- helpers --------------------
class GeminiEmb(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts=texts, task_type="RETRIEVAL_DOCUMENT")
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text=text, task_type="RETRIEVAL_QUERY")


def _llm(temp: float = 0.0) -> Optional[ChatGoogleGenerativeAI]:
    if not API_KEY:
        return None
    return ChatGoogleGenerativeAI(model=GENAI_QUERY_MODEL, google_api_key=API_KEY, temperature=temp)

def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
# -------------------- core engine --------------------
class QueryEngine:
    def __init__(self, url: str = "http://localhost:", port: int = QDRANT_PORT):
        _ensure_event_loop()
        self.has_summary = False
        self.qc = QdrantClient(url=f"{url}{port}", prefer_grpc=False)
        self.emb = GeminiEmb(model=EMBED_MODEL, google_api_key=API_KEY)
        self.llm = _llm(0.0)
        # cache collections to avoid searching missing ones
        try:
            self._collections = {c.name for c in self.qc.get_collections().collections}
        except Exception:
            self._collections = set()
        self.has_seed = SEED_COLLECTION in self._collections

    # -------- query rewrite --------
    def rewrite_query(self, question: str) -> Dict[str, Any]:
        if not self.llm:
            return {"rewritten": question.strip(), "keywords": []}
        prompt = (
            "You are a query rewriting assistant for RAG.\n"
            "Rewrite the user's question to maximize document retrieval recall while keeping meaning.\n"
            "Return STRICT JSON with keys: rewritten (string), keywords (array of 3-8 tokens/phrases).\n\n"
            f"[QUESTION]\n{question}"
        )
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            data = json.loads(content)
            rw = str(data.get("rewritten") or question).strip()
            kws = [str(x).strip() for x in (data.get("keywords") or [])]
            return {"rewritten": rw, "keywords": kws}
        except Exception:
            return {"rewritten": question.strip(), "keywords": []}

    # -------- low-level search wrappers --------
    # trong QueryEngine
    BOOK_ID_KEYS = ["book_id", "metadata.book_id"]
    SECTION_ID_KEYS = ["section_id", "metadata.section_id"]

    def _or_eq(self, keys, value):
        return [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=value)) for k in keys]

    def _filter_book(self, book_id: str | None, extra_must=None):
        if not book_id and not extra_must:
            return None
        flt = qmodels.Filter()
        if extra_must:
            flt.must = list(extra_must)
        if book_id:
            flt.should = self._or_eq(BOOK_ID_KEYS, str(book_id))  # OR 2 key-path
        return flt

    def _filter_sections(self, section_ids: list[str] | None):
        if not section_ids:
            return []
        uniq, seen = [], set()
        for s in section_ids:
            if s is None: continue
            v = str(s).strip()
            if not v or v in seen: continue
            seen.add(v);
            uniq.append(v)
        if not uniq:
            return []
        if len(uniq) == 1:
            return self._or_eq(SECTION_ID_KEYS, uniq[0])  # 1 section: OR 2 key-path
        # nhiều section: MatchAny cho mỗi key-path
        return [
            qmodels.FieldCondition(key="section_id", match=qmodels.MatchAny(any=uniq)),
            qmodels.FieldCondition(key="metadata.section_id", match=qmodels.MatchAny(any=uniq)),
        ]

    def _search(self, collection: str, vector: List[float], k: int,
                flt: Optional[qmodels.Filter] = None) -> List[Hit]:
        if collection not in self._collections:
            return []
        try:
            res = self.qc.search(
                collection_name=collection,
                query_vector=vector,
                limit=k,
                with_payload=True,
                score_threshold=None,
                query_filter=flt,
            )
        except Exception as e:
            log.warning(f"search failed on {collection}: {e}")
            return []
        return [Hit(score=r.score, payload=r.payload or {}) for r in res]

    def _search_in_sections(self, vector: List[float], sections: Sequence[str], k: int,
                            book_id: Optional[str]) -> List[Hit]:
        must: List[qmodels.FieldCondition] = []
        if book_id:
            must.append(qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=str(book_id))))
        if sections:
            must.append(qmodels.FieldCondition(key="section_id", match=qmodels.MatchAny(any=list(map(str, sections)))))
        flt = qmodels.Filter(must=must) if must else None
        return self._search(COLLECTION, vector, k, flt)

    def _search_by_neighbors(self, vector: List[float], page: int, section_order: int, window: int,
                             book_id: Optional[str], k: int = 12) -> List[Hit]:
        must: List[qmodels.FieldCondition] = []
        if book_id:
            must.append(qmodels.FieldCondition(key="book_id", match=qmodels.MatchValue(value=str(book_id))))
        must.append(qmodels.FieldCondition(key="page_number", match=qmodels.MatchValue(value=int(page))))
        must.append(qmodels.FieldCondition(
            key="section_order",
            range=qmodels.Range(gte=int(section_order - window), lte=int(section_order + window))
        ))
        flt = qmodels.Filter(must=must)
        return self._search(COLLECTION, vector, k, flt)

    # -------- reranking & aggregation --------
    @staticmethod
    def _distinct_by_text(hits: List[Hit]) -> List[Hit]:
        seen = set()
        uniq: List[Hit] = []
        for h in hits:
            t = (h.text or "").strip()
            key = t[:80]
            if not t or key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        return uniq

    @staticmethod
    def _section_rank_bonus(section_id: Optional[str], ranked_sections: List[str]) -> float:
        if not section_id or not ranked_sections:
            return 0.0
        try:
            r = ranked_sections.index(section_id)
        except ValueError:
            return 0.0
        return max(0.0, 1.0 - (r * 0.2))  # top‑1 gets 1.0, top‑5 ~ 0.2

    @staticmethod
    def _level_prior(heading_level: int) -> float:
        pri = {1: 0.15, 2: 0.10, 3: 0.05, 4: 0.0}
        return pri.get(int(heading_level), 0.05)

    @staticmethod
    def _type_prior(node_type: str) -> float:
        # light penalty for captions so they don't dominate
        if node_type == "object_caption":
            return -0.04
        return 0.0

    def rerank(self, hits: List[Hit], ranked_sections: List[str]) -> List[Tuple[Hit, float]]:
        rescored: List[Tuple[Hit, float]] = []
        for h in hits:
            s = float(h.score or 0)
            s += 0.18 * self._section_rank_bonus(h.section_id, ranked_sections)
            s += 0.06 * self._level_prior(h.heading_level)
            s += self._type_prior(h.node_type)
            rescored.append((h, s))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored

    # -------- context building & answer generation --------
    @staticmethod
    def build_context(hits: Sequence[Hit], max_chars: int = 10000) -> Tuple[str, List[Dict[str, Any]]]:
        ctx_parts: List[str] = []
        cites: List[Dict[str, Any]] = []
        used = 0
        for h in hits:
            txt = (h.text or "").strip()
            if not txt:
                continue
            frag = txt
            if used + len(frag) > max_chars:
                frag = frag[: max(0, max_chars - used)]
            used += len(frag)
            tag_extra = " [FIG/TA]" if h.node_type == "object_caption" else ""
            tag = f"(p.{h.page}{' — ' + h.heading if h.heading else ''}){tag_extra}"
            ctx_parts.append(f"{frag}\n{tag}")
            cites.append({
                "page": h.page,
                "heading": h.heading,
                "section_id": h.section_id,
                "score": h.score,
                "type": h.node_type,
            })
            if used >= max_chars:
                break
        return "\n\n".join(ctx_parts).strip(), cites

    def _make_quote_from_hits(self, hits, max_chars=400) -> str:
        if not hits:
            return ""

        h = hits[0]  # chọn đoạn top-1 (hoặc hit tốt nhất)
        txt = (h.text or "").strip()

        # Giới hạn độ dài
        if len(txt) > max_chars:
            txt = txt[:max_chars]

        return txt

    def generate(self, question: str, context: str) -> str:
        sys_prompt = """
      Bạn là trợ lý RAG cực kỳ thận trọng. Chỉ sử dụng THÔNG TIN có trong CONTEXT để trả lời. Trả lời bằng tiếng Việt.

ĐẦU RA BẮT BUỘC — TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON HỢP LỆ:
{
  "found": true|false,
  "support": {
    "quote": "< Một vài trích đoạn ngắn (≤6000 ký tự) trực tiếp hỗ trợ cho câu trả lời; mỗi trích đoạn kết thúc bằng xuống dòng. Ưu tiên NGUYÊN VĂN từ CONTEXT >",
  },
  "answer": "<câu trả lời ngắn gọn, CHỈ dựa trên CONTEXT>",
  "quiz": [
    {
      "question": "Câu hỏi (dựa 100% vào CONTEXT, không suy đoán)",
      "answers": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct": "A|B|C|D"
    }
    // Nếu không phải yêu cầu trắc nghiệm thì để mảng rỗng []
  ]
}

QUY TẮC:
- Nếu KHÔNG đủ bằng chứng trong CONTEXT:
  • "found" = false
  • "answer" = một câu từ chối lịch sự, ngắn
  • "quiz" = []
- Nếu CÓ bằng chứng: "found" = true.

- Cách điền page_display/heading:
  • Mỗi đoạn CONTEXT có thể kết thúc bằng "(p.<nhãn trang> — <heading>)" hoặc "(p.<nhãn trang>)".
  • "page_display": LẤY đúng phần sau "p." (ví dụ "xii", "12", "12–14"). Nếu không có thẻ thì để null.
  • "heading": Nếu có phần sau dấu "—" thì lấy nguyên văn; nếu không có thì null.

- Trắc nghiệm (quiz):
  • CHỈ tạo "quiz" khi người dùng RÕ RÀNG yêu cầu trắc nghiệm/quiz (ví dụ: “đặt N câu hỏi trắc nghiệm…”, “quiz…”, “trắc nghiệm…”). Nếu không, để "quiz": [].
  • Nếu người dùng chỉ định số lượng N, tạo đúng N mục; nếu KHÔNG chỉ định, mặc định tạo 1 mục.
  • Mỗi mục gồm đúng 3–5 phương án; mặc định 4 phương án ["A.","B.","C.","D."].
  • "correct" PHẢI là một trong {"A","B","C","D"} và PHẢI khớp nội dung trong "answers".
  • Nội dung câu hỏi và đáp án phải có thể được CHỨNG MINH trực tiếp từ CONTEXT. Không suy đoán, không thêm kiến thức ngoài.
  • Tránh trùng lặp câu hỏi; dùng ngôn ngữ ngắn gọn, rõ ràng, chỉ hỏi 1 ý/ câu.
  • Nếu thông tin trong CONTEXT mơ hồ/thiếu để lập câu hỏi có đáp án đơn nhất → KHÔNG tạo quiz (đặt "quiz": [] và xử lý theo quy tắc thiếu bằng chứng nếu cần).
- Tóm tắt ( answer):
  • Chỉ sử dụng phương pháp tóm tắt khi người dùng sử dụng các câu hỏi về nội dung và câu hỏi tóm tắt đoạn 
  • câu trả lời mang tính chất đọc hiểu đoạn văn không quá cứng nhắc 

- KHÔNG thêm bất kỳ văn bản nào ngoài JSON (không markdown, không giải thích, không tiền tố/hậu tố).
- Tuyệt đối không suy đoán ngoài CONTEXT.

Bạn sẽ nhận:
<QUESTION>...</QUESTION>
<CONTEXT>...</CONTEXT>
Chỉ dựa vào CONTEXT để tạo JSON ở trên.
"""
        if not self.llm:
            return (
                "[NO LLM KEY]\n"
                "Below is a stitched context from retrieval; manually inspect citations.\n\n" + context
            )
        prompt = (
            f"<SYSTEM>\n{sys_prompt}\n</SYSTEM>\n\n"
            f"<QUESTION>\n{question}\n</QUESTION>\n\n"
            f"<CONTEXT>\n{context}\n</CONTEXT>\n"
        )
        try:
            resp = self.llm.invoke(prompt)
            return getattr(resp, "content", str(resp))
        except Exception as e:
            log.warning(f"LLM generation failed: {e}")
            return context

    # -------- retrieval strategies --------
    def seed_first_retrieve(self, qvec: List[float], book_id: Optional[str],
                            top_sections: int = 8, per_section_k: int = 6) -> List[Hit]:
        """
        1) Tìm seed trong SEED_COLLECTION → rút các section_id tốt nhất (tối đa top_sections)
        2) Tìm chunks trong COLLECTION theo (book_id ∧ section_id ∈ {…})
        3) Fallback: nếu theo section không có hit, thử lại book-only
        4) Rerank theo thứ tự section đã chọn
        """
        if not self.has_seed:
            return []

        # 1) SEED search (lọc theo book nếu có)
        seed_flt = self._filter_book(book_id)  # OR giữa book_id và metadata.book_id nếu bạn đã cập nhật _filter_book
        seed_hits = self._search(SEED_COLLECTION, qvec,
                                 k=max(20, top_sections * 2),
                                 flt=seed_flt)
        seed_hits = self._distinct_by_text(seed_hits)

        # 2) Rút ra các section_id duy nhất (ưu tiên theo thứ tự xuất hiện)
        sec_ids: List[str] = []
        for h in seed_hits:
            pl = h.payload or {}
            sid = pl.get("section_id") or (pl.get("metadata", {}) or {}).get("section_id")
            if sid:
                s = str(sid).strip()
                if s and s not in sec_ids:
                    sec_ids.append(s)
            if len(sec_ids) >= top_sections:
                break

        # Nếu không có section nào → fallback thẳng sang book-only / all-books
        if not sec_ids:
            # book-only
            if book_id:
                chunk_hits = self._search(COLLECTION, qvec,
                                          k=per_section_k * max(1, top_sections),
                                          flt=self._filter_book(book_id))
            else:
                # toàn bộ collection
                chunk_hits = self._search(COLLECTION, qvec,
                                          k=per_section_k * max(1, top_sections),
                                          flt=None)
            rescored = self.rerank(self._distinct_by_text(chunk_hits), ranked_sections=[])
            return [h for (h, _) in rescored]

        # 3) Drill chunks theo (book_id ∧ section_id ∈ {sec_ids})
        #    Dùng _filter_sections nếu bạn đã thêm; nếu chưa, dùng phương án B (inline)
        if hasattr(self, "_filter_sections"):
            extra_must = self._filter_sections(sec_ids)  # sẽ tạo OR giữa section_id và metadata.section_id
        else:
            # Phương án B: inline 2 điều kiện MatchAny cho 2 key-path
            extra_must = [
                qmodels.FieldCondition(key="section_id", match=qmodels.MatchAny(any=sec_ids)),
                qmodels.FieldCondition(key="metadata.section_id", match=qmodels.MatchAny(any=sec_ids)),
            ]

        flt = self._filter_book(book_id, extra_must=extra_must)
        k_total = max(per_section_k * len(sec_ids), per_section_k * top_sections)
        chunk_hits: List[Hit] = self._search(COLLECTION, qvec, k=k_total, flt=flt)

        # 4) Fallback: nếu lọc theo section không ra gì, thử lại book-only
        if not chunk_hits and book_id:
            chunk_hits = self._search(COLLECTION, qvec, k=k_total, flt=self._filter_book(book_id))

        # 5) Rerank và distinct
        rescored = self.rerank(self._distinct_by_text(chunk_hits), ranked_sections=sec_ids)
        return [h for (h, _) in rescored]
    def direct_retrieve(self, qvec: List[float], book_id: Optional[str], k: int = 24) -> List[Hit]:
        hits = self._search(COLLECTION, qvec, k=k, flt=self._filter_book(book_id))
        return self._distinct_by_text(hits)

    def iterative_retrieval(self, qvec: List[float], book_id: Optional[str],
                            target_chars: int = 2400) -> List[Hit]:
        # round 1: seed → summary → direct (whichever available first with enough context)
        hits: List[Hit] = []
        if self.has_seed:
            hits = self.seed_first_retrieve(qvec, book_id, top_sections=8, per_section_k=6)
        if not hits:
            hits = self.direct_retrieve(qvec, book_id, k=24)

        context, _ = self.build_context(hits, max_chars=target_chars)
        if len(context) >= target_chars * 0.7:
            return hits

        # round 2: add neighbor sections around top‑2 sections
        top2 = [h for h in hits if h.section_id][:2]
        added: List[Hit] = []
        for h in top2:
            if h.page is None or h.section_order is None:
                continue
            added.extend(self._search_by_neighbors(qvec, page=h.page, section_order=h.section_order,
                                                   window=1, book_id=book_id, k=12))
        merged = self._distinct_by_text(hits + added)
        rescored = self.rerank(merged, ranked_sections=[h.section_id for h in top2 if h.section_id])
        hits2 = [h for (h, _) in rescored]
        context2, _ = self.build_context(hits2, max_chars=target_chars)
        if len(context2) >= target_chars * 0.85:
            return hits2

        # round 3: widen search
        more = self.direct_retrieve(qvec, book_id, k=48)
        merged2 = self._distinct_by_text(hits2 + more)
        rescored2 = self.rerank(merged2, ranked_sections=[h.section_id for h in hits2 if h.section_id][:6])
        return [h for (h, _) in rescored2]

    # -------- public API (returns dict) --------
    def run_query(self, question: str, book_id: Optional[str] = None, k: int = 20,
                  target_chars: int = 6000, dry_run: bool = False) -> Dict[str, Any]:
        log.info(f"run_query question={question!r} book_id={book_id} k={k} dry_run={dry_run}")

        # 1) rewrite
        rw = self.rewrite_query(question)
        qtext = rw.get("rewritten", question)
        if rw.get("keywords"):
            log.info(f"rewrite: {qtext} | keywords={rw['keywords']}")

        # 2) embed
        qvec = self.emb.embed_query(qtext)

        # 3) retrieve (iterative seed/summary/direct)
        hits = self.iterative_retrieval(qvec, book_id, target_chars=target_chars)

        # 4) negative-rejection check (configurable & non-destructive)
        best_score = max([h.score for h in hits], default=0.0)
        context, citations = self.build_context(hits, max_chars=target_chars)

        is_negative = (best_score < RAG_MIN_BEST_SCORE) or (len(context) < RAG_MIN_CONTEXT_CHARS)
        if is_negative:
            answer = (
                "Mình chưa tìm thấy đủ bằng chứng trong tài liệu để trả lời chắc chắn.\n"
                "Bạn có thể chỉ rõ chương/mục hoặc từ khoá cụ thể hơn không?"
            )
            return {
                "question": question,
                "rewritten": qtext,
                "answer": answer,
                "context": context,  # giữ nguyên context thu được
                "citations": citations,  # bổ sung trích dẫn cho frontend
                "policy": {
                    "negative_rejection": True,
                    "best_score": best_score,
                    "has_seed": self.has_seed,
                    "has_summary": self.has_summary,
                    "min_best_score": RAG_MIN_BEST_SCORE,
                    "min_context_chars": RAG_MIN_CONTEXT_CHARS,
                },
            }

        # 5) generate
        answer = self.generate(question, context)

        return {
            "question": question,
            "rewritten": qtext,
            "answer": answer,
            "context": context,
            "citations": citations,
            "policy": {
                "negative_rejection": False,
                "best_score": best_score,
                "has_seed": self.has_seed,
            },
        }


# -------------------- Convenience wrapper for Streamlit --------------------
def _format_answer_md(out: Dict[str, Any]) -> str:
    ans = out.get("answer", "")
    cits = out.get("citations", []) or []
    if not cits:
        return ans
    # collapse duplicate page/heading pairs
    seen = set()
    lines: List[str] = []
    for c in cits[:8]:
        key = (c.get("page"), c.get("heading") or "")
        if key in seen:
            continue
        seen.add(key)
        pg = c.get("page")
        hd = c.get("heading")
        t = c.get("type")
        tag = " (hình/bảng)" if t == "object_caption" else ""
        lines.append(f"- p.{pg}{' — ' + hd if hd else ''}{tag}")
    src = "\n".join(lines)
    return f"{ans}\n\n**Nguồn trích**\n{src}"


def run_query(question: str, k: int = 20, book_id: Optional[str] = None) -> str:
    """Simple wrapper used by Streamlit app. Returns Markdown string only."""
    eng = QueryEngine()
    out = eng.run_query(question=question, book_id=book_id, k=k, target_chars=3200)
    return _format_answer_md(out)


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--book-id", type=str)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--target-chars", type=int, default=6000)
    args = parser.parse_args()

    eng = QueryEngine()
    out = eng.run_query(
        question=args.question,
        book_id=args.book_id,
        k=args.k,
        target_chars=args.target_chars,
    )
    # pretty print
    print(json.dumps({k: v for k, v in out.items() if k != "context"}, ensure_ascii=False, indent=2))
    print("\n----- ANSWER -----\n")
    print(out.get("answer", ""))
