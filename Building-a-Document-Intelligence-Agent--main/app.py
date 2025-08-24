import streamlit as st
import os
import io
import tempfile
import requests
import time
import numpy as np
import re
from urllib.parse import urljoin
from typing import List, Optional, Dict, Tuple

import pandas as pd

# Mistral 0.4.2 (legacy client)
from mistralai.client import MistralClient

import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader

# Word document processing
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import docx2txt
    HAS_DOCX2TXT = True
except ImportError:
    HAS_DOCX2TXT = False

# --- Dashboard
import plotly.express as px
import plotly.graph_objects as go
import json
import string

# WordCloud opsional
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False


# --------------------- Page config ---------------------
st.set_page_config(page_title="Document Intelligence Agent", layout="wide")
st.title("Document Intelligence Agent")
st.markdown("Upload documents or URL to extract information and ask questions")


# --------------------- Sidebar: API Keys ----------------
with st.sidebar:
    st.header("API Configuration")
    mistral_api_key = st.text_input("Mistral AI API Key (legacy 0.4.2)", type="password")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password")
    model_preference = st.selectbox(
        "Model preference",
        ["Auto (Pro→Flash)", "Flash only", "Pro only"],
        index=0,
        help="Fallback otomatis ke Flash saat Pro kena rate limit/kuota."
    )
    answer_language = st.selectbox(
        "Answer language",
        ["Bahasa Indonesia", "English"],
        index=0,
        help="Bahasa jawaban untuk fitur Q&A."
    )

    st.markdown("---")
    st.caption("Tutorial API Key (YouTube)")
    st.markdown("- Mistral AI API Key — YouTube\n- Google API Key (Gemini) — YouTube")

# Disimpan global agar helper bisa akses
MODEL_PREFERENCE = model_preference
ANSWER_LANGUAGE = answer_language

# MistralClient (legacy)
mistral_client = None
if mistral_api_key:
    try:
        mistral_client = MistralClient(api_key=mistral_api_key)
        st.success("✅ Mistral API connected (legacy client 0.4.2)")
    except Exception as e:
        st.error(f"Failed to initialize Mistral client: {e}")

# Gemini
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        st.success("✅ Google API connected")
    except Exception as e:
        st.error(f"Failed to initialize Google API: {e}")


# --------------------- Helpers (Gemini OCR) -------------------------
def _is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg or "exceeded" in msg

def _get_model_candidates() -> list:
    if MODEL_PREFERENCE == "Flash only":
        return ["gemini-1.5-flash"]
    if MODEL_PREFERENCE == "Pro only":
        return ["gemini-1.5-pro"]
    return ["gemini-1.5-pro", "gemini-1.5-flash"]

def _generate_with_fallback(parts_or_prompt):
    last_error = None
    candidates = _get_model_candidates()
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(parts_or_prompt)
            text = getattr(resp, "text", "").strip()
            if text:
                if model_name != candidates[0]:
                    st.info(f"Using fallback model: {model_name}")
                return text
        except Exception as e:
            last_error = e
            if _is_quota_error(e):
                continue
            else:
                return f"Error generating response: {e}"
    if last_error and _is_quota_error(last_error):
        time.sleep(30)
        try:
            model = genai.GenerativeModel(model_name=candidates[-1])
            resp = model.generate_content(parts_or_prompt)
            return getattr(resp, "text", "").strip()
        except Exception as e2:
            return f"Error generating response: {e2}"
    return f"Error generating response: {last_error}"

def _truncate_context(text: str, max_chars: int = 20000) -> str:
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# --------------------- RAG (opsional mini) ---------------------
EMBEDDING_MODEL = "models/text-embedding-004"

def _chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    end = max(chunk_size, 1)
    text_len = len(text)
    while start < text_len:
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, 0)
        end = min(start + chunk_size, text_len)
    return chunks

def _extract_embedding_values(resp) -> list:
    emb = getattr(resp, "embedding", None)
    if emb is not None:
        values = getattr(emb, "values", emb)
        if isinstance(values, (list, tuple)):
            return list(values)
        if isinstance(values, dict) and "values" in values:
            return list(values["values"])
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])
        if isinstance(emb, (list, tuple)):
            return list(emb)
    return []

def _embed_text(text: str) -> np.ndarray:
    try:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        values = _extract_embedding_values(resp)
        if not values:
            return np.array([])
        return np.array(values, dtype=np.float32)
    except Exception:
        return np.array([])

def _build_retrieval_index(full_text: str):
    text_hash = str(hash(full_text))
    if (st.session_state.get("cached_text_hash") == text_hash and 
        st.session_state.get("retrieval_embeddings") is not None):
        return
    chunks = _chunk_text(full_text)
    embeddings = []
    max_chunks = min(len(chunks), 10)
    chunks = chunks[:max_chunks]
    for ch in chunks:
        try:
            vec = _embed_text(ch)
            if vec.size == 0:
                embeddings = []
                break
            embeddings.append(vec)
        except Exception:
            embeddings = []
            break
    
    if embeddings:
        matrix = np.vstack(embeddings)
        st.session_state.retrieval_chunks = chunks
        st.session_state.retrieval_embeddings = matrix
        st.session_state.retrieval_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        st.session_state.cached_text_hash = text_hash
    else:
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None
        st.session_state.cached_text_hash = None

def _retrieve_top_k(query: str, k: int = 5) -> list:
    if not query:
        return []
    chunks = st.session_state.get("retrieval_chunks")
    emb = st.session_state.get("retrieval_embeddings")
    norms = st.session_state.get("retrieval_norms")
    if not chunks or emb is None or norms is None:
        return []
    q_vec = _embed_text(query)
    if q_vec.size == 0:
        return []
    q_vec = q_vec.reshape(1, -1)
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10
    sims = (emb @ q_vec.T) / (norms * q_norm)
    sims = sims.ravel()
    top_idx = np.argsort(-sims)[: max(1, k)]
    return [chunks[i] for i in top_idx]


# --------------------- Ekstraksi dokumen ---------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        return text
    except Exception:
        return ""

def extract_text_from_word_bytes(word_bytes: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith('.docx') and HAS_DOCX:
            doc = Document(io.BytesIO(word_bytes))
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            return "\n\n".join(text_parts)
        if HAS_DOCX2TXT:
            return docx2txt.process(io.BytesIO(word_bytes))
        return ""
    except Exception as e:
        st.warning(f"Word document text extraction failed: {e}. Falling back to OCR.")
        return ""

def gemini_ocr_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    prompt_doc = (
        "Convert this document image into clean Markdown. "
        "Preserve headings, lists, and tables (use Markdown tables). "
        "Maintain natural reading order."
    )
    text = _generate_with_fallback([prompt_doc, img])
    if not text or len(text.strip()) < 30:
        prompt_cap_id = ("Jelaskan gambar ini secara ringkas, jelas, dan akurat. "
                         "Sebutkan objek utama, konteks, warna, teks (jika ada), dan hal penting lainnya.")
        prompt_cap_en = ("Describe this image concisely and accurately. "
                         "Mention main objects, context, colors, any visible text, and other important details.")
        prompt_cap = prompt_cap_id if ANSWER_LANGUAGE == "Bahasa Indonesia" else prompt_cap_en
        text = _generate_with_fallback([prompt_cap, img])
    return text

def gemini_ocr_pdf(pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
    suffix = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        file_obj = genai.upload_file(path=tmp_path, mime_type="application/pdf", display_name=filename)
        try:
            for _ in range(30):
                f = genai.get_file(file_obj.name)
                state = getattr(getattr(f, "state", None), "name", getattr(f, "state", ""))
                if str(state).upper() == "ACTIVE":
                    break
                time.sleep(1)
            else:
                return "Error: File processing timed out. Please try again."
        except Exception:
            time.sleep(2)
        prompt = ("Extract the full content of this PDF as clean Markdown. "
                  "Preserve headings and tables. If pages are scanned, perform OCR first.")
        return _generate_with_fallback([file_obj, prompt])
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def process_document_with_gemini(kind: str, name: str, data: bytes) -> str:
    if kind == "pdf":
        text = extract_text_from_pdf_bytes(data)
        if len(text) >= 200:
            return text
        return gemini_ocr_pdf(data, filename=name)
    elif kind == "word":
        text = extract_text_from_word_bytes(data, name)
        if len(text) >= 100:
            return text
        return gemini_ocr_pdf(data, filename=name)
    else:  # image
        return gemini_ocr_image(data)

def answer_from_image(image_bytes: bytes, question: str) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if ANSWER_LANGUAGE == "Bahasa Indonesia":
            prompt = ("Anda adalah asisten analisis visual. Jawab pertanyaan pengguna hanya berdasarkan gambar ini.\n"
                      "Jika informasi tidak terlihat pada gambar, katakan tidak ada.\n"
                      f"Pertanyaan: {question}")
        else:
            prompt = ("You are a visual analysis assistant. Answer the user's question based only on this image.\n"
                      "If the information is not visible in the image, say so.\n"
                      f"Question: {question}")
        return _generate_with_fallback([prompt, img]) or "No response text."
    except Exception as e:
        return f"Error generating visual answer: {e}"


# ===================== ANGKA & TABEL (ROBUST & FLEXIBLE) =====================
_num_pat = re.compile(r"([\-]?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{1,2})?)")
nim_hp_pat = re.compile(r"^\d{8,}$")  # 8+ digit beruntun → kemungkinan NIM/HP
LIKELY_SCORE_COLS = {"nilai","skor","score","uts","uas","rata-rata","rata2","rata_rata"}

def looks_like_id_or_phone(raw: str) -> bool:
    raw = str(raw).strip()
    raw = re.sub(r"[^\d]", "", raw)  # buang pemisah
    return bool(nim_hp_pat.match(raw))

def _parse_number(s: str) -> Optional[float]:
    """
    Parser angka lintas-lokal:
    - Decimal koma ('96,29'), decimal titik ('93.7')
    - Ribuan titik '1.234,56' atau ribuan koma '1,234.56'
    - Persen '85,2%' -> 0.852
    - Negatif pakai kurung '(1.234,56)' -> -1234.56
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None

    # ID/HP langsung discard
    if looks_like_id_or_phone(s):
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s_clean = re.sub(r"[^\d\.,%-]", "", s)
    is_percent = s_clean.endswith("%")
    if is_percent:
        s_clean = s_clean[:-1]

    last_dot = s_clean.rfind(".")
    last_com = s_clean.rfind(",")

    def apply_decimal(dec_char, other_char):
        t = s_clean
        t = t.replace(other_char, "")  # buang ribuan
        t = t.replace(dec_char, ".")   # ganti desimal ke '.'
        return t

    if "." in s_clean and "," in s_clean:
        canon = apply_decimal(".", ",") if last_dot > last_com else apply_decimal(",", ".")
    elif "," in s_clean:
        parts = s_clean.split(",")
        canon = apply_decimal(",", ".") if len(parts[-1]) in (1, 2) else s_clean.replace(",", "")
    elif "." in s_clean:
        parts = s_clean.split(".")
        canon = s_clean if len(parts[-1]) in (1, 2) else s_clean.replace(".", "")
    else:
        canon = s_clean

    canon = re.sub(r"[^0-9.\-]", "", canon)
    if canon.count(".") > 1:
        head, _, tail = canon.rpartition(".")
        canon = re.sub(r"\.", "", head) + "." + tail

    try:
        val = float(canon)
        if neg:
            val = -val
        if is_percent:
            val = val / 100.0
        return val
    except Exception:
        return None

# Deteksi blok tabel markdown
_table_block_pat = re.compile(r"(?:^\s*\|.*\|\s*$\n?){2,}", re.M)

def _clean_table_block(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln.startswith("|"): ln = ln[1:]
        if ln.endswith("|"): ln = ln[:-1]
        cleaned.append(ln)
    return "\n".join(cleaned)

def _read_markdown_table_to_df(block: str) -> Optional[pd.DataFrame]:
    cleaned = _clean_table_block(block)
    lines = cleaned.splitlines()
    try:
        tsv = "\n".join(["\t".join([c.strip() for c in ln.split("|")]) for ln in lines])
        df = pd.read_csv(io.StringIO(tsv), sep="\t", header=0, engine="python")
        # hilangkan baris alignment --- jika ada
        if df.shape[0] > 0 and df.iloc[0].astype(str).str.match(r"^:?-{2,}:?$").all():
            df = df.iloc[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.applymap(lambda x: str(x).strip())
        df = df.loc[:, (df != "").any(axis=0)]
        if df.empty:
            return None
        return df
    except Exception:
        return None

def extract_tables_from_markdown(md_text: str) -> List[pd.DataFrame]:
    tables = []
    for m in _table_block_pat.finditer(md_text or ""):
        block = m.group(0)
        df = _read_markdown_table_to_df(block)
        if df is not None and not df.empty:
            tables.append(df)
    return tables

def df_to_numeric_values(df: pd.DataFrame, col_whitelist: Optional[List[str]] = None) -> List[float]:
    nums: List[float] = []
    # jika tidak ada whitelist dari query, gunakan whitelist default kolom skor
    if col_whitelist:
        cols = [c for c in df.columns if re.sub(r"\s+"," ",str(c).strip().lower()) in set(col_whitelist)]
    else:
        cols = [c for c in df.columns if re.sub(r"\s+"," ",str(c).strip().lower()) in LIKELY_SCORE_COLS]
        if not cols:
            cols = list(df.columns)

    for col in cols:
        # skip kolom yang mayoritas terlihat seperti ID/HP
        sample = df[col].astype(str).head(20).tolist()
        id_like = sum(looks_like_id_or_phone(x) for x in sample)
        if id_like >= max(5, int(0.4*len(sample))):  # ≥40% sample terlihat ID
            continue

        series = df[col].apply(_parse_number)
        series = series[pd.notnull(series)]
        if not series.empty:
            nums.extend(series.astype(float).tolist())
    return nums

def extract_numbers_from_text_as_chart(md_text: str) -> List[float]:
    numbers: List[float] = []
    # "label: number" atau "label - number"
    for ln in (md_text or "").splitlines():
        m = re.search(r"[:\-]\s*([\-]?\d[\d\.,]*%?)\s*$", ln)
        if m:
            raw = m.group(1)
            if looks_like_id_or_phone(raw):
                continue
            val = _parse_number(raw)
            if val is not None:
                numbers.append(val)
    # deret angka dipisah spasi/koma/titik koma (>=3 angka)
    for m in re.finditer(r"(?:^|\n)\s*(?:[\-]?\d[\d\.,]*%?\s*[,;\s]\s*){2,}[\-]?\d[\d\.,]*%?", md_text or ""):
        seq = m.group(0)
        for num in re.findall(r"[\-]?\d[\d\.,]*%?", seq):
            if looks_like_id_or_phone(num):
                continue
            val = _parse_number(num)
            if val is not None:
                numbers.append(val)
    return numbers

def build_table_and_chart_caches():
    parsed = []
    chart_nums = []
    audit = {
        "tables_detected": 0,
        "numbers_from_tables": 0,
        "numbers_from_text": 0,
    }
    if not st.session_state.get("documents"):
        st.session_state.parsed_tables = parsed
        st.session_state.chart_numbers = chart_nums
        st.session_state.parse_audit = audit
        return
    for doc in st.session_state.documents:
        md = doc.get("content") or ""
        tbs = extract_tables_from_markdown(md)
        audit["tables_detected"] += len(tbs)
        for i, df in enumerate(tbs):
            parsed.append({"doc": doc["name"], "index": i + 1, "df": df})
            audit["numbers_from_tables"] += sum(pd.notnull(df.applymap(_parse_number)).sum())
        nums = extract_numbers_from_text_as_chart(md)
        if nums:
            for v in nums:
                chart_nums.append({"doc": doc["name"], "value": v})
            audit["numbers_from_text"] += len(nums)
    st.session_state.parsed_tables = parsed
    st.session_state.chart_numbers = chart_nums
    st.session_state.parse_audit = audit

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def filter_values_scope(doc_hint: Optional[str], col_hint: Optional[str]) -> List[float]:
    values: List[float] = []

    col_whitelist = None
    if col_hint:
        col_whitelist = {_norm(col_hint)}

    def match_doc(name: str) -> bool:
        if not doc_hint or _norm(doc_hint) in {"semua", "semua dokumen", "all", "all documents", "kedua", "dua dokumen", "keduanya", "both"}:
            return True
        return _norm(doc_hint) in _norm(name)

    for item in st.session_state.get("parsed_tables", []):
        if match_doc(item["doc"]):
            if col_whitelist is None:
                vals = df_to_numeric_values(item["df"], None)
            else:
                allow = [c for c in item["df"].columns if _norm(c) in col_whitelist]
                vals = df_to_numeric_values(item["df"], [_norm(c) for c in allow]) if allow else df_to_numeric_values(item["df"], None)
            values.extend(vals)

    for obj in st.session_state.get("chart_numbers", []):
        if match_doc(obj["doc"]):
            try:
                v = float(obj["value"])
                values.append(v)
            except Exception:
                pass

    return values

def apply_range_filter(values: List[float]) -> List[float]:
    use_range = st.session_state.get("use_range", True)
    vmin = st.session_state.get("min_val", 0.0)
    vmax = st.session_state.get("max_val", 100.0)
    if use_range and vmin is not None and vmax is not None:
        return [v for v in values if isinstance(v, (int,float)) and vmin <= v <= vmax]
    return [v for v in values if isinstance(v, (int,float))]

def detect_numeric_intent(q: str) -> Optional[Dict[str, Optional[str]]]:
    if not q:
        return None
    ql = q.lower()

    if not any(k in ql for k in ["tabel", "table", "grafik", "chart", "plot", "data"]):
        return None

    op = None
    if any(k in ql for k in ["rata-rata", "rata rata", "average", "mean"]): op = "mean"
    elif any(k in ql for k in ["jumlah", "total", "sum", "akumulasi"]): op = "sum"
    elif "median" in ql: op = "median"
    elif any(k in ql for k in ["min", "minimum", "terkecil"]): op = "min"
    elif any(k in ql for k in ["max", "maks", "maksimum", "terbesar"]): op = "max"
    elif any(k in ql for k in ["std", "stdev", "deviasi"]): op = "std"
    elif any(k in ql for k in ["hitung", "berapa banyak", "count"]): op = "count"
    else:
        if any(k in ql for k in ["nilai", "angka", "skor"]):
            op = "mean"

    if not op:
        return None

    doc_hint = None
    mdoc = re.search(r"dokumen\s+([^\.,;:\n]+)", ql)
    if mdoc:
        doc_hint = mdoc.group(1).strip()
    if any(k in ql for k in ["kedua dokumen", "kedua", "keduanya", "dua dokumen", "both documents", "both"]):
        doc_hint = "kedua"
    if any(k in ql for k in ["semua dokumen", "semua", "all documents"]):
        doc_hint = "semua"

    col_hint = None
    mcol = re.search(r"kolom\s+([^\.,;:\n]+)", ql)
    if mcol:
        col_hint = mcol.group(1).strip()

    return {"op": op, "doc_hint": doc_hint, "col_hint": col_hint}

def aggregate_numbers_from_all_tables_and_charts(op: str, doc_hint: Optional[str], col_hint: Optional[str]) -> Tuple[Optional[float], int]:
    values = filter_values_scope(doc_hint, col_hint)
    values = apply_range_filter(values)
    if not values:
        return None, 0
    s = pd.Series(values, dtype=float)
    op = op.lower()
    if op in {"avg", "average", "mean", "rata", "rata-rata"}:
        return float(s.mean()), len(values)
    if op in {"sum", "jumlah", "total"}:
        return float(s.sum()), len(values)
    if op in {"median"}:
        return float(s.median()), len(values)
    if op in {"min", "minimum", "terkecil"}:
        return float(s.min()), len(values)
    if op in {"max", "maksimum", "terbesar", "maks"}:
        return float(s.max()), len(values)
    if op in {"std", "stdev", "deviasi", "stddev"}:
        return float(s.std(ddof=1)), len(values)
    if op in {"count", "hitung", "berapa banyak"}:
        return int(s.count()), len(values)
    return float(s.mean()), len(values)


# --------------------- API Fallback & Q&A ---------------------
def generate_with_mistral_fallback(prompt: str) -> str:
    if not mistral_client:
        return "Error: No Mistral API key configured for fallback."
    try:
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Mistral fallback: {e}"

def generate_response(context: str, query: str) -> str:
    if not context or len(context) < 10:
        if "image_bytes" in st.session_state and isinstance(st.session_state.image_bytes, dict):
            for img_name, img_bytes in st.session_state.image_bytes.items():
                if any(word.lower() in img_name.lower() for word in query.lower().split()):
                    return answer_from_image(img_bytes, query)
            first_img = next(iter(st.session_state.image_bytes.values()))
            return answer_from_image(first_img, query)
        elif "image_bytes" in st.session_state:
            return answer_from_image(st.session_state.image_bytes, query)
        return "Error: Document context is empty or too short."
    
    try:
        retrieved_chunks = _retrieve_top_k(query, k=5)
        if retrieved_chunks:
            context_block = "\n\n---\n\n".join(retrieved_chunks)
        else:
            context_block = context
        
        doc_context = ""
        if st.session_state.get("documents") and len(st.session_state.documents) > 1:
            doc_names = [doc['name'] for doc in st.session_state.documents]
            doc_context = f"\n\nAvailable documents: {', '.join(doc_names)}"
        
        if ANSWER_LANGUAGE == "Bahasa Indonesia":
            prompt = f"""
Anda adalah asisten analisis dokumen. Gunakan konteks berikut untuk menjawab:

{context_block}{doc_context}

Pertanyaan pengguna:
{query}

Jawab dalam Bahasa Indonesia secara ringkas, jelas, dan akurat. Jika jawabannya tidak terdapat pada konteks, katakan: "Tidak ditemukan pada dokumen". Jika ada beberapa dokumen, sebutkan dari dokumen mana informasi berasal.
"""
        else:
            prompt = f"""
You are a document analysis assistant. Use only the context below to answer:

{context_block}{doc_context}

User question:
{query}

Respond in English concisely. If the answer is not in the context, say so. If there are multiple documents, mention which document the information comes from.
"""
        return _generate_with_fallback(prompt) or "No response text."
    except Exception as e:
        return f"Error generating response: {e}"

def generate_response_with_fallback(context: str, query: str) -> str:
    # Intersep pertanyaan numerik
    intent = detect_numeric_intent(query)
    if intent:
        if "parsed_tables" not in st.session_state:
            build_table_and_chart_caches()
        val, n = aggregate_numbers_from_all_tables_and_charts(
            intent["op"], intent["doc_hint"], intent["col_hint"]
        )
        scope_txt = "semua dokumen"
        if intent and intent.get("doc_hint"):
            dh = intent["doc_hint"]
            if dh in {"semua", "all", "all documents"}:
                scope_txt = "semua dokumen"
            elif dh in {"kedua", "both"}:
                scope_txt = "kedua dokumen"
            else:
                scope_txt = f"dokumen yang cocok '{dh}'"
        col_txt = f", kolom '{intent['col_hint']}'" if intent and intent.get("col_hint") else ""
        if val is not None and n > 0:
            if ANSWER_LANGUAGE == "Bahasa Indonesia":
                return f"Hasil **{intent['op']}** dari angka pada **tabel/grafik** ({scope_txt}{col_txt}) adalah **{val:,.4f}** berdasarkan {n} nilai."
            else:
                return f"The **{intent['op']}** across numbers in **tables/charts** ({scope_txt}{col_txt}) is **{val:,.4f}** based on {n} values."
        else:
            return "Tidak ditemukan angka yang sesuai filter. Coba nonaktifkan filter rentang di sidebar atau perjelas dokumen/kolomnya."

    # bukan numerik → LLM
    try:
        return generate_response(context, query)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Gemini API quota exceeded. Falling back to Mistral API...")
            if ANSWER_LANGUAGE == "Bahasa Indonesia":
                mistral_prompt = f"""
Anda adalah asisten analisis dokumen. Gunakan konteks berikut untuk menjawab:

{context}

Pertanyaan pengguna:
{query}

Jawab dalam Bahasa Indonesia secara ringkas, jelas, dan akurat. Jika jawabannya tidak terdapat pada konteks, katakan: "Tidak ditemukan pada dokumen".
"""
            else:
                mistral_prompt = f"""
You are a document analysis assistant. Use only the context below to answer:

{context}

User question:
{query}

Respond in English concisely. If the answer is not in the context, say so.
"""
            return generate_with_mistral_fallback(mistral_prompt)
        else:
            return f"Error generating response: {e}"


# --------------------- Document Management Helpers ---------------------
def clear_all_document_state():
    st.session_state.documents = []
    st.session_state.ocr_content = None
    st.session_state.retrieval_chunks = None
    st.session_state.retrieval_embeddings = None
    st.session_state.retrieval_norms = None
    st.session_state.image_bytes = {}
    st.session_state.chat_history = []
    st.session_state.parsed_tables = []
    st.session_state.chart_numbers = []
    st.session_state.parse_audit = {}

def rebuild_document_content():
    if st.session_state.documents:
        all_content = [f"--- DOCUMENT: {d['name']} ---\n{d['content']}" for d in st.session_state.documents]
        st.session_state.ocr_content = "\n\n".join(all_content)
        _build_retrieval_index(st.session_state.ocr_content)
        build_table_and_chart_caches()
    else:
        st.session_state.ocr_content = None
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None
        st.session_state.parsed_tables = []
        st.session_state.chart_numbers = []
        st.session_state.parse_audit = {}


# ======================= DASHBOARD HELPERS =======================
_ID_STOPWORDS = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","ini","itu","ada","tidak","atau","karena","sebagai",
    "dalam","atas","oleh","sebuah","para","akan","juga","sudah","belum","saat","kami","kita","mereka","anda",
    "ia","dia","tersebut","rp","usd","pt","tbk","persero","co","ltd","inc"
}
_EN_STOPWORDS = {
    "the","and","of","to","in","for","on","at","by","with","from","as","is","are","was","were","be","been","a","an",
    "this","that","these","those","it","its","we","you","they","he","she","them","our","your","their",
    "or","not","but","if","then","so","than","such","per","vs"
}
STOPWORDS = _ID_STOPWORDS | _EN_STOPWORDS

def _tokenize(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    toks = [t for t in text.split() if t and t not in STOPWORDS and not t.isdigit() and len(t) > 2]
    return toks

def _compute_ocr_quality(text: str) -> float:
    if not text: return 0.0
    n = len(text)
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,:;-%()[]|/+\n" for ch in text)
    bad = text.count("�")
    short_lines = sum(1 for ln in text.splitlines() if 0 < len(ln.strip()) < 3)
    score = (good / n) * 100.0
    score -= min(25, bad * 0.5)
    score -= min(15, short_lines * 0.2)
    return max(0.0, min(100.0, score))

def _structure_stats(md_text: str) -> dict:
    if not md_text:
        return {"text": 0, "tables": 0, "images": 0}
    lines = md_text.splitlines()
    table_lines = sum(1 for ln in lines if ln.strip().startswith("|") and ln.count("|") >= 2)
    image_tags = md_text.count("![") + md_text.lower().count("<img")
    text_chars = len(md_text)
    return {"text": text_chars, "tables": table_lines, "images": image_tags}

def _extract_sections(md_text: str):
    sections = []
    current_title = "Intro"
    current_buf = []
    for ln in md_text.splitlines():
        if ln.startswith("--- DOCUMENT:"):
            if current_buf:
                sections.append((current_title, "\n".join(current_buf).strip()))
                current_buf = []
            current_title = ln.replace("--- DOCUMENT:", "").strip()
        elif re.match(r"^\s{0,3}#{1,3}\s+\S", ln):
            if current_buf:
                sections.append((current_title, "\n".join(current_buf).strip()))
                current_buf = []
            current_title = re.sub(r"^\s{0,3}#{1,3}\s+", "", ln).strip()
        else:
            current_buf.append(ln)
    if current_buf:
        sections.append((current_title, "\n".join(current_buf).strip()))
    return sections[:12] if sections else [("All", md_text)]

def _missing_matrix(sections):
    cats = ["N/A/NA", "Empty Lines", "Dashes(-/—)", "Question(?)"]
    labels = [title[:40] + ("…" if len(title) > 40 else "") for title,_ in sections]
    matrix = []
    for _, txt in sections:
        lines = txt.splitlines()
        na = sum(bool(re.search(r"\b(n/?a|tidak tersedia|kosong)\b", ln, re.I)) for ln in lines)
        empty = sum(1 for ln in lines if not ln.strip())
        dashes = sum(ln.count("-") + ln.count("—") for ln in lines)
        ques = txt.count("?")
        matrix.append([na, empty, dashes, ques])
    return labels, cats, np.array(matrix).T

def _is_financial_report(text: str) -> bool:
    keys = ["revenue","pendapatan","penjualan","income","profit","laba","rugi",
            "neraca","balance sheet","arus kas","cash flow","laba kotor","gross","operating","net"]
    t = text.lower()
    return any(k in t for k in keys)


# --------------------- UI Layout -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload multiple documents (DOC, DOCX, PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg", "doc", "docx"],
        accept_multiple_files=True,
    )
    url_input = st.text_input("Or enter a URL (web page or document):")
    st.session_state["url_input"] = url_input

    process_button = st.button("Process Documents")

    # init states
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "ocr_content" not in st.session_state:
        st.session_state.ocr_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "parsed_tables" not in st.session_state:
        st.session_state.parsed_tables = []
    if "chart_numbers" not in st.session_state:
        st.session_state.chart_numbers = []
    if "parse_audit" not in st.session_state:
        st.session_state.parse_audit = {}

    # --------------------- URL processing helper -----------------------
    def process_url_to_content(url: str) -> tuple:
        try:
            r = requests.get(
                url, timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            clean_url = url.split("?")[0]
            ext = os.path.splitext(clean_url)[1].lower()
            data = r.content
            is_pdf_sig = data[:4] == b"%PDF"
            is_png_sig = data[:8] == b"\x89PNG\r\n\x1a\n"
            is_jpg_sig = data[:3] == b"\xff\xd8\xff"
            chosen_kind = None
            if is_pdf_sig or "pdf" in content_type or ext == ".pdf":
                chosen_kind = "pdf"
            elif is_png_sig or is_jpg_sig or any(img_ct in content_type for img_ct in ["image/png", "image/jpeg", "image/jpg"]) or ext in [".png", ".jpg", ".jpeg"]:
                chosen_kind = "image"
            if not chosen_kind and content_type.startswith("text/html"):
                html_text = r.text
                links = re.findall(r'href=[\"\']([^\"\']+\.(?:pdf|png|jpe?g))(?:[\#\?][^\"\']*)?[\"\']', html_text, flags=re.IGNORECASE)
                if links:
                    target_url = urljoin(url, links[0])
                    rr = requests.get(target_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
                    rr.raise_for_status()
                    target_ct = rr.headers.get("Content-Type", "").lower()
                    tdata = rr.content
                    if tdata[:4] == b"%PDF" or "pdf" in target_ct:
                        chosen_kind = "pdf"; data = tdata; clean_url = target_url.split("?")[0]
                    elif tdata[:8] == b"\x89PNG\r\n\x1a\n" or tdata[:3] == b"\xff\xd8\xff" or any(ic in target_ct for ic in ["image/png", "image/jpeg", "image/jpg"]):
                        chosen_kind = "image"; data = tdata; clean_url = target_url.split("?")[0]
                        st.session_state.image_bytes = data
                if not chosen_kind:
                    stripped = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
                    stripped = re.sub(r"<style[\s\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
                    text_only = re.sub(r"<[^>]+>", " ", stripped)
                    text_only = re.sub(r"\s+", " ", text_only).strip()
                    st.session_state.ocr_content = text_only
                    if st.session_state.ocr_content:
                        _build_retrieval_index(st.session_state.ocr_content)
                        build_table_and_chart_caches()
                        return True, "Webpage processed as text."
                    return False, "No content extracted from webpage."
            if not chosen_kind:
                return False, f"Unsupported content type: {content_type or ext}"
            st.session_state.ocr_content = process_document_with_gemini(chosen_kind, os.path.basename(clean_url) or "download", data)
            if chosen_kind == "image":
                st.session_state.image_bytes = data
            if st.session_state.ocr_content:
                _build_retrieval_index(st.session_state.ocr_content)
                build_table_and_chart_caches()
                return True, "Document processed successfully!"
            return False, "No content extracted."
        except Exception as e:
            return False, f"Error processing document: {e}"

    if process_button:
        if not google_api_key:
            st.error("Please provide a valid Google API Key for OCR/processing.")
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_content = []
                for uploaded_file in uploaded_files:
                    try:
                        ext = os.path.splitext(uploaded_file.name)[1].lower()
                        if ext == ".pdf":
                            kind = "pdf"
                        elif ext in [".doc", ".docx"]:
                            kind = "word"
                        else:
                            kind = "image"
                        if kind == "image":
                            if "image_bytes" not in st.session_state:
                                st.session_state.image_bytes = {}
                            st.session_state.image_bytes[uploaded_file.name] = uploaded_file.getvalue()
                        content = process_document_with_gemini(kind, uploaded_file.name, uploaded_file.getvalue())
                        if content:
                            doc_info = {"name": uploaded_file.name, "type": kind, "content": content, "size": len(uploaded_file.getvalue())}
                            st.session_state.documents.append(doc_info)
                            all_content.append(f"--- DOCUMENT: {uploaded_file.name} ---\n{content}")
                            st.success(f"Document {uploaded_file.name} processed successfully!")
                        else:
                            st.warning(f"No content extracted from {uploaded_file.name}.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                if all_content:
                    st.session_state.ocr_content = "\n\n".join(all_content)
                    _build_retrieval_index(st.session_state.ocr_content)
                    build_table_and_chart_caches()
                    st.success(f"All {len(uploaded_files)} documents processed and combined!")
        if url_input:
            with st.spinner("Downloading & processing from URL..."):
                success, msg = process_url_to_content(url_input)
                if success:
                    if st.session_state.get("ocr_content"):
                        url_doc_info = {"name": f"URL: {url_input[:50]}...", "type": "url", "content": st.session_state.ocr_content, "size": len(st.session_state.ocr_content)}
                        st.session_state.documents.append(url_doc_info)
                    st.success(msg)
                else:
                    st.error(msg)
        if not uploaded_files and not url_input:
            st.warning("Please upload a document or provide a URL.")

with col2:
    st.header("Document Q&A")

    # ========== Sidebar: Quick Numeric Viz & Export (dengan filter) ==========
with st.sidebar:
    st.markdown("---")
    st.subheader("📈 Quick Numeric Viz & Export")

    # Filter rentang nilai (default aktif 0–100)
    st.caption("Filter angka agar hanya nilai ujian yang masuk (mis. 0–100)")
    use_range = st.checkbox("Aktifkan filter rentang nilai", value=True, key="use_range")
    min_val = st.number_input("Min nilai", value=0.0, step=1.0, key="min_val")
    max_val = st.number_input("Max nilai", value=100.0, step=1.0, key="max_val")

    # Hints opsional
    doc_hint_input = st.text_input("Filter dokumen (opsional)", placeholder="mis. Harber / kedua / semua")
    col_hint_input = st.text_input("Filter kolom (opsional)", placeholder="mis. skor / nilai")

    agg_op = st.selectbox(
        "Agregasi",
        ["mean (rata-rata)", "sum (total)", "median", "min", "max", "std (deviasi baku)", "count (jumlah)"],
        index=0
    )
    run_viz = st.button("▶️ Jalankan Analisis")

# ========== Konten kanan (lanjutan) ==========
with col2:
    if st.session_state.documents:
        st.markdown(f"**{len(st.session_state.documents)} document(s) loaded.**")
        
        total_chars = sum(doc['size'] for doc in st.session_state.documents)
        total_mb = total_chars / (1024 * 1024)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: st.metric("Total Documents", len(st.session_state.documents))
        with c2: st.metric("Total Content", f"{total_chars:,} chars")
        with c3: st.metric("Memory Usage", f"{total_mb:.1f} MB")
        
        if len(st.session_state.documents) > 10:
            st.warning("⚠️ Many documents loaded. Consider removing some to improve performance.")
        elif total_chars > 500000:
            st.warning("⚠️ Very large document collection. Consider removing some documents to avoid processing limits.")
        elif total_chars > 200000 and len(st.session_state.documents) > 3:
            st.warning("⚠️ Large document collection. Consider removing some documents to avoid processing limits.")
        
        with st.expander("📋 Document List"):
            for i, doc in enumerate(st.session_state.documents):
                cc1, cc2 = st.columns([4, 1])
                with cc1:
                    st.markdown(f"**{i+1}. {doc['name']}** ({doc['type']}) - {doc['size']} chars")
                with cc2:
                    if st.button(f"🗑️", key=f"del_{i}", help=f"Delete {doc['name']}"):
                        deleted_doc = st.session_state.documents.pop(i)
                        if deleted_doc['type'] == 'image' and 'image_bytes' in st.session_state:
                            if isinstance(st.session_state.image_bytes, dict):
                                st.session_state.image_bytes.pop(deleted_doc['name'], None)
                            else:
                                st.session_state.image_bytes = {}
                        rebuild_document_content()
                        st.rerun()
            if st.session_state.documents:
                if st.button("🔄 Reset All Documents", type="secondary"):
                    clear_all_document_state()
                    st.rerun()

        # ---------- Quick actions ----------
        if st.session_state.documents:
            st.markdown("**Quick Actions:**")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                if st.button("🗑️ Clear Chat", help="Clear chat history but keep documents"):
                    st.session_state.chat_history = []
                    st.rerun()
            with q2:
                if st.button("📊 Document Stats", help="Show dashboard with charts"):
                    st.session_state.show_stats = not st.session_state.get('show_stats', False)
                    st.rerun()
            with q3:
                if st.button("🔎 Parse/Refresh Tables", help="Parse/refresh tables from current documents"):
                    build_table_and_chart_caches()
                    a = st.session_state.get("parse_audit", {})
                    st.success(f"Parsed {len(st.session_state.parsed_tables)} table(s); found {len(st.session_state.chart_numbers)} 'chart-number(s)'. Tables detected: {a.get('tables_detected',0)}.")
            with q4:
                if st.button("🧪 Audit Angka", help="Lihat breakdown angka yang terdeteksi & difilter"):
                    build_table_and_chart_caches()
                    a = st.session_state.get("parse_audit", {})
                    st.info(f"Audit parsing → tables: {a.get('tables_detected',0)}, numbers_from_tables: {a.get('numbers_from_tables',0)}, numbers_from_text: {a.get('numbers_from_text',0)}")

        # ---------- DASHBOARD ----------
        if st.session_state.get('show_stats', False):
            st.markdown("## 📊 Document Analytics Dashboard")

            stats_data = []
            combined_texts = []
            struct_total = {"text":0, "tables":0, "images":0}
            for doc in st.session_state.documents:
                txt = doc['content'] or ""
                combined_texts.append(txt)
                ocr_q = _compute_ocr_quality(txt)
                words = len(_tokenize(txt))
                lines = len(txt.splitlines())
                stt = {"Document": doc['name'], "Type": doc['type'], "Size (chars)": doc['size'], "Words": words, "Lines": lines, "OCR Quality": round(ocr_q,1)}
                stats_data.append(stt)
                s = _structure_stats(txt)
                struct_total["text"] += s["text"]
                struct_total["tables"] += s["tables"]
                struct_total["images"] += s["images"]

            combined = "\n\n".join(combined_texts)
            st.dataframe(stats_data, use_container_width=True)

            st.markdown("### Document Health")
            dh1, dh2 = st.columns([1,1])
            with dh1:
                ocr_score = _compute_ocr_quality(combined)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ocr_score,
                    number={'suffix': " /100"},
                    title={'text': "OCR Quality Score"},
                    gauge={'axis': {'range': [0,100]},
                           'bar': {'thickness': 0.3},
                           'steps': [
                               {'range': [0,50], 'color': "#fce4ec"},
                               {'range': [50,75], 'color': "#fff3e0"},
                               {'range': [75,100], 'color': "#e8f5e9"},
                           ]}
                ))
                st.plotly_chart(fig_g, use_container_width=True)

            with dh2:
                labels = ["Text", "Tables", "Images"]
                values = [max(1, struct_total["text"]), max(1, struct_total["tables"]), max(1, struct_total["images"])]
                fig_donut = px.pie(values=values, names=labels, hole=0.6, title="Document Structure Overview")
                st.plotly_chart(fig_donut, use_container_width=True)

            secs = _extract_sections(combined)
            sec_labels, cat_labels, mat = _missing_matrix(secs)
            fig_heat = px.imshow(mat, aspect="auto", color_continuous_scale="Viridis",
                                 labels=dict(x="Section", y="Missing Type", color="Count"),
                                 x=sec_labels, y=cat_labels)
            fig_heat.update_layout(title="Missing Data Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("### Parsed Tables Summary")
            st.write(f"Detected **{len(st.session_state.parsed_tables)}** table(s) and **{len(st.session_state.chart_numbers)}** chart-number(s).")
            for item in st.session_state.parsed_tables[:3]:
                st.caption(f"Sample table from **{item['doc']}** (Table #{item['index']})")
                st.dataframe(item["df"], use_container_width=True)

        # ---------- Quick Numeric Viz & Export ----------
        def collect_values_grouped_by_doc(doc_hint: Optional[str], col_hint: Optional[str]) -> Dict[str, list]:
            grouped: Dict[str, list] = {}
            if "parsed_tables" not in st.session_state:
                build_table_and_chart_caches()

            def match_doc(name: str) -> bool:
                if not doc_hint:
                    return True
                dh = _norm(doc_hint)
                if dh in {"semua", "semua dokumen", "all", "all documents", "kedua", "dua dokumen", "keduanya", "both"}:
                    return True
                return dh in _norm(name)

            # Tabel
            for item in st.session_state.get("parsed_tables", []):
                dname = item["doc"]
                if not match_doc(dname):
                    continue
                df = item["df"]
                if col_hint:
                    allow_cols = [c for c in df.columns if _norm(c) == _norm(col_hint)]
                    vals = df_to_numeric_values(df, [_norm(c) for c in allow_cols]) if allow_cols else df_to_numeric_values(df, None)
                else:
                    vals = df_to_numeric_values(df, None)
                if vals:
                    grouped.setdefault(dname, []).extend(vals)

            # Grafik/teks
            for obj in st.session_state.get("chart_numbers", []):
                dname = obj["doc"]
                if not match_doc(dname):
                    continue
                try:
                    v = float(obj["value"])
                    grouped.setdefault(dname, []).append(v)
                except Exception:
                    pass

            # Terapkan filter rentang
            for k in list(grouped.keys()):
                grouped[k] = apply_range_filter(grouped[k])
                if not grouped[k]:
                    grouped.pop(k, None)

            return grouped

        if run_viz:
            if "parsed_tables" not in st.session_state:
                build_table_and_chart_caches()

            op_map = {
                "mean (rata-rata)": "mean",
                "sum (total)": "sum",
                "median": "median",
                "min": "min",
                "max": "max",
                "std (deviasi baku)": "std",
                "count (jumlah)": "count",
            }
            op_key = op_map.get(agg_op, "mean")

            grouped = collect_values_grouped_by_doc(doc_hint_input, col_hint_input)
            all_vals = [val for arr in grouped.values() for val in arr]

            st.markdown("## 📈 Hasil Analisis Angka")
            if not grouped:
                st.warning("Tidak ada angka yang cocok dengan filter. Coba nonaktifkan filter rentang di sidebar atau perbarui parsing tabel (Quick Actions → Parse/Refresh Tables).")
            else:
                # Ringkasan per dokumen
                rows = []
                for dname, vals in grouped.items():
                    s = pd.Series(vals, dtype=float)
                    row = {
                        "Dokumen": dname,
                        "Count": int(s.count()),
                        "Mean": float(s.mean()) if s.count() else None,
                        "Median": float(s.median()) if s.count() else None,
                        "Min": float(s.min()) if s.count() else None,
                        "Max": float(s.max()) if s.count() else None,
                        "Std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                        "Sum": float(s.sum()) if s.count() else None,
                    }
                    rows.append(row)
                df_summary = pd.DataFrame(rows).sort_values("Dokumen").reset_index(drop=True)

                # Agregat global
                val_global, n_global = aggregate_numbers_from_all_tables_and_charts(
                    op_key, doc_hint_input if doc_hint_input else None, col_hint_input if col_hint_input else None
                )
                if val_global is not None:
                    if ANSWER_LANGUAGE == "Bahasa Indonesia":
                        st.success(f"**{agg_op}** ({'semua dokumen' if not doc_hint_input else f'filter dokumen: {doc_hint_input}' }{', kolom: ' + col_hint_input if col_hint_input else ''}) = **{val_global:,.4f}** dari **{n_global}** nilai.")
                    else:
                        st.success(f"**{agg_op}** ({'all documents' if not doc_hint_input else f'doc filter: {doc_hint_input}' }{', column: ' + col_hint_input if col_hint_input else ''}) = **{val_global:,.4f}** based on **{n_global}** values.")

                st.markdown("### Ringkasan per Dokumen")
                st.dataframe(df_summary, use_container_width=True)

                # Visualisasi
                try:
                    long_rows = []
                    for dname, vals in grouped.items():
                        for v in vals:
                            long_rows.append({"Dokumen": dname, "Nilai": float(v)})
                    dfl = pd.DataFrame(long_rows)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Histogram Nilai")
                        fig_hist = px.histogram(dfl, x="Nilai", nbins=30, color="Dokumen")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with c2:
                        st.markdown("#### Box Plot per Dokumen")
                        fig_box = px.box(dfl, x="Dokumen", y="Nilai", points="outliers")
                        st.plotly_chart(fig_box, use_container_width=True)

                    st.markdown("#### Rata-rata per Dokumen (Bar)")
                    df_bar = dfl.groupby("Dokumen", as_index=False)["Nilai"].mean().rename(columns={"Nilai": "Mean"})
                    fig_bar = px.bar(df_bar, x="Dokumen", y="Mean", text="Mean")
                    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig_bar.update_layout(yaxis_title="Mean", xaxis_title="Dokumen")
                    st.plotly_chart(fig_bar, use_container_width=True)

                except Exception as e:
                    st.warning(f"Gagal membuat grafik: {e}")

                # Ekspor CSV
                try:
                    csv_df = pd.DataFrame([(doc, v) for doc, arr in grouped.items() for v in arr], columns=["Dokumen", "Nilai"])
                    if col_hint_input:
                        csv_df["KolomFilter"] = col_hint_input
                    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download Nilai (CSV)", data=csv_bytes, file_name="extracted_numeric_values.csv", mime="text/csv")
                except Exception as e:
                    st.warning(f"Gagal menyiapkan CSV: {e}")

    # ---------- Chat history ----------
    if st.session_state.get("chat_history"):
        st.markdown("### Chat")
        for m in st.session_state.chat_history:
            role = "You" if m["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {m['content']}")

    # ---------- Q&A input ----------
    if st.session_state.documents:
        with st.form("qa_form_docs", clear_on_submit=True):
            user_q = st.text_input("Your question (can ask about specific documents or compare them):", key="qa_input")
            submitted = st.form_submit_button("Ask")
        if submitted and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Generating response..."):
                if not google_api_key:
                    ans = "Please provide a valid Google API Key."
                else:
                    if "parsed_tables" not in st.session_state:
                        build_table_and_chart_caches()
                    ans = generate_response_with_fallback(st.session_state.ocr_content, user_q)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
    else:
        st.info("No documents processed yet. You can either upload files or just type a URL below and press Ask.")
        with st.form("qa_form_nodocs", clear_on_submit=True):
            user_q = st.text_input("Your question (can ask about specific documents or compare them):", key="qa_input")
            submitted = st.form_submit_button("Ask")

        if submitted and user_q:
            url_candidate = st.session_state.get("url_input")
            if url_candidate and not st.session_state.get("ocr_content"):
                st.warning("Processing URL before answering...")
                # NOTE: memanggil helper di kolom lain tidak nyaman; minta user klik Process Documents.
                st.stop()
            if st.session_state.get("ocr_content"):
                build_table_and_chart_caches()
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.spinner("Generating response..."):
                    if not google_api_key:
                        ans = "Please provide a valid Google API Key."
                    else:
                        ans = generate_response_with_fallback(st.session_state.ocr_content, user_q)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()
            else:
                st.warning("Please provide a URL or upload a document first.")

# Tampilkan konten hasil OCR/ekstraksi
if st.session_state.get("documents"):
    with st.expander("📄 View All Document Contents"):
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"### {doc['name']} ({doc['type']})")
            st.markdown(doc['content'])
            st.markdown("---")
