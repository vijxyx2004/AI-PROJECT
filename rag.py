import json
import os
import uuid
from typing import Any

import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(p.strip() for p in parts if p.strip())


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _headers(auth_token: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if auth_token:
        h["Authorization"] = auth_token
    return h


def endee_health(*, endee_url: str, auth_token: str | None = None) -> bool:
    try:
        r = requests.get(
            f"{endee_url.rstrip('/')}/api/v1/health",
            headers=_headers(auth_token) if auth_token else None,
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def ensure_index(*, endee_url: str, index_name: str, dim: int, auth_token: str | None = None) -> None:
    payload = {"index_name": index_name, "dim": int(dim), "space_type": "cosine"}
    r = requests.post(
        f"{endee_url.rstrip('/')}/api/v1/index/create",
        headers=_headers(auth_token),
        json=payload,
        timeout=60,
    )
    if r.status_code not in (200, 409):
        r.raise_for_status()


def store_in_endee(
    chunks: list[str],
    *,
    endee_url: str = "http://localhost:8080",
    index_name: str = "docs",
    source_name: str = "document.pdf",
    auth_token: str | None = None,
) -> int:
    if not chunks:
        return 0
    if not endee_health(endee_url=endee_url, auth_token=auth_token):
        raise ConnectionError(
            f"Endee server is not reachable at {endee_url.rstrip('/')}. Start Endee and try again."
        )
    embeddings = _embedder.encode(chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True).tolist()
    dim = len(embeddings[0]) if embeddings else 0
    if dim <= 0:
        return 0

    ensure_index(endee_url=endee_url, index_name=index_name, dim=dim, auth_token=auth_token)

    vectors: list[dict[str, Any]] = []
    for i, (text, vec) in enumerate(zip(chunks, embeddings, strict=False)):
        chunk_id = f"{os.path.basename(source_name)}::{i}::{uuid.uuid4().hex[:8]}"
        meta = json.dumps({"source": source_name, "text": text}, ensure_ascii=False)
        vectors.append({"id": chunk_id, "vector": vec, "meta": meta})

    r = requests.post(
        f"{endee_url.rstrip('/')}/api/v1/index/{index_name}/vector/insert",
        headers=_headers(auth_token),
        json=vectors,
        timeout=120,
    )
    r.raise_for_status()
    return len(vectors)


def _search_endee_msgpack(
    *,
    endee_url: str,
    index_name: str,
    query_vector: list[float],
    top_k: int,
    auth_token: str | None,
) -> list[dict[str, Any]]:
    try:
        import msgpack  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency `msgpack`. Install with `pip install -r requirements.txt`."
        ) from e

    payload = {"k": int(top_k), "vector": query_vector}
    h = _headers(auth_token)
    h["Accept"] = "application/msgpack"
    r = requests.post(
        f"{endee_url.rstrip('/')}/api/v1/index/{index_name}/search",
        headers=h,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()

    unpacked = msgpack.unpackb(r.content, raw=False)
    results = unpacked[0] if isinstance(unpacked, list) and unpacked else []
    parsed: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, list) or len(item) < 6:
            continue
        similarity, _id, meta, _filter, _norm, _vec = item[:6]
        meta_text = ""
        if isinstance(meta, (bytes, bytearray)):
            meta_text = meta.decode("utf-8", errors="ignore")
        parsed.append({"similarity": float(similarity), "id": str(_id), "meta_text": meta_text})
    return parsed


def search_endee(
    query: str,
    *,
    endee_url: str = "http://localhost:8080",
    index_name: str = "docs",
    top_k: int = 3,
    auth_token: str | None = None,
) -> list[str]:
    if not endee_health(endee_url=endee_url, auth_token=auth_token):
        raise ConnectionError(
            f"Endee server is not reachable at {endee_url.rstrip('/')}. Start Endee and try again."
        )
    q_vec = _embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits = _search_endee_msgpack(
        endee_url=endee_url,
        index_name=index_name,
        query_vector=q_vec,
        top_k=top_k,
        auth_token=auth_token,
    )
    texts: list[str] = []
    for h in hits:
        meta_text = h.get("meta_text") or ""
        try:
            meta_obj = json.loads(meta_text) if meta_text else {}
        except Exception:
            meta_obj = {}
        t = (meta_obj.get("text") or "").strip()
        if t:
            texts.append(t)
    return texts