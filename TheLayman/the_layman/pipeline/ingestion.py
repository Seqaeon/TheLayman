from __future__ import annotations

import hashlib
import io
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT_S = 20
USER_AGENT = "TheLayman/0.1 (+https://example.local)"


@dataclass
class PaperContent:
    paper_id: str
    source: str
    title: str
    authors: list[str]
    url: str
    abstract: str
    introduction: str
    conclusion: str
    full_text: str


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _paper_id(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
        return resp.read()


def _extract_sections(text: str) -> tuple[str, str, str]:
    normalized = _normalize_text(text)
    if not normalized:
        return "unknown", "unknown", "unknown"

    abstract = normalized[:1200] if len(normalized) > 0 else "unknown"

    lower = normalized.lower()
    intro_idx = lower.find("introduction")
    concl_idx = max(lower.rfind("conclusion"), lower.rfind("concluding"))

    introduction = normalized[intro_idx:intro_idx + 2500] if intro_idx != -1 else normalized[:2500]
    conclusion = normalized[concl_idx:concl_idx + 2500] if concl_idx != -1 else normalized[-2500:]
    return abstract or "unknown", introduction or "unknown", conclusion or "unknown"


def _extract_pdf_text_from_bytes(pdf_data: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_data))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks.append(page_text)
    return _normalize_text("\n".join(chunks))


def _parse_arxiv_abs_html(html: str) -> tuple[str, str]:
    title_match = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', html, flags=re.I)
    abs_match = re.search(r'<meta\s+name="citation_abstract"\s+content="([^"]+)"', html, flags=re.I)

    if not abs_match:
        block = re.search(r'<blockquote[^>]*class="abstract[^>]*>(.*?)</blockquote>', html, flags=re.I | re.S)
        if block:
            cleaned = re.sub(r"<[^>]+>", " ", block.group(1))
            cleaned = cleaned.replace("Abstract:", "")
            abstract = _normalize_text(cleaned)
        else:
            abstract = "unknown"
    else:
        abstract = _normalize_text(abs_match.group(1)) or "unknown"

    title = _normalize_text(title_match.group(1)) if title_match else "unknown"
    return title or "unknown", abstract or "unknown"


def ingest_doi(doi: str) -> PaperContent:
    doi = doi.strip().lower()
    if not doi or "/" not in doi:
        raise ValueError("invalid DOI")

    title = "unknown"
    authors: list[str] = ["unknown"]
    abstract = "unknown"
    url = f"https://doi.org/{doi}"

    api = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    try:
        payload = _http_get(api).decode("utf-8", errors="ignore")
        import json

        obj = json.loads(payload).get("message", {})
        title_list = obj.get("title") or []
        if title_list:
            title = _normalize_text(title_list[0]) or "unknown"
        author_list = obj.get("author") or []
        parsed_authors = []
        for a in author_list[:8]:
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            name = f"{given} {family}".strip()
            if name:
                parsed_authors.append(name)
        if parsed_authors:
            authors = parsed_authors
        raw_abstract = obj.get("abstract")
        if raw_abstract:
            abstract = _normalize_text(re.sub(r"<[^>]+>", " ", raw_abstract)) or "unknown"
        if obj.get("URL"):
            url = obj.get("URL")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError):
        pass

    full_text = abstract
    intro, conclusion = "unknown", "unknown"
    if abstract != "unknown":
        _, intro, conclusion = _extract_sections(abstract)

    return PaperContent(
        paper_id=_paper_id(doi, "doi"),
        source="doi",
        title=title,
        authors=authors,
        url=url,
        abstract=abstract,
        introduction=intro,
        conclusion=conclusion,
        full_text=full_text if full_text else "unknown",
    )


def ingest_arxiv(arxiv_url: str) -> PaperContent:
    if "arxiv.org" not in arxiv_url:
        raise ValueError("invalid arXiv URL")

    parsed = urllib.parse.urlparse(arxiv_url)
    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        arxiv_id = path.replace("abs/", "", 1)
    elif path.startswith("pdf/"):
        arxiv_id = path.replace("pdf/", "", 1).removesuffix(".pdf")
    else:
        arxiv_id = path
    if not arxiv_id:
        raise ValueError("invalid arXiv URL")

    title = "unknown"
    authors: list[str] = ["unknown"]
    abstract = "unknown"

    api = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        xml_data = _http_get(api).decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_data)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is not None:
            t = entry.findtext("a:title", default="", namespaces=ns)
            s = entry.findtext("a:summary", default="", namespaces=ns)
            title = _normalize_text(t) or "unknown"
            abstract = _normalize_text(s) or "unknown"
            found_authors = [
                _normalize_text(a.findtext("a:name", default="", namespaces=ns))
                for a in entry.findall("a:author", ns)
            ]
            found_authors = [a for a in found_authors if a]
            if found_authors:
                authors = found_authors[:8]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ET.ParseError):
        pass

    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    if abstract == "unknown":
        try:
            html = _http_get(abs_url).decode("utf-8", errors="ignore")
            html_title, html_abstract = _parse_arxiv_abs_html(html)
            if title == "unknown":
                title = html_title
            abstract = html_abstract
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

    full_text = abstract
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        pdf_bytes = _http_get(pdf_url)
        parsed_pdf_text = _extract_pdf_text_from_bytes(pdf_bytes)
        if parsed_pdf_text:
            full_text = parsed_pdf_text
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        pass

    _, introduction, conclusion = _extract_sections(full_text)
    return PaperContent(
        paper_id=_paper_id(arxiv_id, "arxiv"),
        source="arxiv",
        title=title,
        authors=authors,
        url=abs_url,
        abstract=abstract,
        introduction=introduction,
        conclusion=conclusion,
        full_text=full_text if full_text else "unknown",
    )


def ingest_pdf(pdf_path: Path) -> PaperContent:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("invalid PDF upload")
    if pdf_path.stat().st_size > MAX_PDF_SIZE_BYTES:
        raise ValueError("PDF exceeds max allowed size")

    pdf_bytes = pdf_path.read_bytes()
    text = _extract_pdf_text_from_bytes(pdf_bytes)
    if not text:
        text = "unknown"

    abstract, introduction, conclusion = _extract_sections(text)
    return PaperContent(
        paper_id=_paper_id(str(pdf_path.resolve()), "pdf"),
        source="pdf",
        title=pdf_path.stem,
        authors=["unknown"],
        url="unknown",
        abstract=abstract,
        introduction=introduction,
        conclusion=conclusion,
        full_text=text,
    )
