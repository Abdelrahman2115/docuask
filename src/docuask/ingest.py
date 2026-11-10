"""Load FCC PDFs and HTML pages, clean the text, and split it into cited chunks.

Pipeline:  raw file -> Units (one PDF page, or one HTML section) -> Chunks

A Unit is the smallest piece we can cite. Chunks are made *inside* a unit and
never cross a page or section boundary, so every chunk has one exact citation.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    doc_id: str
    title: str
    url: str
    source_file: str          # e.g. "slamming.pdf"
    page: int | None          # PDF page number (1-based); None for HTML
    section: str | None       # HTML section heading; None for PDF
    text: str


@dataclass
class Chunk:
    chunk_id: str             # "<doc_id>:p2:c0" or "<doc_id>:s3:c1"
    doc_id: str
    title: str
    url: str
    source_file: str
    page: int | None
    section: str | None
    chunk_index: int          # position of the chunk inside its unit
    text: str

    @property
    def citation(self) -> str:
        where = f"p. {self.page}" if self.page is not None else f"§ {self.section}"
        return f"[{self.source_file}, {where}]"


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_BULLET = re.compile(r"^\s*[••●▪\-–]\s+")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*of\s*\d{1,3})?\s*$", re.I)
_ALT_FORMATS = re.compile(r"Alternate formats?\s+To request this article.*?fcc504@fcc\.gov\.?", re.I | re.S)
_LAST_REVIEWED = re.compile(r"^\s*(Last Reviewed|Date Last Updated)[^\n]*$", re.I | re.M)


def _is_header_footer(line: str) -> bool:
    """FCC letterhead lines: agency name, address and phone separated by middle dots (·).

    Matching on the middle dot alone is deliberate: body text never uses it, while
    a rule like "contains 1-888-CALL-FCC" would also delete real 'By phone:' bullets.
    """
    return "·" in line


def clean_pdf_text(raw: str) -> str:
    """Turn pypdf output into readable paragraphs.

    - drop letterhead/footer lines and bare page numbers
    - re-join lines that were wrapped mid-sentence
    - keep bullets and short heading-like lines on their own line
    - remove the boilerplate 'Alternate formats' paragraph
    """
    lines = [ln.strip() for ln in raw.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if not _is_header_footer(ln) and not _PAGE_NUMBER.match(ln)]

    out: list[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")          # paragraph break
            continue
        if out and out[-1].endswith("-") and not out[-1].endswith(" -") and not _BULLET.match(ln):
            out[-1] = out[-1] + ln      # "TTY-" + "based" -> "TTY-based", "835-" + "5322" -> "835-5322"
            continue
        starts_new_line = (
            not out
            or out[-1] == ""
            or _BULLET.match(ln)
            or _looks_like_heading(out[-1])
            or _looks_like_heading(ln)
        )
        if starts_new_line:
            out.append(ln)
        else:
            out[-1] = out[-1] + " " + ln

    text = "\n".join(out)
    text = _ALT_FORMATS.sub("", text)
    text = _LAST_REVIEWED.sub("", text)
    return normalize_whitespace(text)


def _looks_like_heading(line: str) -> bool:
    """Short line, starts with a capital, no closing punctuation: e.g. 'Public Wi-Fi Access'."""
    return (
        0 < len(line) < 60
        and line[0].isupper()
        and not line.endswith((".", ",", ";", ":", "?", "!", "-"))
        and not _BULLET.match(line)
    )


def normalize_whitespace(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)   # "toll -free" -> "toll-free"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_pdf(path: Path, doc_id: str, title: str, url: str) -> list[Unit]:
    reader = PdfReader(str(path))
    units = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(page.extract_text() or "")
        if len(text) >= 100:             # skip blank, image-only or address-only pages
            units.append(Unit(doc_id, title, url, path.name, i, None, text))
    return units


_STOP_HEADINGS = {"printable version", "alternate format requests"}


def html_to_sections(html_text: str, min_chars: int = 200) -> list[tuple[str, str]]:
    """Split the main <article> into (heading, text) sections on h1-h3 headings.

    h4+ headings stay inside the section text. Content after a 'Printable
    Version' heading (site footer) is dropped. Sections shorter than
    `min_chars` are appended to the previous section (or, for the very first
    section, to the next one) so tiny sections don't become chunks of their own.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    root = soup.select_one("article.guide") or soup.select_one("article") or soup.body or soup
    for tag in root(["style", "script", "noscript", "nav", "form", "button"]):
        tag.decompose()

    page_title = soup.title.get_text(strip=True) if soup.title else "Introduction"
    sections: list[list] = [[page_title.split("|")[0].strip(), []]]
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "div"]):
        if el.find(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "div"]):
            continue                          # only leaf blocks, avoids duplicate text
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h4", "h5") and text.lower() in _STOP_HEADINGS:
            break
        if el.name in ("h1", "h2", "h3"):
            if text.lower() in _STOP_HEADINGS:
                break
            if el.name == "h1" and not any(s[1] for s in sections):
                sections[0][0] = text          # article title replaces page title
                continue
            sections.append([text, []])
            continue
        prefix = "- " if el.name == "li" else ""
        sections[-1][1].append(prefix + text)

    merged: list[tuple[str, str]] = []
    for heading, parts in sections:
        body = normalize_whitespace("\n".join(parts))
        if not body:
            continue
        if len(body) < min_chars and merged:
            # too short: append to the previous section, keeping its heading as a line of text
            prev_heading, prev_body = merged[-1]
            merged[-1] = (prev_heading, f"{prev_body}\n{heading}\n{body}")
        elif merged and len(merged[-1][1]) < min_chars:
            # previous (first) section was too short: fold it into this one
            prev_heading, prev_body = merged.pop()
            merged.append((prev_heading, f"{prev_body}\n{heading}\n{body}"))
        else:
            merged.append((heading, body))
    return merged


def load_html(path: Path, doc_id: str, title: str, url: str) -> list[Unit]:
    html_text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Unit(doc_id, title, url, path.name, None, heading, f"{heading}\n{body}")
        for heading, body in html_to_sections(html_text)
    ]


def load_corpus(raw_dir: Path, sources_csv: Path) -> tuple[list[Unit], list[str]]:
    """Load every source listed in sources.csv. Returns (units, missing_doc_ids)."""
    units, missing = [], []
    with open(sources_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            doc_id, kind = row["doc_id"], row["kind"]
            url = row["url"]
            if kind == "zendesk":   # cite the human-readable page, not the API URL
                url = url.replace("/api/v2/help_center/en-us/articles/", "/hc/en-us/articles/").replace(".json", "")
            path = raw_dir / ("pdf" if kind == "pdf" else "html") / f"{doc_id}.{'pdf' if kind == 'pdf' else 'html'}"
            if not path.exists():
                missing.append(doc_id)
                continue
            loader = load_pdf if kind == "pdf" else load_html
            units.extend(loader(path, doc_id, row["title"], url))
    return units, missing


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # Try to split on paragraph, then line, then sentence, then word boundaries.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "; ", ", ", " ", ""],
        keep_separator="end",
    )


MIN_CHUNK_CHARS = 100


def _merge_tiny(pieces: list[str], min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Attach fragments shorter than `min_chars` (a lone title, an address line,
    a lead-in like 'Consumers will continue to receive:') to the next piece, or
    to the previous one if it is last. Alone they match queries but answer nothing.
    """
    out: list[str] = []
    carry = ""
    for p in (x.strip() for x in pieces):
        if not p:
            continue
        if carry:
            p, carry = f"{carry}\n{p}", ""
        if len(p) < min_chars:
            carry = p
            continue
        out.append(p)
    if carry:
        if out:
            out[-1] = f"{out[-1]}\n{carry}"
        else:
            out.append(carry)
    return out


def chunk_units(units: list[Unit], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split each unit separately. A merged tiny fragment can push a chunk up to
    chunk_size + MIN_CHUNK_CHARS characters, which still fits the embedding limit."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    section_counter: dict[str, int] = {}
    for u in units:
        if u.page is not None:
            loc = f"p{u.page}"
        else:
            section_counter[u.doc_id] = section_counter.get(u.doc_id, 0) + 1
            loc = f"s{section_counter[u.doc_id]}"
        for j, piece in enumerate(_merge_tiny(splitter.split_text(u.text))):
            chunks.append(Chunk(
                chunk_id=f"{u.doc_id}:{loc}:c{j}",
                doc_id=u.doc_id, title=u.title, url=u.url, source_file=u.source_file,
                page=u.page, section=u.section, chunk_index=j, text=piece,
            ))
    return chunks


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
