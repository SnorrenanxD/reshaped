# src/ingest.py
# Loads the raw PDF, extracts the text, and saves it as a JSON file with sections.

import re
import json
from difflib import SequenceMatcher

import pdfplumber

FOOTER = re.compile(r"SMM:\s*([\d.]+)\s*Revision:\s*\d+\s*Effective Date:\s*(?:\d{1,2}\s+[A-Za-z]{3}\s+\d{4})?")
HEADER = re.compile(
    r"(?P<title>.*?)\nMarine Ops: Originator: Approved By:\nSafety Management (?:System|Manual).*?\n",
    re.DOTALL
)
CHAPTER = re.compile(r"Section (\d+) (.+?) \d{2} NOV \d{4}", re.DOTALL)
APPENDIX = re.compile(r"^Appendix:\s*(SMF\s*[\d.]+(?:\([a-z]\))?)\s*(.*)$")


def parse_chapter_titles(pdf) -> dict[str, str]:
    toc = "\n".join((pdf.pages[i].extract_text() or "") for i in range(1, 4))
    titles = {m.group(1): m.group(2).replace("\n", " ").strip() for m in CHAPTER.finditer(toc)}
    # chapters 5 and 12 wrap through the TOC date column
    titles["5"] = "Master's Responsibilities and Authority"
    titles["12"] = "Company Verification, Review and Evaluation"
    return titles


def clean_page(page_text: str) -> tuple[str, str]:
    match = HEADER.search(page_text)
    if not match:
        return page_text, ""
    title = match.group("title").replace("\n", " ").strip()
    remainder = page_text[match.end():]

    # drop up to 2 leading lines repeating the title or chapter heading
    for _ in range(2):
        line, _, rest = remainder.partition("\n")
        s = line.strip()
        if s and (s in title or title in s or re.match(r"^(Section )?\d+(\.\d+)*:\s", s)):
            remainder = rest
        else:
            break

    return remainder.replace("[Table of Contents]", ""), title


def build_chunks(pdf_path: str) -> list[dict]:
    with pdfplumber.open(pdf_path) as pdf:
        chapters = parse_chapter_titles(pdf)
        pages = [p.extract_text(x_tolerance=1) or "" for p in pdf.pages]

    chunks, by_id = [], {}
    for raw in pages:
        footer = FOOTER.search(raw)
        appendix = APPENDIX.match(raw.split("\n", 1)[0])
        if not footer and not appendix:
            continue

        body, header_title = clean_page(raw)
        body = FOOTER.sub("", body).strip()

        if footer:
            cid = footer.group(1)
            # some pages carry the previous section's header (source doc error)
            if not header_title.startswith(cid):
                title = ""
            else:
                chapter = chapters.get(cid.split(".")[0], "")
                title = re.sub(r"^[\d.]+\s*:?\s*", "", header_title)
                similar = SequenceMatcher(None, chapter.lower(), title.lower()).ratio() > 0.8
                if chapter and not similar:
                    title = f"{chapter}: {title}"
        else:
            cid = re.sub(r"\s+", " ", appendix.group(1))
            title = f"Forms: {appendix.group(2).strip()}"

        if cid in by_id:
            existing = chunks[by_id[cid]]
            existing["text"] += "\n" + body
            if not existing["title"] and title:
                existing["title"] = title
        else:
            by_id[cid] = len(chunks)
            chunks.append({"id": cid, "title": title, "text": body})

    return chunks


if __name__ == "__main__":
    chunks = build_chunks("data/raw/Case Manual marineops_sms.pdf")
    print(f"Chunks: {len(chunks)}")
    with open("data/processed/sections.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("Saved to data/processed/sections.json")