import re
import json
import pdfplumber

HEADER_START = re.compile(
    r"(?P<title>.*?)\nMarine Ops: Originator: Approved By:\nSafety Management System.*?\n",
    re.DOTALL
)
FOOTER_PATTERN = re.compile(
    r"SMM:\s*([\d.]+)\s*Revision:\s*(\d+)\s*Effective Date:\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})"
)

def clean_page(page_text: str) -> str:
    match = HEADER_START.search(page_text)
    if not match:
        return page_text
    title = match.group("title").replace("\n", " ").strip()
    remainder = page_text[match.end():]
    next_line, _, rest = remainder.partition("\n")
    stripped_next = next_line.strip()
    if stripped_next and (stripped_next in title or title in stripped_next):
        remainder = rest
    return remainder

def extract_pages(pdf_path: str) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        raw_pages = [page.extract_text(x_tolerance=1) or "" for page in pdf.pages]
    return [clean_page(p) for p in raw_pages]

def build_chunks(pages: list[str]) -> list[dict]:
    full_text = "\n".join(pages)
    footers = list(FOOTER_PATTERN.finditer(full_text))

    chunks = []
    start = 0
    for footer in footers:
        section_id = footer.group(1)
        revision = footer.group(2)
        effective_date = footer.group(3).strip()
        piece = full_text[start:footer.start()].strip()

        if chunks and chunks[-1]["id"] == section_id:
            chunks[-1]["text"] += "\n" + piece
            chunks[-1]["revision"] = revision
            chunks[-1]["effective_date"] = effective_date
        else:
            chunks.append({
                "id": section_id,
                "revision": revision,
                "effective_date": effective_date,
                "text": piece,
            })
        start = footer.end()

    return chunks

if __name__ == "__main__":
    pages = extract_pages("data/raw/Case Manual marineops_sms.pdf")
    chunks = build_chunks(pages)
    print(f"Aantal chunks: {len(chunks)}")

    with open("data/processed/sections.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("Opgeslagen naar data/processed/sections.json")