# src/format.py
# Recovers headers/paragraphs/lists from ingested text so a raw PDF dump doesn't
# render as one wall of text — the words are unchanged, only the layout.
import re

# Recurring subsection labels used across the manual's template (found by frequency
# analysis of the ingested text); a line matching one of these starts a new header.
SECTION_LABELS = {
    "purpose", "general", "responsibilities", "responsibility", "procedures",
    "procedure", "reporting", "scope", "references", "reference",
    "definitions", "definition", "records", "recording",
}
LIST_ITEM_RE = re.compile(r"^(\d+)\.\s+")


def format_section_text(text: str) -> str:
    """Turns raw ingested section text into markdown with bold headers and lists."""
    blocks: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    buf_type = "para"
    last_num = None  # previous list number, so a restart (e.g. back to "1.") starts a new list

    def flush():
        nonlocal buf
        if buf:
            blocks.append((buf_type, buf))
        buf = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # A known subsection label on its own line starts a new header block.
        if line.lower() in SECTION_LABELS and len(line.split()) <= 2:
            flush()
            blocks.append(("header", [line]))
            buf_type = "para"
            last_num = None
            continue

        # A new number — or a break in the sequence — starts a new list.
        match = LIST_ITEM_RE.match(line)
        if match:
            num = int(match.group(1))
            if buf_type != "list" or last_num is None or num != last_num + 1:
                flush()
                buf_type = "list"
            buf.append(line)
            last_num = num
            continue

        # Otherwise it's a wrapped continuation line: join onto the current item/paragraph.
        if buf_type == "list" and buf:
            buf[-1] = f"{buf[-1]} {line}"
        else:
            buf.append(line)
    flush()

    parts = []
    for kind, content in blocks:
        if kind == "header":
            parts.append(f"**{content[0]}**")
        elif kind == "list":
            parts.append("\n".join(content))
        else:
            parts.append(" ".join(content))
    return "\n\n".join(parts)
