import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    # Remove repeated spaces, fix OCR apostrophes, etc.
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def paragraph_chunks(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged: List[str] = []
    buffer = ""
    for para in paragraphs:
        if not buffer:
            buffer = para
            continue
        if len(buffer) + len(para) < 700:
            buffer = f"{buffer}\n\n{para}"
        else:
            merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)
    return merged


def sliding_window(chunks: Iterable[str], max_chars: int, overlap: int) -> List[str]:
    windows: List[str] = []
    current = ""
    for chunk in chunks:
        if not current:
            current = chunk
            continue
        if len(current) + overlap + len(chunk) <= max_chars:
            current = f"{current}\n\n{chunk}"
        else:
            windows.append(current)
            current = chunk
    if current:
        windows.append(current)
    return windows

