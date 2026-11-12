import logging
import re
from pathlib import Path
from typing import List

import spacy

path_to_root = "your_root_path

INPUT = Path(path_to_root) / "out/transl"
OUTPUT = Path(path_to_root) / "out/anon"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "en_core_web_trf"  # fallback: en_core_web_lg
BATCH_SIZE = 32

# spaCy entity label -> placeholder
ENT_MAP = {
    "PERSON": "[NAME]",
    "GPE": "[ADDRESS]",
    "LOC": "[ADDRESS]",
    "FAC": "[ADDRESS]",
    "MONEY": "[AMOUNT]",
}

# Applied before NER. Order matters.
CURRENCY = r"(?:€|EUR|eur(?:os?)?|euros?|mk|markka[a-z]*|FIM|\$|USD|£|GBP)"
NUM = r"\d{1,3}(?:[ .,\u00a0]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
REGEX_RULES = [
    (re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{2,4}){3,8}\b"), "[ACCOUNT]"),  # IBAN
    (re.compile(r"\b\d{6}-\d{2,8}\b"), "[ACCOUNT]"),  # old Finnish account no.
    (re.compile(r"\b\d{6}[-+A]\d{3}[0-9A-Y]\b"), "[ID]"),  # Finnish personal ID
    (re.compile(rf"(?:{CURRENCY}\s?(?:{NUM})|(?:{NUM})\s?{CURRENCY})(?![A-Za-z])", re.I), "[AMOUNT]"),
    (re.compile(r"\b[A-Z][a-zA-Z]+(?:katu|tie|kuja|polku|vagen|gatan|street|road|avenue)\s+\d+\s?[A-Za-z]?\b", re.I), "[ADDRESS]"),
    (re.compile(r"\b\d{5}\s+[A-Z][a-zA-Z]+\b"), "[ADDRESS]"),  # postcode + city
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
_log = logging.getLogger(__name__)

spacy.prefer_gpu()
try:
    nlp = spacy.load(MODEL_NAME, disable=["lemmatizer"])
except OSError:
    _log.warning(f"{MODEL_NAME} not found, falling back to en_core_web_lg")
    nlp = spacy.load("en_core_web_lg", disable=["lemmatizer"])
nlp.max_length = 2_000_000

# Embedded base64 images must not be touched.
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")


def apply_regex(text: str) -> str:
    for pattern, repl in REGEX_RULES:
        text = pattern.sub(repl, text)
    return text


def anonymize_lines(lines: List[str]) -> List[str]:
    """Regex pass, then batched NER pass. Lines with embedded images are
    split so only the surrounding text is processed."""
    segments: List[str] = []  # processable text pieces
    layout = []  # per line: list of ("raw", str) or ("seg", index)

    for line in lines:
        parts = []
        last = 0
        for m in IMAGE_RE.finditer(line):
            if m.start() > last:
                parts.append(("seg", len(segments)))
                segments.append(apply_regex(line[last:m.start()]))
            parts.append(("raw", m.group(0)))
            last = m.end()
        tail = line[last:]
        if tail or not parts:
            parts.append(("seg", len(segments)))
            segments.append(apply_regex(tail))
        layout.append(parts)

    results: List[str] = []
    for seg, doc in zip(segments, nlp.pipe(segments, batch_size=BATCH_SIZE)):
        out, last = [], 0
        for ent in doc.ents:
            if ent.label_ in ENT_MAP:
                out.append(seg[last:ent.start_char])
                out.append(ENT_MAP[ent.label_])
                last = ent.end_char
        out.append(seg[last:])
        results.append("".join(out))

    return [
        "".join(results[p[1]] if p[0] == "seg" else p[1] for p in parts)
        for parts in layout
    ]


def process_markdown(md_path: Path):
    output_file = OUTPUT / f"{md_path.stem.replace('_translated', '')}_anonymized.md"

    if output_file.exists():
        _log.info(f"Skipping: {md_path.name}")
        return

    _log.info(f"Processing: {md_path.name}")

    try:
        text = md_path.read_text(encoding="utf-8")
        anonymized = anonymize_lines(text.split("\n"))
        output_file.write_text("\n".join(anonymized), encoding="utf-8")
        _log.info(f"Finished: {md_path.name}")
    except Exception:
        _log.exception(f"FAILED: {md_path.name}")
        raise


def main():
    md_files = sorted(INPUT.glob("*_translated.md"))

    if not md_files:
        _log.warning("No files found.")
        return

    _log.info(f"Found {len(md_files)} files.")

    for md_file in md_files:
        try:
            process_markdown(md_file)
        except Exception:
            continue

    _log.info("All files processed.")


if __name__ == "__main__":
    main()
