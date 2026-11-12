import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer

# pip install langdetect
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from docling_core.types.doc import ImageRefMode, TableItem, TextItem
from docling.document_converter import DocumentConverter


path_to_root = "/home/users/s/selyanin/Finland"

INPUT = Path(path_to_root) / "out/transc"
OUTPUT = Path(path_to_root) / "out/transl"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Language code -> MarianMT model
MODEL_NAMES = {
    "fi": "Helsinki-NLP/opus-mt-fi-en",
    "sv": "Helsinki-NLP/opus-mt-sv-en",
}

# Chunks are batched together across the WHOLE document before being sent
# to the model. Raise this if you have GPU memory to spare -- it directly
# trades memory for throughput.
BATCH_SIZE = 32

# How many characters of the document to use for language detection.
DETECT_SAMPLE_CHARS = 20000

# langdetect is non-deterministic by default; fix the seed.
DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
_log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_log.info(f"Using device: {device}")

use_amp = device.type == "cuda"


class Translator:
    """A tokenizer + MarianMT model pair for one source language."""

    def __init__(self, model_name: str):
        _log.info(f"Loading model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def chunk_by_tokens(self, text: str, max_tokens: int = 400) -> List[str]:
        if not text or not text.strip():
            return []

        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_ids = tokens[i:i + max_tokens]
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))

        return chunks

    def translate_batch(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        inputs = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model.generate(**inputs, max_new_tokens=512)
            else:
                outputs = self.model.generate(**inputs, max_new_tokens=512)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_many(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[str]:
        """Translate a list of independent text strings, batching ALL of their
        chunks together so the model runs in a handful of large forward passes
        instead of one small pass per text."""
        if not texts:
            return []

        all_chunks: List[str] = []
        chunk_counts: List[int] = []
        for text in texts:
            chunks = self.chunk_by_tokens(text)
            chunk_counts.append(len(chunks))
            all_chunks.extend(chunks)

        translated_chunks: List[str] = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            translated_chunks.extend(self.translate_batch(batch))

        results = []
        idx = 0
        for text, count in zip(texts, chunk_counts):
            if count == 0:
                results.append(text)
            else:
                results.append(" ".join(translated_chunks[idx: idx + count]))
                idx += count

        return results


# Models are loaded lazily the first time a language is needed, then reused.
_translators: Dict[str, Translator] = {}


def get_translator(lang: str) -> Translator:
    if lang not in _translators:
        _translators[lang] = Translator(MODEL_NAMES[lang])
    return _translators[lang]


def detect_language(texts: List[str]) -> Optional[str]:
    """Return 'fi' or 'sv' for the document, or None if it is neither.

    Detection is restricted to Finnish vs Swedish: we take the detector's
    probabilities for those two languages and pick the larger one."""
    sample = "\n".join(texts)[:DETECT_SAMPLE_CHARS]
    if not sample.strip():
        return None

    try:
        probs = {r.lang: r.prob for r in detect_langs(sample)}
    except LangDetectException:
        return None

    candidates = {lang: probs.get(lang, 0.0) for lang in MODEL_NAMES}
    best = max(candidates, key=candidates.get)
    if candidates[best] == 0.0:
        _log.warning(f"Neither fi nor sv detected (got: {probs})")
        return None

    _log.info(f"Language probabilities: {probs}")
    return best


def process_markdown(md_path: Path, doc_converter: DocumentConverter):
    output_file = OUTPUT / f"{md_path.stem.replace('_original','')}_translated.md"

    if output_file.exists():
        _log.info(f"Skipping: {md_path.name}")
        return

    _log.info(f"Processing: {md_path.name}")

    conv_res = doc_converter.convert(md_path)
    conv_doc = conv_res.document

    try:
        # Collect every translatable text span in the document first, so we
        # can translate them all together instead of one at a time.
        targets: List[Tuple] = []  # (kind, object)
        texts: List[str] = []

        for element, _ in conv_doc.iterate_items():
            if isinstance(element, TextItem):
                if element.text:
                    targets.append(("text_item", element))
                    texts.append(element.text)

            elif isinstance(element, TableItem):
                for cell in element.data.table_cells:
                    if cell.text:
                        targets.append(("table_cell", cell))
                        texts.append(cell.text)

        lang = detect_language(texts)
        if lang is None:
            _log.warning(f"Could not determine Finnish/Swedish, skipping: {md_path.name}")
            return

        _log.info(f"Detected language for {md_path.name}: {lang}")
        translator = get_translator(lang)

        translated_texts = translator.translate_many(texts)

        for (kind, obj), translated in zip(targets, translated_texts):
            obj.text = translated

        conv_doc.save_as_markdown(
            output_file,
            image_mode=ImageRefMode.EMBEDDED
        )

        _log.info(f"Finished: {md_path.name}")

    except Exception:
        _log.exception(f"FAILED: {md_path.name}")
        raise


def main():
    md_files = sorted(INPUT.glob("*_original.md"))

    if not md_files:
        _log.warning("No files found.")
        return

    _log.info(f"Found {len(md_files)} files.")

    # Built once, reused for every file.
    doc_converter = DocumentConverter()

    for md_file in md_files:
        try:
            process_markdown(md_file, doc_converter)
        except Exception:
            continue

    _log.info("All files processed.")


if __name__ == "__main__":
    main()