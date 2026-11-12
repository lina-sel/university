import logging
from pathlib import Path

from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

path_to_root = "your_file_path"
INPUT = Path(path_to_root) / "out/qc_scan_pass"
OUTPUT = Path(path_to_root) / "out/transc"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_RESOLUTION_SCALE = 2.0

NUM_THREADS = 4

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def build_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["fi", "sv", "en"])

    # Explicitly pin to GPU + the number of CPU threads allocated.
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=NUM_THREADS,
        device=AcceleratorDevice.CUDA,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def main():
    pdf_files = sorted(INPUT.glob("*.pdf"))

    if not pdf_files:
        _log.info("No PDF files found.")
        return

    # Skip files already processed, before we even hand them to the converter.
    todo = [p for p in pdf_files if not (OUTPUT / f"{p.stem}_original.md").exists()]
    skipped = len(pdf_files) - len(todo)
    if skipped:
        _log.info(f"Skipping {skipped} already-processed file(s).")

    if not todo:
        _log.info("Nothing to do.")
        return

    _log.info(f"Found {len(todo)} PDFs to process.")

    # Build the converter (and load its models onto the GPU) exactly once.
    doc_converter = build_converter()

    for conv_res in doc_converter.convert_all(todo, raises_on_error=False):
        pdf_path = conv_res.input.file
        original_md_path = OUTPUT / f"{pdf_path.stem}_original.md"

        if conv_res.status.name != "SUCCESS":
            _log.error(f"FAILED: {pdf_path.name} -> {conv_res.status}")
            for err in conv_res.errors:
                _log.error(f"  {err}")
            continue

        conv_res.document.save_as_markdown(
            original_md_path,
            image_mode=ImageRefMode.EMBEDDED,
        )
        _log.info(f"Finished: {pdf_path.name}")

    _log.info("All files processed.")


if __name__ == "__main__":
    main()
