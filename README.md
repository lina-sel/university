# university
repository for work associate with the Behavioral Philanthropy Lab


# Finnish/Swedish PDF → English Markdown pipeline

Two scripts, run in order:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Transcribe | `transcription-5.py` | `out/qc_scan_pass/*.pdf` | `out/transc/<name>_original.md` |
| 2. Translate | `translation-3.py` | `out/transc/*_original.md` | `out/transl/<name>_translated.md` |

Step 1 runs OCR and layout analysis on scanned PDFs with [Docling](https://github.com/docling-project/docling) (EasyOCR). Step 2 detects whether each document is Finnish or Swedish and translates it to English with the matching MarianMT model (`Helsinki-NLP/opus-mt-fi-en` or `opus-mt-sv-en`).

Both scripts skip files that already have an output, so you can re-run them safely after a crash or after adding new PDFs.

## Requirements

- Python 3.10+
- An NVIDIA GPU with a working CUDA driver (strongly recommended; see [No GPU](#no-gpu))
- Internet access on first run to download models (see [Offline / cluster nodes](#offline--cluster-nodes))

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch with CUDA

Install this **first**, using the command from <https://pytorch.org/get-started/locally/> that matches your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check that it sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

If you are on a cluster, you may need to load a CUDA module first (e.g. `module load CUDA`). Check what your cluster offers with `module avail`.

### 3. Install the remaining packages

```bash
pip install docling docling-core easyocr
pip install transformers sentencepiece langdetect
```

What each is for:

| Package | Used by | Purpose |
|---------|---------|---------|
| `docling`, `docling-core` | both | PDF parsing / layout analysis, document model, Markdown export |
| `easyocr` | transcription | OCR engine (normally pulled in by docling; listed to be explicit) |
| `transformers` | translation | MarianMT models and tokenizer |
| `sentencepiece` | translation | **Required** by `MarianTokenizer`; without it the tokenizer fails to load |
| `langdetect` | translation | Finnish vs Swedish detection |

Or in one go, with a `requirements.txt` (still install `torch` first):

```text
docling
docling-core
easyocr
transformers
sentencepiece
langdetect
```

```bash
pip install -r requirements.txt
```

## Configuration

### Paths

Both scripts hard-code the project root near the top:

```python
path_to_root = "/home/users/s/selyanin/Finland"
```

Change this if you move the project. The expected layout is:

```
Finland/
├── out/
│   ├── qc_scan_pass/    <- put your input PDFs here
│   ├── transc/          <- created by step 1 (Markdown in original language)
│   └── transl/          <- created by step 2 (English Markdown)
```

`transc/` and `transl/` are created automatically. `qc_scan_pass/` must already exist and contain the PDFs.

### OCR language (recommended change)

`transcription-5.py` currently uses `EasyOcrOptions()` with its defaults. Docling's default OCR language list is French, German, Spanish and English, which does not include Finnish or Swedish. Characters such as `ä`, `ö` and `å` may come out wrong. Set the languages explicitly:

```python
pipeline_options.ocr_options = EasyOcrOptions(lang=["fi", "sv", "en"])
```

Poor OCR here will carry through to the translation, so this is worth doing before a large run.

### Tuning

| Setting | File | Default | Notes |
|---------|------|---------|-------|
| `NUM_THREADS` | `transcription-5.py` | `4` | Match the CPU cores you have allocated |
| `IMAGE_RESOLUTION_SCALE` | `transcription-5.py` | `2.0` | Higher = better OCR on small print, slower and more memory |
| `BATCH_SIZE` | `translation-3.py` | `32` | Raise if you have spare GPU memory; lower if you hit out-of-memory errors |
| `DETECT_SAMPLE_CHARS` | `translation-3.py` | `20000` | How much text is used to decide Finnish vs Swedish |

## Running

```bash
python transcription-5.py
python translation-3.py
```

Check the logs as you go:

- Step 1 logs `FAILED: <file>` for PDFs that could not be converted and continues with the rest.
- Step 2 logs the detected language and probabilities for each file. Files that are neither Finnish nor Swedish are skipped with a warning and no output is written.

To redo a file, delete its output (`*_original.md` or `*_translated.md`) and run the script again.

## Model downloads

Models are downloaded automatically on first use and cached:

- Docling layout/table models and EasyOCR models (step 1)
- `opus-mt-fi-en` and `opus-mt-sv-en` from Hugging Face (step 2, each loaded only when a document in that language is found)

### Offline / cluster nodes

If compute nodes have no internet access, do a dry run on a login node (or any machine with internet) first so everything is cached, then point jobs at the same cache. To control where Hugging Face stores models:

```bash
export HF_HOME=/path/to/shared/cache/huggingface
```

You can then force offline mode on the compute node:

```bash
export HF_HUB_OFFLINE=1
```

To pre-fetch the translation models:

```bash
python -c "
from transformers import MarianMTModel, MarianTokenizer
for m in ['Helsinki-NLP/opus-mt-fi-en', 'Helsinki-NLP/opus-mt-sv-en']:
    MarianTokenizer.from_pretrained(m); MarianMTModel.from_pretrained(m)
"
```

Running `transcription-5.py` once on a single PDF will cache the Docling and EasyOCR models.

## No GPU

- `translation-3.py` falls back to CPU automatically (slower, but works).
- `transcription-5.py` is pinned to CUDA. On a machine without a GPU it will fail. Change this line:

  ```python
  device=AcceleratorDevice.CUDA,
  ```

  to `AcceleratorDevice.AUTO` (or `AcceleratorDevice.CPU`).

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch was installed, or the CUDA driver/module is missing. Reinstall torch from the CUDA index (step 2). |
| `ImportError` / error about `sentencepiece` when loading the tokenizer | `pip install sentencepiece` |
| `ModuleNotFoundError: langdetect` | `pip install langdetect` |
| CUDA out-of-memory in step 2 | Lower `BATCH_SIZE` |
| CUDA out-of-memory in step 1 | Lower `IMAGE_RESOLUTION_SCALE` |
| `No files found.` in step 2 | Step 1 has not run, or `out/transc/` has no `*_original.md` files |
| Swedish passages in a Finnish document translate badly | Detection is per document, so the whole file uses one model |
| Output Markdown files are very large | Images are embedded as base64 (`ImageRefMode.EMBEDDED`); this is expected |
