from __future__ import annotations

import os
from functools import lru_cache

import easyocr
import torch


def list_images(folder: str) -> list[str]:
    """Return a list of image file paths in a folder."""
    valid_exts = (".png", ".jpg", ".jpeg")
    return [
        os.path.join(folder, file_name)
        for file_name in os.listdir(folder)
        if file_name.lower().endswith(valid_exts)
    ]


_LANGUAGE_CODE_MAP: dict[str, str] = {
    "eng": "en",
    "en": "en",
    "fra": "fr",
    "fre": "fr",
    "fr": "fr",
    "spa": "es",
    "esp": "es",
    "es": "es",
    "deu": "de",
    "ger": "de",
    "de": "de",
    "por": "pt",
    "pt": "pt",
    "ita": "it",
    "it": "it",
}


def _normalize_language_codes(language: str | None) -> tuple[str, ...]:
    if not language:
        return ("en",)

    candidates = [code.strip().lower() for code in language.split(",") if code.strip()]
    normalized: list[str] = []

    for code in candidates:
        mapped = _LANGUAGE_CODE_MAP.get(code, code)
        if mapped:
            normalized.append(mapped)

    seen: set[str] = set()
    ordered: list[str] = []
    for code in normalized:
        if code not in seen:
            ordered.append(code)
            seen.add(code)

    return tuple(ordered or ("en",))


@lru_cache(maxsize=8)
def _get_easyocr_reader(languages: tuple[str, ...], use_gpu: bool | None) -> easyocr.Reader:
    gpu_flag = use_gpu if use_gpu is not None else torch.cuda.is_available()
    return easyocr.Reader(list(languages), gpu=gpu_flag)


def read_text_from_image(
    image_path: str,
    *,
    language: str = "en",
    use_gpu: bool | None = None,
) -> str:
    """Extract plain text from an image using EasyOCR.

    Parameters
    ----------
    image_path: str
        Absolute or relative path to an image file.
    language: str
        Comma-separated language codes (defaults to English).
        Examples: "en", "en,fr". Tesseract-style codes like "eng" are mapped automatically.
    use_gpu: bool | None
        Override GPU usage. ``True`` forces GPU, ``False`` forces CPU, ``None`` auto-detects.
    """
    resolved_path = os.path.abspath(image_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Image not found at {resolved_path}")

    languages = _normalize_language_codes(language)

    try:
        reader = _get_easyocr_reader(languages, use_gpu)
        results = reader.readtext(resolved_path, detail=0, paragraph=True)
    except Exception as exc:  # pragma: no cover - surface easyocr errors
        raise RuntimeError(f"OCR failed for {os.path.basename(resolved_path)}: {exc}") from exc

    lines = [chunk.strip() for chunk in results if chunk and chunk.strip()]
    return "\n".join(lines)