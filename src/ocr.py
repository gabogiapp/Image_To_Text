"""Utility functions for optical character recognition (OCR)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import easyocr


class OCRExtractor:
    """Extract plain text from images using EasyOCR."""

    def __init__(self, *, language: str = "en", use_gpu: bool | None = None) -> None:
        self.languages = self._normalize_languages(language)
        self.reader = self._create_reader(use_gpu)

    @staticmethod
    def _normalize_languages(language: str | None) -> list[str]:
        if not language:
            return ["en"]

        mapping = {
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

        normalized: list[str] = []
        for code in language.split(","):
            code = code.strip().lower()
            if not code:
                continue
            mapped = mapping.get(code, code)
            if mapped not in normalized:
                normalized.append(mapped)

        return normalized or ["en"]

    def _create_reader(self, use_gpu: bool | None) -> easyocr.Reader:
        gpu_flag = use_gpu if use_gpu is not None else torch.cuda.is_available()
        return easyocr.Reader(self.languages, gpu=gpu_flag)

    def extract_text(self, image_path: str | Path) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.reader.readtext(str(image_path), detail=0, paragraph=True)
        lines = [chunk.strip() for chunk in results if chunk and chunk.strip()]
        return "\n".join(lines)

    def extract_text_batch(self, image_paths: Iterable[str | Path]) -> dict[str, str]:
        results: dict[str, str] = {}
        for image_path in image_paths:
            path_obj = Path(image_path)
            try:
                results[path_obj.name] = self.extract_text(path_obj)
            except Exception as exc:  # fallback to error message
                results[path_obj.name] = f"OCR failed: {exc}"
        return results