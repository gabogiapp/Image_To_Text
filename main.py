"""Command-line utility to extract text from images using Tesseract OCR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from src.utils import list_images, read_text_from_image


def _resolve_image_paths(image: str | None, folder: str | None) -> list[Path]:
    if image:
        return [Path(image)]

    if not folder:
        raise ValueError("Either an image path or a folder must be provided.")

    images = list_images(folder)
    return [Path(path) for path in images]


def _ensure_output_dir(output_file: Path) -> None:
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)


def _save_results(output_file: Path, results: Iterable[dict[str, str]]) -> None:
    _ensure_output_dir(output_file)
    payload = {"images": list(results)}
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def extract_texts(image_paths: Iterable[Path], *, language: str) -> list[dict[str, str]]:
    extracted: list[dict[str, str]] = []

    for path in image_paths:
        try:
            text = read_text_from_image(path, language=language)
            extracted.append({"filename": path.name, "text": text})
        except Exception as exc:
            extracted.append({"filename": path.name, "error": str(exc)})

    return extracted


def display_results(results: Iterable[dict[str, str]]) -> None:
    for result in results:
        header = f"=== {result['filename']} ==="
        print(header)
        print(result.get("text") or result.get("error", "No text found."))
        print("-" * len(header))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract text from images using EasyOCR.")
    parser.add_argument("--image", type=str, help="Path to a single image to process.")
    parser.add_argument("--folder", type=str, help="Folder containing images to process (defaults to data).", default="data")
    parser.add_argument("--language", type=str, help="Language code(s) to use (e.g., 'en', 'eng', 'en,fr').", default="eng")
    parser.add_argument("--output", type=str, help="Optional JSON file to store OCR output.", default="outputs/ocr_results.json")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    image_paths = _resolve_image_paths(args.image, args.folder)

    if not image_paths:
        print("No images found to process.")
        return

    results = extract_texts(image_paths, language=args.language)
    display_results(results)

    if args.output:
        _save_results(Path(args.output), results)
        print(f"OCR results saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()