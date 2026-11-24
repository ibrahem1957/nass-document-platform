"""Static, versioned prompt templates.

Templates are fixed strings with no placeholders: a prompt can never carry
ground truth, extractor text, or any document-derived content. Changing a
template requires a new id so earlier runs stay reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    text: str
    source: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


PROMPTS: dict[str, PromptTemplate] = {
    template.prompt_id: template
    for template in (
        PromptTemplate(
            "vlm_transcribe_v1",
            (
                "Transcribe all text on this page exactly as printed, in natural "
                "reading order. Keep every digit in its original script (Arabic-Indic "
                "or Western) and keep symbols such as %, ≤, ≥ exactly. Write Arabic in "
                "logical order. Render each table as a Markdown table with the header "
                "row first and the row-header column first. Output only the "
                "transcription, without commentary."
            ),
            "AgenticDoc E2-A generic VLM transcription prompt",
        ),
        PromptTemplate(
            "qari_free_ocr_v1",
            "Free OCR.",
            "Qari-OCR-0.4.0 model card usage example",
        ),
        PromptTemplate(
            "paddle_ocr_v1",
            "OCR:",
            "PaddleOCR-VL-1.6 model card, task 'ocr'",
        ),
        PromptTemplate(
            "paddle_table_v1",
            "Table Recognition:",
            "PaddleOCR-VL-1.6 model card, task 'table'",
        ),
    )
}


def get_prompt(prompt_id: str) -> PromptTemplate:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt template: {prompt_id}") from exc
