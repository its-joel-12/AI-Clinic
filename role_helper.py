"""
Utilities for tagging conversational roles (Doctor vs Patient) on top of
chunked transcripts.

Stage 1: Call an LLM (OpenAI Responses API) to classify each chunk.
Stage 2: Smooth the assignments with lightweight heuristics to correct
obvious alternation and greeting issues.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

DEFAULT_ROLE_MODEL = os.getenv("OPENAI_ROLE_MODEL", "gpt-4o-mini")

CLASSIFIER_SYSTEM_PROMPT = """\
You are an expert medical conversation analyst.
For each utterance in the transcript, label the speaker as Doctor, Patient, or Unknown.
- Doctor: clinicians, medical staff, anyone providing care instructions.
- Patient: individuals describing their symptoms, history, or concerns.
- Unknown: transcription is too ambiguous to decide.

Respond in JSON that matches the provided schema. Keep rationales brief (<=140 chars).
"""


@dataclass
class ClassifiedChunk:
    index: int
    role: str
    rationale: str
    confidence: Optional[float] = None


class RoleTagger:
    """LLM-backed role classifier with simple post smoothing."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_ROLE_MODEL,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.logger = logger or logging.getLogger(__name__)

    def classify(self, chunks: List[str]) -> List[Dict[str, Optional[str]]]:
        if not chunks:
            return []

        llm_assignments = self._call_llm(chunks)
        smoothed = self._smooth_roles(chunks, llm_assignments)
        return [
            {
                "index": item.index,
                "role": item.role,
                "rationale": item.rationale,
                "confidence": item.confidence,
            }
            for item in smoothed
        ]

    # ------------------------------------------------------------------ #
    # Stage 1 – LLM classification
    # ------------------------------------------------------------------ #
    def _call_llm(self, chunks: List[str]) -> List[ClassifiedChunk]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": CLASSIFIER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(chunks),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            self.logger.error(
                "Role tagging LLM error %s: %s",
                response.status_code,
                response.text[:400],
            )
            response.raise_for_status()

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
        except (KeyError, ValueError, IndexError) as exc:
            self.logger.error(
                "Failed to parse role tagging response (%s): %s", type(exc).__name__, exc
            )
            raise

        results: List[ClassifiedChunk] = []
        for entry in parsed.get("chunks", []):
            results.append(
                ClassifiedChunk(
                    index=entry.get("index", len(results)),
                    role=entry.get("role", "Unknown"),
                    rationale=entry.get("rationale", "").strip(),
                    confidence=entry.get("confidence"),
                )
            )
        return results

    def _build_user_prompt(self, chunks: List[str]) -> str:
        lines = [
            "Classify each utterance. Use concise rationales. Do not invent content.",
            "",
            "Return a JSON object with the exact shape:",
            '{"chunks": [{"index": 0, "role": "Doctor", "rationale": "...", "confidence": 0.75}]}',
            "Confidence may be null if unsure.",
            "",
            "Transcript:",
        ]
        for idx, chunk in enumerate(chunks):
            lines.append(f"{idx}: {chunk.strip()}")
        lines.append("")
        lines.append("Return JSON with `chunks` array as specified.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Stage 2 – Heuristic smoothing
    # ------------------------------------------------------------------ #
    def _smooth_roles(
        self,
        chunks: List[str],
        assignments: List[ClassifiedChunk],
    ) -> List[ClassifiedChunk]:
        lookup = {item.index: item for item in assignments}
        ordered: List[ClassifiedChunk] = []
        for idx, text in enumerate(chunks):
            ordered.append(
                lookup.get(idx, ClassifiedChunk(index=idx, role="Unknown", rationale=""))
            )

        def guess_from_text(text: str) -> Optional[str]:
            lower = text.lower()
            if "doctor" in lower and ("thank" in lower or lower.startswith("good ")):
                return "Patient"
            if "prescribe" in lower or "take" in lower and "mg" in lower:
                return "Doctor"
            if lower.endswith("?") or lower.startswith(("how", "what", "have you")):
                return "Doctor"
            if lower.startswith(("i ", "my ", "well", "yeah", "no", "thanks")):
                return "Patient"
            return None

        # Choose anchor role
        anchor_idx = next((i for i, item in enumerate(ordered) if item.role != "Unknown"), 0)
        anchor_role = ordered[anchor_idx].role
        if anchor_role == "Unknown":
            guess = guess_from_text(chunks[anchor_idx])
            if guess:
                ordered[anchor_idx].role = guess
                ordered[anchor_idx].rationale = (
                    ordered[anchor_idx].rationale + " | Heuristic anchor"
                ).strip(" |")

        # Forward pass – fill unknowns and fix repetitions
        for idx, item in enumerate(ordered):
            text = chunks[idx].strip()
            if item.role == "Unknown":
                inferred = guess_from_text(text)
                if inferred is None and idx > 0:
                    prev_role = ordered[idx - 1].role
                    if prev_role in {"Doctor", "Patient"}:
                        inferred = "Patient" if prev_role == "Doctor" else "Doctor"
                if inferred:
                    item.role = inferred
                    item.rationale = (item.rationale + " | Heuristic fill").strip(" |")
            else:
                if (
                    idx > 0
                    and item.role == ordered[idx - 1].role
                    and ordered[idx - 1].role in {"Doctor", "Patient"}
                ):
                    prev_text = chunks[idx - 1].strip()
                    if prev_text.endswith("?") and text and text[0].islower():
                        item.role = "Patient" if item.role == "Doctor" else "Doctor"
                        item.rationale = (item.rationale + " | Alternation fix").strip(
                            " |"
                        )

        # Backward pass – ensure ending consistency
        for idx in range(len(ordered) - 2, -1, -1):
            item = ordered[idx]
            if item.role != "Unknown":
                continue
            next_role = ordered[idx + 1].role
            if next_role in {"Doctor", "Patient"}:
                item.role = "Patient" if next_role == "Doctor" else "Doctor"
                item.rationale = (item.rationale + " | Back-fill").strip(" |")

        return ordered

