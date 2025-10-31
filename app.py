"""
Flask service that accepts an uploaded audio file and returns a transcription
using OpenAI Whisper, plus optional Doctor/Patient role tagging.

Expected environment:
    export OPENAI_API_KEY=your-openai-key

Usage:
    pip install flask requests python-dotenv
    python app.py

POST /transcribe with multipart/form-data:
    audio (file, required)  -> Audio file to transcribe (mp3/wav/etc.)
    language_code (str)     -> Optional BCP-47 language code (default en-US)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import OpenAI
from role_helper import RoleTagger
from requests.adapters import HTTPAdapter


load_dotenv()

app = Flask(__name__)


from flask_cors import CORS  # type: ignore

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    expose_headers=["Content-Type", "Authorization"],
)

# Optional response compression to reduce payload transfer time
try:
    from flask_compress import Compress  # type: ignore

    Compress(app)
except Exception:
    pass

# Ensure INFO logs reach the console even when Flask config changes handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
if not app.logger.handlers:
    app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.INFO)

OPENAI_WHISPER_MODEL = "whisper-1"
OPENAI_NOTE_MODEL = os.getenv("OPENAI_NOTE_MODEL", "gpt-4o-mini")
_role_tagger: Optional[RoleTagger] = None
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_MODEL_DEFAULT = os.getenv("CODES_MODEL", "gpt-4.1")
_client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else None

# Shared HTTP session and executor to reduce per-request overhead
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20))
SESSION.mount("http://", HTTPAdapter(pool_connections=10, pool_maxsize=20))

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("WORKER_POOL_SIZE", "4"))
)

ASR_TIMEOUT = int(os.getenv("OPENAI_ASR_TIMEOUT", "120"))
CHAT_TIMEOUT = int(os.getenv("OPENAI_CHAT_TIMEOUT", "60"))


def _openai_headers(include_json: bool = False) -> Dict[str, str]:
    api_key = get_openai_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    if include_json:
        headers["Content-Type"] = "application/json"
    return headers



def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_role_tagger(api_key: str) -> RoleTagger:
    """Create (and cache) the role tagger helper."""
    global _role_tagger
    if _role_tagger is None or _role_tagger.api_key != api_key:
        _role_tagger = RoleTagger(api_key=api_key, logger=app.logger)
    return _role_tagger


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide an API key to use Whisper and note generation."
        )
    return api_key


def transcribe_with_whisper(
    audio_bytes: bytes,
    filename: Optional[str],
    *,
    language_code: Optional[str],
    logger: logging.Logger,
    include_roles: bool = True,
    include_chunks: bool = True,
    include_words: bool = True,
) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper API and enrich with role tags."""
    api_key = get_openai_api_key()

    files = {
        "file": (
            filename or "audio.mp3",
            audio_bytes,
            "application/octet-stream",
        )
    }
    data = {
        "model": OPENAI_WHISPER_MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language_code:
        # Whisper expects ISO-639-1 codes, so trim to the first 2 letters.
        iso_lang = language_code.split("-")[0].lower()
        data["language"] = iso_lang

    logger.info(
        "Sending audio to OpenAI Whisper model=%s language=%s",
        OPENAI_WHISPER_MODEL,
        language_code,
    )

    response = SESSION.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=_openai_headers(),
        data=data,
        files=files,
        timeout=ASR_TIMEOUT,
    )
    if response.status_code != 200:
        logger.error(
            "Whisper API error %s: %s", response.status_code, response.text[:500]
        )
        response.raise_for_status()

    payload = response.json()

    segments = payload.get("segments", []) or []
    transcript = payload.get("text", "").strip()

    chunk_details: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    if include_chunks or include_roles:
        for seg in segments:
            if include_chunks:
                chunk_details.append(
                    {
                        "transcript": seg.get("text", "").strip(),
                        "confidence": seg.get("avg_logprob"),
                        "channel_tag": None,
                    }
                )
            if include_words:
                for word in seg.get("words", []) or []:
                    words.append(
                        {
                            "word": word.get("word", "").strip(),
                            "start_time": word.get("start"),
                            "end_time": word.get("end"),
                            "speaker_tag": word.get("speaker"),
                            "confidence": word.get("confidence"),
                            "channel_tag": None,
                        }
                    )

    role_annotations: List[Dict[str, Any]] = []
    role_model_name: Optional[str] = None
    if include_roles and chunk_details:
        try:
            role_tagger = get_role_tagger(api_key)
            role_annotations = role_tagger.classify(
                [detail["transcript"] for detail in chunk_details]
            )
            role_model_name = role_tagger.model
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Role tagging failed: %s", exc)

    if include_roles and chunk_details:
        for idx, detail in enumerate(chunk_details):
            match = next((item for item in role_annotations if item["index"] == idx), None)
            if match:
                detail["role"] = match["role"]
                detail["role_rationale"] = match["rationale"]
                detail["role_confidence"] = match.get("confidence")
            else:
                detail.setdefault("role", "Unknown")

    dialogue = ""
    if include_chunks:
        dialogue_lines: List[str] = []
        for detail in chunk_details:
            role_label = (detail.get("role") or "Unknown").lower() if include_roles else "speaker"
            dialogue_lines.append(f"{role_label}: {detail.get('transcript', '').strip()}")
        dialogue = "\n".join(dialogue_lines)

    metadata = {
        "provider": "whisper",
        "language_code": language_code,
        "model": OPENAI_WHISPER_MODEL,
        "segments": len(segments),
        "role_model": role_model_name,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    payload: Dict[str, Any] = {
        "transcript": transcript,
        "diarization": [],  # Whisper API does not provide diarization tags.
        "config": metadata,
    }
    if include_chunks:
        payload["chunks"] = [detail["transcript"] for detail in chunk_details]
        payload["chunk_details"] = chunk_details
        payload["dialoge"] = dialogue
        payload["dialogue"] = dialogue
    if include_words:
        payload["words"] = words
    return payload

# CORS fallback headers (in case flask-cors isn't installed)
@app.after_request
def _add_cors_headers(resp):
    try:
        origin = request.headers.get("Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Origin", origin or "*")
        resp.headers.setdefault("Vary", "Origin")
        resp.headers.setdefault("Access-Control-Allow-Credentials", "true")
        req_headers = request.headers.get(
            "Access-Control-Request-Headers", "Authorization, Content-Type"
        )
        resp.headers.setdefault("Access-Control-Allow-Headers", req_headers)
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    except Exception:
        pass
    return resp


def soap_prompt(body: str) -> str:
    return f"""
You are a clinical documentation assistant.

Generate a concise SOAP note using ONLY the supplied doctor-patient dialog and narrative.

Rules:
- Use facts from both sources.
- Do not invent data (vitals, labs, etc.).
- If a section lacks information, output "Not discussed."
- Return plain text only; avoid Markdown, bullet symbols, or decorative characters.

Transcript:
---
{body}
---

Return the SOAP note with sections labeled S, O, A, and P.
"""


def dap_prompt(body: str) -> str:
    return f"""
You are a clinical documentation assistant.

Create a fact-based DAP note (Data, Assessment, Plan) from the dialog/narrative below.
- Use only information explicitly provided.
- If a section has no data, write "Not discussed."
- Output only the final DAP note.
- Return plain text only; avoid Markdown, bullet symbols, or decorative characters.

Transcript:
---
{body}
---
"""


def birp_prompt(body: str) -> str:
    return f"""
You are a clinical documentation assistant.

Produce a concise BIRP note using only the supplied information.
- B: Observable behaviors / symptoms mentioned.
- I: Clinician interventions or guidance; if none, "Not discussed."
- R: Patient response.
- P: Follow-up / plan.
- Do not invent details.
- Return plain text only; avoid Markdown, bullet symbols, or decorative characters.

Transcript:
---
{body}
---
"""


NOTE_PROMPTS = {
    "SOAP": soap_prompt,
    "DAP": dap_prompt,
    "BIRP": birp_prompt,
}

NOTE_LABELS = {
    "SOAP": ["S", "O", "A", "P"],
    "DAP": ["D", "A", "P"],
    "BIRP": ["B", "I", "R", "P"],
}


def openai_chat(messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    api_key = get_openai_api_key()
    response = SESSION.post(
        "https://api.openai.com/v1/chat/completions",
        headers=_openai_headers(include_json=True),
        json={
            "model": OPENAI_NOTE_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=CHAT_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI chat completion error {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Malformed OpenAI response: {data}") from exc


def parse_sections(text: str, labels: List[str]) -> Dict[str, str]:
    label_pattern = "|".join(labels)
    sections = {label: "Not discussed" for label in labels}
    regex_parts = []
    for index, label in enumerate(labels):
        next_labels = "|".join(labels[index + 1 :])
        lookahead = rf"(?=\s*(?:{next_labels})\s*[:\-]|$)" if next_labels else r"$"
        regex_parts.append(rf"(?:{label}\s*[:\-]\s*)(?P<{label}>.*?){lookahead}")
    pattern = re.compile("|".join(regex_parts), re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        for label in labels:
            value = match.groupdict().get(label)
            if value:
                sections[label] = value.strip() or "Not discussed"
    return sections


def generate_note(format_type: str, body: str) -> Dict[str, str]:
    if format_type not in NOTE_PROMPTS:
        raise ValueError(f"Unsupported note format: {format_type}")
    prompt_fn = NOTE_PROMPTS[format_type]
    labels = NOTE_LABELS[format_type]
    prompt = prompt_fn(body)
    text = openai_chat(
        [
            {
                "role": "system",
                "content": "You are a helpful clinical documentation assistant.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    return parse_sections(text, labels)


def generate_notes_bundle(
    dialogue: str,
    narrative: Optional[str] = None,
) -> Dict[str, Any]:
    if not dialogue:
        raise ValueError("dialogue text is required to generate notes.")

    body = f"Dialog:\n{dialogue.strip()}"
    if narrative:
        body = f"{body}\n\nNarrative:\n{narrative.strip()}"

    results: Dict[str, Any] = {}
    future_to_format = {
        EXECUTOR.submit(generate_note, fmt, body): fmt for fmt in NOTE_PROMPTS
    }
    for future in concurrent.futures.as_completed(future_to_format):
        fmt = future_to_format[future]
        try:
            results[fmt] = future.result()
        except Exception as exc:
            results[fmt] = {"error": str(exc)}
    return results


def generate_summary_text(
    notes: List[str],
    note_type: str = "Clinical Notes",
    patient_name: Optional[str] = None,
    max_tokens: int = 2048,
) -> str:
    if not notes:
        raise ValueError("At least one note is required to generate a summary.")

    sanitized: List[str] = []
    for idx, note in enumerate(notes):
        if isinstance(note, dict):
            text = json.dumps(note, indent=2)
        else:
            text = str(note)
        text = re.sub(r"(?mi)^\s*(Patient Name|Patient|Name)\s*:\s*.*$", "", text)
        text = text.strip()
        if text:
            sanitized.append(f"Note {idx + 1}:\n{text}")

    combined_notes = "\n\n".join(sanitized)

    prompt = f"""
You are a clinical summarization assistant.
Using the following {note_type}, generate a clear, structured Visit or Discharge Summary.

Include these sections when information is available:
- Chief Complaint
- History and Examination
- Treatment Summary
- Medications
- Discharge Condition
- Follow-up Plans

Be concise and factual. Do not invent details. Avoid repeating headers or unrelated content.
---
{note_type}:
{combined_notes}
---
"""

    summary = openai_chat(
        [
            {
                "role": "system",
                "content": "You are a helpful clinical summarization assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )

    if patient_name:
        if re.search(r"(?mi)^Patient Name\s*:", summary):
            summary = re.sub(
                r"(?mi)^Patient Name\s*:\s*.*$",
                f"Patient Name: {patient_name}",
                summary,
                count=1,
            )
        else:
            summary = f"Patient Name: {patient_name}\n\n{summary}"

    return summary


def _flatten_notes_for_coding(notes_bundle: Any) -> str:
    """Convert structured notes (SOAP/DAP/BIRP dicts) into a single text blob.
    Only includes notes without errors.
    """
    if not isinstance(notes_bundle, dict):
        return ""
    parts: List[str] = []
    for fmt, sections in notes_bundle.items():
        if isinstance(sections, dict) and "error" not in sections:
            lines: List[str] = []
            for key, val in sections.items():
                if isinstance(val, (str, int, float)):
                    lines.append(f"{key}: {val}")
                else:
                    try:
                        lines.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")
                    except Exception:
                        lines.append(f"{key}: {str(val)}")
            if lines:
                parts.append(f"{fmt} Note\n" + "\n".join(lines))
    return "\n\n".join(parts).strip()


def process_transcription_request(*,include_notes_default: bool = False,include_summary_default: bool = False,force_include_notes: bool = False, force_include_summary: bool = False) -> tuple[Dict[str, Any], int]:
    if "audio" not in request.files:
        return {"error": 'Missing file field "audio"'}, 400

    uploaded = request.files["audio"]
    audio_bytes = uploaded.read()
    if not audio_bytes:
        return {"error": "Uploaded file is empty"}, 400

    filename = getattr(uploaded, "filename", "audio")
    app.logger.info(
        "Received audio upload filename=%s size=%d bytes", filename, len(audio_bytes)
    )

    language_code = request.form.get("language_code", "en-US")
    narrative = request.form.get("narrative")

    include_notes = parse_bool(
        request.form.get("include_notes") or request.args.get("include_notes"),
        include_notes_default,
    )
    include_summary = parse_bool(
        request.form.get("include_summary") or request.args.get("include_summary"),
        include_summary_default,
    )

    include_roles = parse_bool(
        request.form.get("include_roles") or request.args.get("include_roles"), True
    )
    include_chunks = parse_bool(
        request.form.get("include_chunks") or request.args.get("include_chunks"), True
    )
    include_words = parse_bool(
        request.form.get("include_words") or request.args.get("include_words"), False
    )
    include_codes = parse_bool(
        request.form.get("include_codes") or request.args.get("include_codes"), True
    )

    if force_include_notes:
        include_notes = True
    
    if force_include_summary:
        include_summary = True
        

    try:
        result = transcribe_with_whisper(
            audio_bytes,
            filename,
            language_code=language_code,
            logger=app.logger,
            include_roles=include_roles,
            include_chunks=include_chunks,
            include_words=include_words,
        )
    except Exception as exc:
        app.logger.error("Whisper transcription failed: %s", exc)
        return {"error": str(exc)}, 500

    dialogue_text = result.get("dialogue", "") or result.get("dialoge", "")
    notes_bundle: Optional[Dict[str, Any]] = None

    if include_notes and dialogue_text:
        try:
            notes_bundle = generate_notes_bundle(dialogue_text, narrative)
            result["notes"] = notes_bundle
            result.setdefault("config", {})["note_model"] = OPENAI_NOTE_MODEL
        except Exception as exc:
            app.logger.error("Clinical note generation failed: %s", exc)
            result["notes"] = {"error": str(exc)}

    # Run summary generation and medical code extraction concurrently to reduce latency
    futures = {}
    try:

        # Prepare inputs
        notes_text = _flatten_notes_for_coding(result.get("notes"))
        summary_source: List[Any] = []
        if include_summary:
            if notes_bundle:
                for note in notes_bundle.values():
                    if isinstance(note, dict) and "error" not in note:
                        summary_source.append(note)
            if not summary_source and result.get("transcript"):
                summary_source.append(result["transcript"])

        # Submit tasks
        if include_summary:
            if summary_source:
                futures["summary"] = EXECUTOR.submit(
                    generate_summary_text,
                    notes=summary_source,
                    note_type=request.form.get("note_type", "Clinical Notes"),
                    patient_name=request.form.get("patient_name"),
                )
            else:
                result["summary"] = {"error": "No source notes available for summary"}

        # If notes are absent and include_codes is true, fall back to transcript
        if include_codes:
            if not notes_text:
                notes_text = result.get("transcript") or ""
            if notes_text:
                futures["medical_codes"] = EXECUTOR.submit(
                    generate_medical_codes, notes_text
                )

        # Collect results
        for key, fut in futures.items():
            try:
                result[key] = fut.result()
            except Exception as exc:
                app.logger.error("%s generation failed: %s", key, exc)
                result[key] = {"error": str(exc)}
    finally:
        pass

    return result, 200


def generate_medical_codes(
    clinical_notes: str,
    *,
    model: Optional[str] = None,
    max_codes: int = 8,
) -> Dict[str, str]:
    """
    Use an LLM to extract relevant medical codes with evidence from clinical notes.

    Returns a flat dict mapping code -> brief evidence snippet.
    Codes may include ICD-10-CM, CPT, and optionally SNOMED CT when clearly supported.
    """
    if not clinical_notes or not clinical_notes.strip():
        return {}

    if _client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export it before calling generate_medical_codes()."
        )

    chosen_model = model or _MODEL_DEFAULT

    sys_prompt = (
        "You are a careful medical coding assistant. Read the clinical notes and "
        "propose only the most relevant standardized medical codes supported by explicit evidence in the text. "
        "Include ICD-10-CM (diagnoses) and CPT (procedures). Optionally include SNOMED CT when clearly supported. "
        "Return a single JSON object mapping each code string to a short evidence quote or summary extracted from the notes. "
        "Only include codes that have direct support in the notes. Use at most "
        f"{max_codes} codes. Do not add any text outside the JSON object."
    )

    user_prompt = (
        "Clinical Notes:\n" + clinical_notes.strip() + "\n\n"
        "Output format (strict JSON object, no markdown, no commentary):\n"
        "{\n  \"<CODE>\": \"<brief evidence from notes in less then 15 words>\",\n  \"<CODE>\": \"<brief evidence from notes in less then 15 words>\"\n}"
    )

    resp = _client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},  # enforce JSON when supported
    )

    raw = resp.choices[0].message.content or "{}"

    try:
        obj = json.loads(raw)
    except Exception:
        # Fallback: attempt to coerce simple key:value lines into JSON
        # If still failing, return empty mapping to avoid crashing callers.
        try:
            trimmed = raw.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                obj = json.loads(trimmed)
            else:
                obj = {}
        except Exception:
            obj = {}

    # Ensure final type is Dict[str, str]
    result: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not k or not isinstance(k, str):
                continue
            if isinstance(v, (str, int, float)):
                result[k] = str(v)
            elif isinstance(v, dict):
                # If model returned richer structure, try to pick an evidence-like field.
                evidence = (
                    v.get("evidence")
                    or v.get("rationale")
                    or v.get("reason")
                    or v.get("support")
                    or ""
                )
                if evidence:
                    result[k] = str(evidence)
            # ignore lists/other types for simplicity

    # Limit to max_codes entries deterministically (sorted by code string)
    if len(result) > max_codes:
        limited = {}
        for code in sorted(result.keys())[:max_codes]:
            limited[code] = result[code]
        result = limited

    return result


@app.route("/transcribe", methods=["POST"])
def transcribe() -> Any:
    payload, status = process_transcription_request(include_notes_default=False)
    return jsonify(payload), status


@app.route("/transcribe_with_notes", methods=["POST"])
def transcribe_with_notes() -> Any:
    payload, status = process_transcription_request(
        include_notes_default=True, force_include_notes=True,force_include_summary = True
    )
    return jsonify(payload), status


@app.route("/api/med_llm/generate_notes", methods=["POST"])
def generate_notes_endpoint() -> Any:
    data = request.get_json(silent=True) or {}
    dialog = data.get("dialog", "")
    narrative = data.get("narrative")

    if not dialog:
        return jsonify({"error": "Missing 'dialog' text"}), 400

    try:
        notes = generate_notes_bundle(dialog, narrative)
        return jsonify({"notes": notes, "note_model": OPENAI_NOTE_MODEL})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/re_generate_summary", methods=["POST"])
def generate_summary_endpoint() -> Any:
    data = request.get_json(silent=True) or {}
    notes = data.get("notes")
    if not notes or not isinstance(notes, list):
        return jsonify({"error": "Missing or invalid 'notes' (expected list)"}), 400

    patient = data.get("patient") or {}
    patient_name = None
    if isinstance(patient, dict):
        patient_name = patient.get("name") or patient.get("Name")

    note_type = data.get("note_type", "Clinical Notes")

    # Prepare a combined text for medical coding (reuse summary sanitization rules)
    sanitized: List[str] = []
    for idx, note in enumerate(notes):
        if isinstance(note, dict):
            text = json.dumps(note, indent=2)
        else:
            text = str(note)
        text = re.sub(r"(?mi)^\s*(Patient Name|Patient|Name)\s*:\s*.*$", "", text)
        text = text.strip()
        if text:
            sanitized.append(f"Note {idx + 1}:\n{text}")
    coding_text = "\n\n".join(sanitized)

    result: Dict[str, Any] = {}
    # Run summary and code extraction concurrently using the shared executor
    fut_summary = EXECUTOR.submit(
        generate_summary_text,
        notes=notes,
        note_type=note_type,
        patient_name=patient_name,
    )
    fut_codes = EXECUTOR.submit(generate_medical_codes, coding_text)

    # Collect results independently so one failure doesn't hide the other
    try:
        result["summary"] = fut_summary.result()
    except Exception as exc:
        result["summary"] = {"error": str(exc)}
    try:
        result["medical_codes"] = fut_codes.result()
    except Exception as exc:
        result["medical_codes"] = {"error": str(exc)}

    return jsonify(result)


# OPTIONS handlers for CORS preflight
@app.route("/transcribe", methods=["OPTIONS"])
def _transcribe_options():  # pragma: no cover
    return ("", 204)


@app.route("/transcribe_with_notes", methods=["OPTIONS"])
def _transcribe_with_notes_options():  # pragma: no cover
    return ("", 204)


@app.route("/api/med_llm/generate_notes", methods=["OPTIONS"])
def _generate_notes_options():  # pragma: no cover
    return ("", 204)


@app.route("/api/med_llm/generate_summary", methods=["OPTIONS"])
def _generate_summary_options():  # pragma: no cover
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
