# AI Clinic

Flask API for medical audio transcription and clinical documentation. It accepts an audio file, transcribes it with OpenAI Whisper, optionally tags speaker roles (Doctor/Patient), generates structured clinical notes (SOAP, DAP, BIRP), produces a visit/discharge summary, and extracts relevant medical codes.

Key endpoints:
- `POST /transcribe` — Transcribe audio, with optional role tags, chunks, words, and codes.
- `POST /transcribe_with_notes` — Transcribe and also generate notes + summary + codes.
- `POST /api/med_llm/generate_notes` — Create SOAP/DAP/BIRP notes from dialog text.
- `POST /re_generate_summary` — Create a summary (and codes) from provided notes.


## Quick Start

- Python 3.10+ recommended
- An OpenAI API key with access to Whisper and chat models

1) Create and activate a virtualenv
   - Windows PowerShell:
     - `python -m venv venv`
     - `venv\Scripts\activate`
   - macOS/Linux:
     - `python3 -m venv venv`
     - `source venv/bin/activate`

2) Install dependencies
   - `pip install -r requirements.txt`

3) Provide your OpenAI key
   - Create a `.env` file in the project root:
     - `OPENAI_API_KEY=sk-...`
   - Or export in shell:
     - Windows PowerShell: `$env:OPENAI_API_KEY = "sk-..."`
     - macOS/Linux: `export OPENAI_API_KEY=sk-...`

4) Run the server
   - `python app.py`
   - The app listens on `http://0.0.0.0:5000` by default (set `PORT` to change).


## Configuration

These environment variables are supported (defaults shown where applicable):

- `OPENAI_API_KEY` (required) — API key used for Whisper and chat models.
- `OPENAI_NOTE_MODEL` (default: `gpt-4o-mini`) — Chat model for note/summary generation.
- `OPENAI_ROLE_MODEL` (default: `gpt-4o-mini`) — Chat model for Doctor/Patient role tagging.
- `CODES_MODEL` (default: `gpt-4.1`) — Chat model for medical code extraction.
- `OPENAI_ASR_TIMEOUT` (default: `120`) — Seconds to wait for Whisper.
- `OPENAI_CHAT_TIMEOUT` (default: `60`) — Seconds to wait for chat endpoints.
- `WORKER_POOL_SIZE` (default: `4`) — Thread pool size for parallel tasks.
- `PORT` (default: `5000`) — Flask server port.

Notes
- CORS is enabled for all origins. `flask-cors` is used when available; a fallback adds headers.
- Response compression uses `flask-compress` when installed (optional).


## API

### POST `/transcribe`

Multipart form-data fields:
- `audio` (file, required) — Audio to transcribe (`.mp3`, `.wav`, etc.)
- `language_code` (string, optional) — BCP‑47 code (e.g., `en-US`). Whisper uses ISO‑639‑1 internally.
- `narrative` (string, optional) — Additional free text context for notes/summary if generated.
- Feature flags (string booleans: `true/false/1/0`):
  - `include_roles` (default: `true`)
  - `include_chunks` (default: `true`)
  - `include_words` (default: `false`)
  - `include_notes` (default: `false`)
  - `include_summary` (default: `false`)
  - `include_codes` (default: `true`)

Response (typical):
- `transcript` — Full transcript text
- `chunks` — Array of chunk strings (when `include_chunks=true`)
- `chunk_details` — Array with per-chunk confidence and role annotations
- `words` — Word timings (when `include_words=true`)
- `dialogue` — Convenience field containing reconstructed dialog text
- `config` — Metadata (model names, language, etc.)
- May also include `notes`, `summary`, and `medical_codes` if requested and/or generated

Example
```
curl -X POST http://localhost:5000/transcribe \
  -F "audio=@sample.wav" \
  -F "language_code=en-US" \
  -F "include_roles=true" \
  -F "include_chunks=true" \
  -F "include_codes=true"
```


### POST `/transcribe_with_notes`

Same inputs as `/transcribe`, but forces notes and summary generation. Returns the transcription plus `notes`, `summary`, and `medical_codes` (when available).

Example
```
curl -X POST http://localhost:5000/transcribe_with_notes \
  -F "audio=@sample.wav" \
  -F "narrative=Patient reports 3 days of cough and fever."
```


### POST `/api/med_llm/generate_notes`

JSON body:
```
{
  "dialog": "Doctor: ...\nPatient: ...",
  "narrative": "optional free text"
}
```

Response:
- `notes` — Object with `SOAP`, `DAP`, and `BIRP` sections (each a dict of labeled fields)
- `note_model` — Model used for generation

Example
```
curl -X POST http://localhost:5000/api/med_llm/generate_notes \
  -H "Content-Type: application/json" \
  -d '{
    "dialog": "Doctor: How are you?\nPatient: I have a cough.",
    "narrative": "Adult with 3 days of dry cough, no SOB."
  }'
```


### POST `/re_generate_summary`

Generate a summary and extract codes from existing notes.

JSON body:
```
{
  "notes": ["...", {"S": "...", "O": "..."}, "..."],
  "patient": {"name": "Jane Doe"},
  "note_type": "Clinical Notes"  // optional label for prompt context
}
```

Response:
- `summary` — Structured visit/discharge summary
- `medical_codes` — Map of code -> brief evidence snippet

Example
```
curl -X POST http://localhost:5000/re_generate_summary \
  -H "Content-Type: application/json" \
  -d '{
    "notes": ["S: Cough\nO: Afebrile\nA: Viral URI\nP: Rest"],
    "patient": {"name": "Jane Doe"},
    "note_type": "Clinical Notes"
  }'
```


## Implementation Notes

- Transcription uses OpenAI Whisper (`whisper-1`) via `POST /v1/audio/transcriptions`.
- Role tagging uses a small LLM with light heuristics to smooth labels.
- Notes and summary use chat completions with `OPENAI_NOTE_MODEL`.
- Medical codes use `CODES_MODEL` and return a JSON object of code -> evidence.
- Concurrency: a shared thread pool runs independent tasks in parallel to reduce latency.


## Troubleshooting

- `OPENAI_API_KEY is not set` — Ensure the key is exported or in `.env`.
- HTTP 401/429 from OpenAI — Check key validity, access, and rate limits.
- Empty transcript — Confirm the uploaded `audio` file is valid and non‑empty.
- Windows path quoting — In `curl`, wrap file paths with quotes: `-F "audio=@\"C:\\path\\to\\file.wav\""`.


## Development

- Run dev server: `python app.py`
- For production, run behind a WSGI server (e.g., gunicorn, gevent) and a reverse proxy; configure timeouts and `WORKER_POOL_SIZE` as needed.
