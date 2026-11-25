# Subtitle Translator (TranslateLLM — Rust)

Web-based subtitle translator (SRT / ASS / SSA / VTT) powered by a local LLM
served through **llama-server** (llama.cpp's OpenAI-compatible API).
Byte-faithful round-trip: only the visible text is translated — timestamps,
styles, markup and cue layout are preserved exactly.

## Features

- **Byte-faithful subtitle handling** — SRT/ASS/SSA/VTT in, same format out.
- **Line-level dedupe** — identical lines are sent to the LLM once and mapped
  back to every position that uses them.
- **Translation memory (SQLite)** — exact-match cache of past translations
  (`data.db`). Rerunning the same file is ~7x faster. Untrustworthy entries
  (echoes of the source text) are never served; they self-heal: the line is
  re-translated and the cache entry is replaced.
- **Context analysis pass** — one cheap LLM call per file scans the opening
  lines and injects a short brief (characters / places / tone) into the system
  prompt for consistent naming across the whole file.
- **Cross-batch context** — each batch prompt carries the preceding dialogue
  lines so continuity survives batch boundaries.
- **Opt-in repair pass** — failed/echoed lines are retried iteratively (batch
  retry → per-line retry with a temperature swing) toward ~100% first-run
  coverage.
- **Job history** — past jobs listed in the UI with re-download.

## Build & run

Requirements: Rust (2021 edition) and a running llama-server.

```
cargo build --release
cargo run --release
```

The web UI is at `http://127.0.0.1:<PORT>` (default `8000`).

### Standalone deployment

`dist/` is a self-contained layout (`dist/subtitle-translator.exe` +
`dist/app/static`). Start it with `dist/run-rust.bat` (port 8010). To rebuild
it after code changes:

```
cargo build --release
copy target\release\subtitle-translator.exe dist\
```

## LLM server (llama.cpp)

The app is a stateless HTTP client; a llama-server must be running, e.g.:

```
llama-server -m <model.gguf> --host 127.0.0.1 --port 8080
```

Recommended flags for this app:

```
--parallel <N>        # match LLM_CONCURRENCY (default 1). Concurrency above 1
                      # is only useful if the server has parallel slots.
--cache-ram <MiB>     # host-memory prompt cache. Prompt caches are slot-local
                      # under --parallel > 1 (llama.cpp issues #22942/#32568);
                      # --cache-ram lets slots reuse each other's computed
                      # prefixes instead of re-evaluating the shared system
                      # prompt on every batch.
```

The app already sends `reasoning_effort: "none"` so thinking-mode models
(Qwen3.x) do not burn the token budget on reasoning.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `LLM_MODEL_NAME` | `local-llm` | model id sent to the API |
| `LLM_TIMEOUT` | `120` | per-request timeout (s) |
| `LLM_CONNECT_TIMEOUT` | `5` | connect timeout (s) |
| `LLM_MAX_RETRIES` | `1` | retries per batch (with backoff) |
| `LLM_CONCURRENCY` | `1` | batches in flight (needs server `--parallel`) |
| `BATCH_SIZE` | `15` | lines per LLM request (UI-overridable, max 100) |
| `TEMPERATURE` | `0.3` | sampling temperature |
| `TOP_P` | `0.9` | nucleus sampling |
| `MAX_TOKENS` | `2048` | generation budget per batch |
| `MAX_UPLOAD_BYTES` | `52428800` | per-file upload limit |
| `JOB_TTL_SECONDS` | `3600` | in-memory job retention |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | HTTP bind |

### Quality / efficiency toggles

| Variable | Default | Meaning |
|---|---|---|
| `TM_ENABLED` | `true` | Translation memory (SQLite `data.db`): exact-match reuse of past translations; identical lines are translated once. Echo entries self-heal on rerun |
| `CONTEXT_LINES` | `2` | preceding dialogue lines injected into each batch prompt for continuity (0 disables) |
| `ANALYZE_CONTEXT` | `true` | one-shot analysis pass: scans the opening lines and injects a characters/places/tone brief into the system prompt (1 extra LLM call per file, ~0.5s) |
| `ANALYZE_LINES` | `150` | how many opening lines the analysis pass reads |

### Repair toggle (UI setting, default **On**)

"Repair Failed Lines" in the settings panel:

- **On (default)** — adds iterative repair rounds: numbered batch retry, then
  per-line retry with a temperature swing (0.2 → 0.8) to break echo loops.
  Pushes first-run coverage to ~100%; costs roughly 20% extra time depending
  on how many lines the model misses.
- **Off** — fastest (measured parity with the plain Python engine, e.g. 85s
  vs 86s on a 927-line file). Lines the model fails to translate keep the
  original text on the first run; rerunning the same file self-heals them via
  the translation memory.

## Data locations

The base directory is the exe's directory when it contains `app/static` (the
`dist/` layout), otherwise the repo parent directory. The base directory
contains:

| Path | Contents |
|---|---|
| `data.db` | job history + translation memory (SQLite, created on first run) |
| `uploads/` | temporary uploaded files |
| `output/` | translated files (kept for re-download) |

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/translate` | POST | translate one uploaded file (multipart: `file`, `source_lang`, `target_lang`, `batch_size`, `max_tokens`, `glossary`, `repair`) |
| `/api/translate-batch` | POST | translate multiple files |
| `/api/progress/{job}` | GET | progress polling (status, percent, batches, lines, speed) |
| `/api/download/{job}` | GET | download the translated file |
| `/api/cancel/{job}` | POST | cancel a running job |
| `/api/history` | GET | job history |
| `/api/languages` | GET | supported language list (107) |
| `/api/llm-status` | GET | llama-server reachability + model info |

## Project structure

```
src/
  main.rs        startup, router, static file serving
  api.rs         HTTP endpoints, multipart parsing, job registry
  pipeline.rs    orchestration: dedupe → TM lookup → batching → LLM → save
  llm.rs         prompt building, chat completions, retries, repair loop
  batcher.rs     batch planning + per-batch context windows
  db.rs          history + translation memory (rusqlite)
  glossary.rs    user glossary parsing/validation
  langs.rs       language list (107 languages)
  resolver.rs    pronoun-resolution context helper
  config.rs      env-var settings
  state.rs       shared app state
  subtitle/      subtitle parsing/writing (SRT/ASS/SSA/VTT, tag-safe)
app/static/      web UI (vanilla HTML/JS/CSS)
```
