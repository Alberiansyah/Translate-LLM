# Subtitle Translator

Subtitle translator for SRT, ASS, SSA and VTT files, running on a local LLM
through llama-server (llama.cpp's OpenAI-compatible API). Only the spoken
text gets translated. Timestamps, styles and tags stay untouched, so the
output drops straight back into your video player.

This is a Rust rewrite of the same tool.

## How it works

The app batches subtitle lines into numbered requests, but it tries hard not
to send more to the LLM than it has to:

1. Identical lines are only translated once (dedupe).
2. Every translated line is stored in a SQLite translation memory, so
   re-translating the same file is mostly cache hits.
3. A short one-shot pass reads the first ~150 lines of the file and writes a
   small brief (character names, places, tone) that gets added to the system
   prompt. Without it a 2B model happily renames your protagonist halfway
   through the episode.
4. Each batch also gets the couple of dialogue lines right before it, so the
   translator knows what the conversation was about.

## Build and run

You need Rust and a running llama-server.

```
cargo build --release
cargo run --release
```

Then open http://127.0.0.1:8000 in the browser.

### llama-server flags worth knowing

```
llama-server -m model.gguf --host 127.0.0.1 --port 8080
```

- `--parallel N`: only useful if you also raise `LLM_CONCURRENCY`. With one
  slot, keep both at 1.
- `--cache-ram`: prompt caches are per-slot, so with `--parallel` above 1 the
  shared system prompt gets recomputed per slot unless you give it some host
  memory to share prefixes (llama.cpp issues #22942/#32568).

The app sends `reasoning_effort: "none"`, so thinking-mode models (Qwen3.x)
don't waste the token budget on reasoning.

## The "Repair failed lines" option (on by default)

Small local models have one annoying habit: sometimes they just copy the
source line back instead of translating it. With this option on, the app
notices and retries. First it re-asks for the missing lines as a batch, then
one line per request, and the last attempt bumps the sampling temperature,
because the same prompt at the same temperature tends to produce the same
wrong answer.

This pushes first-run coverage to basically 100% and costs around 20% extra
time, depending on how many lines the model misses.

Turn it off if you care about speed. A plain run skips all of that retry
machinery, which saves roughly 20% on the first translation of a file. Lines
the model missed stay untranslated for that run, but the next run of the same
file fixes them automatically: the translation memory refuses to serve cache
entries that still look like the source text, re-translates them, and
replaces them.

## Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080` | llama-server address |
| `LLM_MODEL_NAME` | `local-llm` | model id sent to the API |
| `LLM_TIMEOUT` | `120` | per-request timeout (s) |
| `LLM_CONNECT_TIMEOUT` | `5` | connect timeout (s) |
| `LLM_MAX_RETRIES` | `1` | retries per batch, with backoff |
| `LLM_CONCURRENCY` | `1` | batches in flight (needs server `--parallel`) |
| `BATCH_SIZE` | `15` | lines per request, UI-overridable, max 100 |
| `TEMPERATURE` | `0.3` | sampling temperature |
| `TOP_P` | `0.9` | nucleus sampling |
| `MAX_TOKENS` | `2048` | generation budget per batch |
| `MAX_UPLOAD_BYTES` | `52428800` | per-file upload limit |
| `JOB_TTL_SECONDS` | `3600` | how long finished jobs stay in memory |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | HTTP bind |
| `TM_ENABLED` | `true` | translation memory on/off |
| `CONTEXT_LINES` | `2` | dialogue lines sent as context before each batch |
| `ANALYZE_CONTEXT` | `true` | the opening-lines brief (1 extra call per file) |
| `ANALYZE_LINES` | `150` | how many opening lines the brief pass reads |

## Deployment

`dist/` is a self-contained folder (exe + the static UI) that can be dropped
anywhere. Start it with `run-rust.bat`, it listens on port 8010. After code
changes rebuild and copy the exe over:

```
cargo build --release
copy target\release\subtitle-translator.exe dist\
```

Everything the app writes lives next to the exe: `data.db` (history +
translation memory), `uploads/` for temporary files and `output/` for the
translated ones. When you run from `target/` instead, it falls back to the
parent directory of the repo.

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/translate` | POST | translate one file (multipart) |
| `/api/translate-batch` | POST | translate several files |
| `/api/progress/{job}` | GET | progress polling |
| `/api/download/{job}` | GET | download the result |
| `/api/cancel/{job}` | POST | cancel a running job |
| `/api/history` | GET | past jobs |
| `/api/languages` | GET | supported languages (107) |
| `/api/llm-status` | GET | is llama-server up, which model |

The translate endpoints take `file`, `source_lang`, `target_lang`,
`batch_size`, `max_tokens`, `glossary` (a JSON array of
`{"source": "...", "target": "..."}`) and `repair` (`0` or `1`).
