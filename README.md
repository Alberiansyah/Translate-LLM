# subtitle-translator (Rust)

Web-based subtitle translator (SRT/ASS/SSA/VTT) using a local LLM via
llama-server (OpenAI-compatible API). Byte-faithful round-trip: only the
visible text is translated; timestamps, styles and tags are preserved.

## Run

Dev (from `target/debug` or `target/release`):

```
cargo build --release
cargo run --release
```

Standalone deployment: the `dist/` folder is a self-contained layout
(`dist/subtitle-translator.exe` + `dist/app/static`). Start it with
`dist/run-rust.bat` (default port 8010).

The web UI is at `http://127.0.0.1:<PORT>`.

## LLM server (llama.cpp)

The app is a stateless HTTP client; a llama-server must be running:

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

### Translation quality/efficiency features

| Variable | Default | Meaning |
|---|---|---|
| `TM_ENABLED` | `true` | Translation memory (SQLite `data.db`): exact-match reuse of past translations; identical lines are translated once |
| `CONTEXT_LINES` | `2` | preceding dialogue lines injected into each batch prompt for continuity (0 disables) |
| `ANALYZE_CONTEXT` | `true` | one-shot analysis pass: scans the opening lines and injects a characters/terms/tone brief into the system prompt (1 extra LLM call per file) |
| `ANALYZE_LINES` | `150` | how many opening lines the analysis pass reads |

### Repair toggle (UI)

"Repair Failed Lines" in the settings panel (default **Off**):

- **Off** — fastest (measured parity with the plain Python engine). Lines the
  model fails to translate keep the original text on the first run; rerunning
  the same file self-heals them via the translation memory.
- **On (default)** — adds iterative repair rounds (batch retry → per-line
  retry with a temperature swing), pushing first-run coverage to ~100% at the
  cost of roughly 20% extra time, depending on how many lines the model
  misses.

## Data locations

The base directory is the exe's directory when it contains `app/static`
(the `dist/` layout), otherwise the repo parent. Base directory contains:
`data.db` (history + translation memory), `uploads/` (temporary),
`output/` (translated files).
