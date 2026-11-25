//! REST API, port of app/api/routes.py. All error bodies are
//! `{"detail": "<message>"}` with the Python status codes because the
//! frontend (app/static/app.js) reads `err.detail`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tracing::{error, info};

use crate::batcher::ContextBatcher;
use crate::config::Settings;
use crate::db::{self, HistoryRow};
use crate::glossary::GlossaryEntry;
use crate::langs;
use crate::models::{
    round1, round2, HistoryEntry, LanguageInfo, ProgressResponse, TranslateResponse,
};
use crate::pipeline::{Pipeline, TranslationProgress};
use crate::state::{cleanup_jobs, AppState, FileMeta, Job};
use crate::subtitle::{load_subtitle, save_subtitle, SubtitleFile};

// ---------------------------------------------------------------------------
// Error type: every failure is JSON {"detail": ...} like FastAPI HTTPException
// ---------------------------------------------------------------------------

pub struct ApiError {
    pub status: StatusCode,
    pub detail: String,
}

impl ApiError {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> ApiError {
        ApiError { status, detail: detail.into() }
    }
    pub fn bad(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, detail)
    }
    pub fn not_found(detail: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::NOT_FOUND, detail)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(serde_json::json!({ "detail": self.detail }))).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/languages", get(get_languages))
        .route("/api/llm-status", get(get_llm_status))
        .route("/api/translate", post(start_translation))
        .route("/api/translate-batch", post(start_batch_translation))
        .route("/api/progress/{job_id}", get(get_progress))
        .route("/api/cancel/{job_id}", post(cancel_translation))
        .route("/api/download/{job_id}", get(download_translation))
        .route("/api/download-batch/{job_id}", get(download_batch))
        .route("/api/history", get(list_history))
        .route("/api/history/{job_id}/download", get(download_history))
        .route("/api/history/{job_id}", delete(delete_history_entry))
}

// ---------------------------------------------------------------------------
// Multipart collection
// ---------------------------------------------------------------------------

struct UploadedPart {
    filename: String,
    bytes: Vec<u8>,
}

/// Collected multipart form: file parts grouped by field name, text fields
/// by name (first occurrence wins; missing fields fall back to defaults).
struct FormData {
    files: HashMap<String, Vec<UploadedPart>>,
    text: HashMap<String, String>,
}

async fn read_multipart(mut mp: Multipart) -> ApiResult<FormData> {
    let mut files: HashMap<String, Vec<UploadedPart>> = HashMap::new();
    let mut text: HashMap<String, String> = HashMap::new();
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| ApiError::bad(format!("Invalid multipart form: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if field.file_name().is_some() {
            let filename = field.file_name().unwrap_or("").to_string();
            let bytes = field
                .bytes()
                .await
                .map_err(|e| ApiError::bad(format!("Failed to read upload: {e}")))?;
            files.entry(name).or_default().push(UploadedPart {
                filename,
                bytes: bytes.to_vec(),
            });
        } else {
            let value = field
                .text()
                .await
                .map_err(|e| ApiError::bad(format!("Failed to read form field: {e}")))?;
            text.entry(name).or_insert(value);
        }
    }
    Ok(FormData { files, text })
}

// ---------------------------------------------------------------------------
// Validation / lifecycle helpers (port of routes.py helpers)
// ---------------------------------------------------------------------------

fn validate_languages(source_lang: &str, target_lang: &str) -> ApiResult<()> {
    if !langs::is_valid_lang_name(source_lang) {
        return Err(ApiError::bad(format!("Unsupported source_lang: {source_lang}")));
    }
    if !langs::is_valid_lang_name(target_lang) {
        return Err(ApiError::bad(format!("Unsupported target_lang: {target_lang}")));
    }
    Ok(())
}

fn validate_params(
    batch_size: i64,
    temperature: f64,
    max_tokens: i64,
    top_p: f64,
) -> ApiResult<()> {
    let max_lines = Settings::global().max_lines_per_batch;
    if batch_size < 1 || batch_size > max_lines as i64 {
        return Err(ApiError::bad(format!(
            "batch_size must be between 1 and {max_lines}"
        )));
    }
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::bad("temperature must be between 0.0 and 2.0"));
    }
    if top_p <= 0.0 || top_p > 1.0 {
        return Err(ApiError::bad("top_p must be greater than 0.0 and at most 1.0"));
    }
    if max_tokens < 1 || max_tokens > 32768 {
        return Err(ApiError::bad("max_tokens must be between 1 and 32768"));
    }
    Ok(())
}

/// Parse the glossary form field; empty array / empty string -> None.
fn parse_glossary(raw: &str) -> ApiResult<Option<Vec<GlossaryEntry>>> {
    if raw.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|_| ApiError::bad("Invalid glossary"))?;
    let arr = value.as_array().ok_or_else(|| ApiError::bad("Invalid glossary"))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = match item.as_object() {
            Some(o) => o,
            None => return Err(ApiError::bad("Invalid glossary")),
        };
        let source = match obj.get("source").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return Err(ApiError::bad("Invalid glossary")),
        };
        let target = match obj.get("target").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return Err(ApiError::bad("Invalid glossary")),
        };
        let case_sensitive = obj
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        out.push(GlossaryEntry { source, target, case_sensitive });
    }
    if out.is_empty() {
        return Ok(None);
    }
    Ok(Some(out))
}

/// Port of routes._check_upload. Returns (safe_filename, extension-with-dot).
fn check_upload(filename: &str, content: &[u8]) -> ApiResult<(String, String)> {
    let safe_name = Path::new(filename)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Path::suffix parity: dot included, lowercased; empty when no extension.
    let ext = Path::new(&safe_name)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    let settings = Settings::global();
    if !settings.allowed_extensions.contains(&ext.as_str()) {
        return Err(ApiError::bad(format!(
            "Unsupported format: {ext}. Use SRT, ASS, SSA, or VTT."
        )));
    }
    if content.len() as u64 > settings.max_upload_bytes {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("File too large (max {} bytes)", settings.max_upload_bytes),
        ));
    }
    Ok((safe_name, ext))
}

/// uuid.uuid4().hex[:12] parity: first 12 chars of the v4 simple hex.
fn new_job_id() -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    hex[..12].to_string()
}

fn text_or<'a>(text: &'a HashMap<String, String>, key: &str, default: &'a str) -> String {
    match text.get(key) {
        Some(v) if !v.is_empty() => v.clone(),
        _ => default.to_string(),
    }
}

/// Parsed + validated translation parameters shared by both endpoints.
struct Params {
    source_lang: String,
    target_lang: String,
    batch_size: usize,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    glossary: Option<Vec<GlossaryEntry>>,
    /// Repair-on-failure toggle (UI checkbox). Adds retry rounds that push
    /// coverage toward 100% at the cost of extra LLM requests. Default: on.
    repair: bool,
}

impl Params {
    /// Defaults mirror the Form(...) defaults in routes.py. Malformed numeric
    /// fields fall back to the default (Python would return 422; divergence
    /// documented — the bundled frontend always sends valid numbers).
    fn from_form(text: &HashMap<String, String>) -> ApiResult<Params> {
        let source_lang = text_or(text, "source_lang", "English");
        let target_lang = text_or(text, "target_lang", "Indonesian");
        let batch_size = text
            .get("batch_size")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(15);
        let temperature = text
            .get("temperature")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.3);
        let max_tokens = text
            .get("max_tokens")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(2048);
        let top_p = text
            .get("top_p")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.9);
        let glossary_raw = text_or(text, "glossary", "[]");
        // Checkbox fields arrive as "1"/"true"/"on" when checked, and are
        // typically omitted entirely when unchecked. Absent -> true (repair
        // on) so older clients still get max coverage by default.
        let repair = text
            .get("repair")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "on" || v == "yes"
            })
            .unwrap_or(true);
        validate_languages(&source_lang, &target_lang)?;
        validate_params(batch_size, temperature, max_tokens, top_p)?;
        let glossary = parse_glossary(&glossary_raw)?;
        Ok(Params {
            source_lang,
            target_lang,
            batch_size: batch_size as usize,
            temperature,
            max_tokens: max_tokens as u32,
            top_p,
            glossary,
            repair,
        })
    }
}

// ---------------------------------------------------------------------------
// Simple GET endpoints
// ---------------------------------------------------------------------------

async fn get_languages() -> Json<Vec<LanguageInfo>> {
    Json(
        langs::LANGUAGES
            .iter()
            .map(|(c, n)| LanguageInfo { code: c.to_string(), name: n.to_string() })
            .collect(),
    )
}

async fn get_llm_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status = state.status_engine.check_connection().await;
    Json(serde_json::json!({
        "connected": status.connected,
        "model": status.model,
        "base_url": status.base_url,
        "error": status.error,
    }))
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn attachment_response(
    bytes: Vec<u8>,
    filename: &str,
    content_type: &'static str,
) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\"")),
        ],
        bytes,
    )
        .into_response()
}

/// Build a deflate ZIP in memory; entries are (entry_name, bytes).
fn build_zip(entries: &[(String, Vec<u8>)]) -> Result<Vec<u8>, String> {
    let buf = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(buf);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in entries {
        zip.start_file(name.as_str(), options.clone())
            .map_err(|e| e.to_string())?;
        std::io::Write::write_all(&mut zip, data).map_err(|e| e.to_string())?;
    }
    let inner = zip.finish().map_err(|e| e.to_string())?;
    Ok(inner.into_inner())
}

// ---------------------------------------------------------------------------
// GET /api/progress/{job_id}
// ---------------------------------------------------------------------------

async fn get_progress(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Json<ProgressResponse>> {
    let jobs = state.jobs.lock().await;
    let job = jobs.get(&job_id).ok_or_else(|| ApiError::not_found("Job not found"))?;

    let (p_status, p_percent, p_total_batches, p_completed_batches, p_error, p_last_batch, p_device, p_total_files, p_completed_files, p_current_file, p_completed_lines) = {
        let p = job.progress.lock().unwrap();
        (
            p.status.clone(),
            p.percent(),
            p.total_batches as i64,
            p.completed_batches as i64,
            p.error.clone(),
            p.last_batch_seconds,
            p.device_used.clone(),
            p.total_files as i64,
            p.completed_files as i64,
            p.current_file.clone(),
            p.completed_lines as i64,
        )
    };
    let total_lines = job.total_lines as i64;

    // Port of the routes.py math.
    let elapsed = if p_status != "idle" && p_status != "queued" {
        round2(job.start_time.elapsed().as_secs_f64())
    } else {
        0.0
    };

    let completed_lines = if p_status == "completed" { total_lines } else { p_completed_lines };

    let lps = if elapsed > 0.0 && completed_lines > 0 {
        round1(completed_lines as f64 / elapsed)
    } else {
        0.0
    };
    let remaining = (total_lines - completed_lines).max(0);
    let eta = if lps > 0.0 && remaining > 0 {
        round1(remaining as f64 / lps)
    } else {
        0.0
    };

    Ok(Json(ProgressResponse {
        job_id,
        status: p_status,
        percent: p_percent,
        total_batches: p_total_batches,
        completed_batches: p_completed_batches,
        error: p_error,
        elapsed_seconds: elapsed,
        last_batch_seconds: p_last_batch,
        device_used: p_device,
        lines_per_second: lps,
        total_files: p_total_files,
        completed_files: p_completed_files,
        current_file: p_current_file,
        total_lines,
        completed_lines,
        eta_seconds: eta,
    }))
}

// ---------------------------------------------------------------------------
// POST /api/cancel/{job_id}
// ---------------------------------------------------------------------------

async fn cancel_translation(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let jobs = state.jobs.lock().await;
    let job = jobs.get(&job_id).ok_or_else(|| ApiError::not_found("Job not found"))?;

    let status = job.progress.lock().unwrap().status.clone();
    if status == "completed" || status == "failed" || status == "cancelled" {
        return Err(ApiError::bad(format!("Job already {status}")));
    }

    job.cancel.store(true, Ordering::SeqCst);
    info!("Cancellation requested for job {job_id}");
    Ok(Json(serde_json::json!({ "status": "cancelling", "job_id": job_id })))
}

// ---------------------------------------------------------------------------
// GET /api/download/{job_id}
// ---------------------------------------------------------------------------

async fn download_translation(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Response> {
    let (is_batch, files) = {
        let jobs = state.jobs.lock().await;
        let job = jobs.get(&job_id).ok_or_else(|| ApiError::not_found("Job not found"))?;
        (job.is_batch, job.files.clone())
    };

    if is_batch {
        return Err(ApiError::bad(format!(
            "This is a batch job. Use /api/download-batch/{job_id}"
        )));
    }

    let meta = files
        .first()
        .ok_or_else(|| ApiError::not_found("Translated file not ready yet"))?;
    let bytes = tokio::fs::read(&meta.output_path)
        .await
        .map_err(|_| ApiError::not_found("Translated file not ready yet"))?;

    let stem = Path::new(&meta.original_filename)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| meta.original_filename.clone());
    let ext = meta
        .output_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();

    Ok(attachment_response(
        bytes,
        &format!("{stem}{ext}"),
        "application/octet-stream",
    ))
}

// ---------------------------------------------------------------------------
// GET /api/download-batch/{job_id}
// ---------------------------------------------------------------------------

async fn download_batch(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Response> {
    let (is_batch, files) = {
        let jobs = state.jobs.lock().await;
        let job = jobs.get(&job_id).ok_or_else(|| ApiError::not_found("Job not found"))?;
        (job.is_batch, job.files.clone())
    };

    if !is_batch {
        return Err(ApiError::bad("Not a batch job"));
    }

    // ZIP of existing outputs; entry name = original filename (basename).
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for meta in &files {
        if let Ok(bytes) = tokio::fs::read(&meta.output_path).await {
            let name = Path::new(&meta.original_filename)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| meta.original_filename.clone());
            entries.push((name, bytes));
        }
    }
    let zip_bytes = build_zip(&entries)
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(attachment_response(
        zip_bytes,
        &format!("translated_{job_id}.zip"),
        "application/zip",
    ))
}

// ---------------------------------------------------------------------------
// POST /api/translate-batch
// ---------------------------------------------------------------------------

async fn start_batch_translation(
    State(state): State<AppState>,
    multipart: Multipart,
) -> ApiResult<Json<TranslateResponse>> {
    let mut form = read_multipart(multipart).await?;
    let params = Params::from_form(&form.text)?;

    // Frontend always uses repeated `files` fields (even for 1 file).
    let uploads = form.files.remove("files").unwrap_or_default();
    if uploads.is_empty() {
        return Err(ApiError::bad("At least 1 file required"));
    }

    // Parity: cleanup runs after the file-count check.
    cleanup_jobs(&state).await;

    let settings = Settings::global();
    let job_id = new_job_id();

    let mut metas: Vec<FileMeta> = Vec::with_capacity(uploads.len());
    let mut subs: Vec<SubtitleFile> = Vec::with_capacity(uploads.len());

    for upload in uploads {
        let (safe_name, ext) = check_upload(&upload.filename, &upload.bytes)?;

        let file_id = new_job_id();
        let upload_path = settings.upload_dir.join(format!("{file_id}{ext}"));
        let output_path = settings.output_dir.join(format!("{file_id}_translated{ext}"));

        tokio::task::spawn_blocking({
            let bytes = upload.bytes.clone();
            let path = upload_path.clone();
            move || std::fs::write(&path, bytes)
        })
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let parsed = tokio::task::spawn_blocking({
            let path = upload_path.clone();
            move || {
                let result = load_subtitle(&path);
                let _ = std::fs::remove_file(&path);
                result
            }
        })
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let sub = parsed
            .map_err(|e| ApiError::bad(format!("Failed to parse {safe_name}: {e}")))?;

        subs.push(sub);
        metas.push(FileMeta {
            original_filename: safe_name,
            output_path,
        });
    }

    let total_lines: usize = subs.iter().map(|s| s.line_count()).sum();

    let mut progress = TranslationProgress::default();
    progress.status = "queued".to_string();

    let job = Job {
        progress: Arc::new(Mutex::new(progress)),
        cancel: Arc::new(AtomicBool::new(false)),
        is_batch: true,
        files: metas,
        source_lang: params.source_lang.clone(),
        target_lang: params.target_lang.clone(),
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        top_p: params.top_p,
        start_time: Instant::now(),
        started_at: db::now_isoformat(),
        total_lines,
    };

    state.jobs.lock().await.insert(job_id.clone(), job);

    // Message uses the sanitized names, in submission order.
    let display_names: Vec<String> = {
        let jobs = state.jobs.lock().await;
        jobs.get(&job_id)
            .map(|j| j.files.iter().map(|f| f.original_filename.clone()).collect())
            .unwrap_or_default()
    };
    let mut names = display_names.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
    if display_names.len() > 3 {
        names += &format!(" +{} more", display_names.len() - 3);
    }
    let message = format!("Batch translation started: {} files ({names})", subs.len());

    let task_state = state.clone();
    let task_job_id = job_id.clone();
    tokio::spawn(async move {
        run_batch_translation(task_state, task_job_id, subs, params).await;
    });

    Ok(Json(TranslateResponse { job_id, status: "queued".to_string(), message }))
}

/// Port of routes._run_batch_translation (batch background task).
///
/// Divergence (documented): Python composes per-batch progress through an
/// on_progress callback; the Rust pipeline interface only accepts a shared
/// progress Arc, so per-batch interleave is not reproduced. Final states are
/// exact: per-file status/completed_files/current_file are set before each
/// file starts, and completed_batches/completed_lines are corrected after
/// each file completes.
async fn run_batch_translation(
    state: AppState,
    job_id: String,
    subs: Vec<SubtitleFile>,
    params: Params,
) {
    let settings = Settings::global();
    let (progress, cancel, start_time, started_at, total_lines, metas) = {
        let jobs = state.jobs.lock().await;
        match jobs.get(&job_id) {
            Some(job) => (
                job.progress.clone(),
                job.cancel.clone(),
                job.start_time,
                job.started_at.clone(),
                job.total_lines,
                job.files.clone(),
            ),
            None => return,
        }
    };

    // Per-file batch ESTIMATES (dedupe/TM shrink the real count; used only
    // for the progress denominator and per-file offsets).
    let estimated_batches_per_file: Vec<usize> = subs
        .iter()
        .map(|sub| {
            ContextBatcher::new(params.batch_size)
                .create_batches(&sub.lines)
                .len()
        })
        .collect();
    let estimated_total = estimated_batches_per_file.iter().sum::<usize>().max(1);

    let pipeline = Pipeline::new();
    let mut accumulated_lines: usize = 0;
    let mut accumulated_batches: usize = 0;

    for (i, sub) in subs.into_iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            let mut p = progress.lock().unwrap();
            *p = TranslationProgress::default();
            p.status = "cancelled".to_string();
            return;
        }

        let completed_before: usize = accumulated_batches.min(estimated_total);
        let fname = metas
            .get(i)
            .map(|m| m.original_filename.clone())
            .unwrap_or_default();
        let n_files = metas.len();

        {
            let mut p = progress.lock().unwrap();
            p.status = format!("({}/{n_files}) {fname}", i + 1);
            p.total_batches = estimated_total;
            p.completed_batches = completed_before;
            p.total_files = n_files;
            p.completed_files = i;
            p.current_file = fname.clone();
        }

        let output = pipeline
            .translate(
                &sub,
                &params.source_lang,
                &params.target_lang,
                params.glossary.as_deref(),
                params.batch_size,
                params.temperature,
                params.max_tokens,
                params.top_p,
                params.repair,
                progress.clone(),
                cancel.clone(),
            )
            .await;

        if cancel.load(Ordering::SeqCst) {
            let mut p = progress.lock().unwrap();
            *p = TranslationProgress::default();
            p.status = "cancelled".to_string();
            return;
        }

        let file_completed_lines = sub.line_count();
        let save_path = metas[i].output_path.clone();
        let save_result = tokio::task::spawn_blocking(move || {
            save_subtitle(&sub, &output.texts, &save_path)
        })
        .await;

        let save_result = match save_result {
            Ok(r) => r,
            Err(e) => Err(e.to_string()),
        };

        match save_result {
            Ok(_) => {
                accumulated_lines += file_completed_lines;
                accumulated_batches += output.batches_done;
                let mut p = progress.lock().unwrap();
                p.completed_batches = accumulated_batches.min(estimated_total);
                p.completed_lines = accumulated_lines;
            }
            Err(e) => {
                error!("Batch file {fname} failed: {e}");
                let mut p = progress.lock().unwrap();
                *p = TranslationProgress::default();
                p.status = "failed".to_string();
                p.error = Some(e);
                return;
            }
        }
    }

    // Final completed state (exact port).
    {
        let mut p = progress.lock().unwrap();
        *p = TranslationProgress::default();
        p.status = "completed".to_string();
        p.total_batches = estimated_total;
        p.completed_batches = estimated_total;
        p.total_files = metas.len();
        p.completed_files = metas.len();
        p.completed_lines = total_lines;
    }

    let elapsed = round2(start_time.elapsed().as_secs_f64());
    let lps = if elapsed > 0.0 {
        round1(total_lines as f64 / elapsed)
    } else {
        0.0
    };
    let row = HistoryRow {
        id: 0,
        job_id: job_id.clone(),
        source_lang: params.source_lang.clone(),
        target_lang: params.target_lang.clone(),
        is_batch: 1,
        filenames: metas.iter().map(|m| m.original_filename.clone()).collect(),
        output_paths: metas
            .iter()
            .map(|m| m.output_path.to_string_lossy().into_owned())
            .collect(),
        total_lines: total_lines as i64,
        completed_lines: total_lines as i64,
        device: "local-llm".to_string(),
        device_used: "local-llm".to_string(),
        temperature: params.temperature,
        max_tokens: params.max_tokens as i64,
        top_p: params.top_p,
        status: "completed".to_string(),
        elapsed_seconds: elapsed,
        lines_per_second: lps,
        created_at: started_at,
        completed_at: db::now_isoformat(),
    };
    if let Err(e) = tokio::task::spawn_blocking(move || db::save_history(&row)).await {
        error!("Failed to save history for job {job_id}: {e}");
    }
    let _ = settings;
}

// ---------------------------------------------------------------------------
// GET /api/history
// ---------------------------------------------------------------------------

async fn list_history(State(_state): State<AppState>) -> ApiResult<Json<Vec<HistoryEntry>>> {
    let rows = tokio::task::spawn_blocking(|| db::get_history(50))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let entries = rows
        .into_iter()
        .map(|r| HistoryEntry {
            job_id: r.job_id,
            source_lang: r.source_lang,
            target_lang: r.target_lang,
            is_batch: r.is_batch != 0,
            filenames: r.filenames,
            output_paths: r.output_paths,
            total_lines: r.total_lines,
            completed_lines: r.completed_lines,
            device: r.device,
            device_used: r.device_used,
            temperature: r.temperature,
            max_tokens: r.max_tokens,
            top_p: r.top_p,
            status: r.status,
            elapsed_seconds: r.elapsed_seconds,
            lines_per_second: r.lines_per_second,
            created_at: r.created_at,
            completed_at: r.completed_at,
        })
        .collect();
    Ok(Json(entries))
}

// ---------------------------------------------------------------------------
// GET /api/history/{job_id}/download
// ---------------------------------------------------------------------------

async fn download_history(
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Response> {
    let entry = tokio::task::spawn_blocking(move || db::get_history_entry(&job_id))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .ok_or_else(|| ApiError::not_found("History entry not found"))?;

    if entry.output_paths.len() == 1 {
        let p = std::path::PathBuf::from(&entry.output_paths[0]);
        let bytes = tokio::fs::read(&p)
            .await
            .map_err(|_| ApiError::not_found("Output file no longer exists on disk"))?;
        let orig_stem = entry
            .filenames
            .first()
            .and_then(|f| Path::new(f).file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        let ext = p
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        return Ok(attachment_response(
            bytes,
            &format!("{orig_stem}{ext}"),
            "application/octet-stream",
        ));
    }

    // Multiple outputs: ZIP of the files that still exist, entry name =
    // original filename (Path(fn).name parity).
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (op, fn_) in entry.output_paths.iter().zip(entry.filenames.iter()) {
        if let Ok(bytes) = tokio::fs::read(op).await {
            let name = Path::new(fn_)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| fn_.clone());
            entries.push((name, bytes));
        }
    }
    let zip_bytes = build_zip(&entries)
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(attachment_response(
        zip_bytes,
        &format!("translated_{}.zip", entry.job_id),
        "application/zip",
    ))
}

// ---------------------------------------------------------------------------
// DELETE /api/history/{job_id}
// ---------------------------------------------------------------------------

async fn delete_history_entry(
    AxumPath(job_id): AxumPath<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let entry = tokio::task::spawn_blocking({
        let job_id = job_id.clone();
        move || db::get_history_entry(&job_id)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .ok_or_else(|| ApiError::not_found("History entry not found"))?;

    // Delete output files if present; errors ignored (parity: unlink may fail).
    for op in &entry.output_paths {
        if Path::new(op).exists() {
            let _ = tokio::fs::remove_file(op).await;
        }
    }

    let deleted = tokio::task::spawn_blocking({
        let job_id = job_id.clone();
        move || db::delete_history(&job_id)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    if !deleted {
        // Row vanished between the lookup and the delete; still report 404.
        return Err(ApiError::not_found("History entry not found"));
    }

    Ok(Json(serde_json::json!({ "status": "deleted", "job_id": job_id })))
}

// ---------------------------------------------------------------------------
// POST /api/translate (single file)
// ---------------------------------------------------------------------------

async fn start_translation(
    State(state): State<AppState>,
    multipart: Multipart,
) -> ApiResult<Json<TranslateResponse>> {
    // Parity: cleanup runs first.
    cleanup_jobs(&state).await;

    let mut form = read_multipart(multipart).await?;
    let params = Params::from_form(&form.text)?;

    let uploads = form.files.remove("file").unwrap_or_default();
    let upload = match uploads.into_iter().next() {
        Some(u) => u,
        // Python's required File(...) returns 422; 400 keeps err.detail usable.
        None => return Err(ApiError::bad("No file provided")),
    };

    let (safe_name, ext) = check_upload(&upload.filename, &upload.bytes)?;

    let settings = Settings::global();
    let job_id = new_job_id();
    let upload_path = settings.upload_dir.join(format!("{job_id}{ext}"));
    let output_path = settings.output_dir.join(format!("{job_id}_translated{ext}"));

    tokio::task::spawn_blocking({
        let upload_path = upload_path.clone();
        move || std::fs::write(&upload_path, &upload.bytes)
    })
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let upload_path_clone = upload_path.clone();
    let parsed = tokio::task::spawn_blocking(move || {
        let result = load_subtitle(&upload_path_clone);
        // finally: save_path.unlink(missing_ok=True)
        let _ = std::fs::remove_file(&upload_path_clone);
        result
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let sub = parsed
        .map_err(|e| ApiError::bad(format!("Failed to parse subtitle: {e}")))?;

    let total_lines = sub.line_count();
    let mut progress = TranslationProgress::default();
    progress.status = "queued".to_string();

    let job = Job {
        progress: Arc::new(Mutex::new(progress)),
        cancel: Arc::new(AtomicBool::new(false)),
        is_batch: false,
        files: vec![FileMeta {
            original_filename: safe_name.clone(),
            output_path: output_path.clone(),
        }],
        source_lang: params.source_lang.clone(),
        target_lang: params.target_lang.clone(),
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        top_p: params.top_p,
        start_time: Instant::now(),
        started_at: db::now_isoformat(),
        total_lines,
    };

    state.jobs.lock().await.insert(job_id.clone(), job);

    // Background task owns the parsed SubtitleFile.
    let task_state = state.clone();
    let task_job_id = job_id.clone();
    let task_safe_name = safe_name.clone();
    let (msg_source, msg_target) = (params.source_lang.clone(), params.target_lang.clone());
    tokio::spawn(async move {
        run_single_translation(
            task_state,
            task_job_id,
            sub,
            params,
            output_path,
            task_safe_name,
        )
        .await;
    });

    Ok(Json(TranslateResponse {
        job_id,
        status: "queued".to_string(),
        message: format!(
            "Translation started: {safe_name} ({msg_source} -> {msg_target})"
        ),
    }))
}

/// Port of routes._run_translation (single-file background task).
async fn run_single_translation(
    state: AppState,
    job_id: String,
    sub: SubtitleFile,
    params: Params,
    output_path: std::path::PathBuf,
    original_filename: String,
) {
    let (progress, cancel, start_time, started_at, total_lines) = {
        let jobs = state.jobs.lock().await;
        match jobs.get(&job_id) {
            Some(job) => (
                job.progress.clone(),
                job.cancel.clone(),
                job.start_time,
                job.started_at.clone(),
                job.total_lines,
            ),
            None => return,
        }
    };

    let pipeline = Pipeline::new();
    let output = pipeline
        .translate(
            &sub,
            &params.source_lang,
            &params.target_lang,
            params.glossary.as_deref(),
            params.batch_size,
            params.temperature,
            params.max_tokens,
            params.top_p,
            params.repair,
            progress.clone(),
            cancel.clone(),
        )
        .await;
    let result = output.texts;

    // Cancelled -> status cancelled, no save, no history.
    if cancel.load(Ordering::SeqCst) {
        let mut p = progress.lock().unwrap();
        *p = TranslationProgress::default();
        p.status = "cancelled".to_string();
        return;
    }

    let save_path = output_path.clone();
    let save_result = tokio::task::spawn_blocking(move || save_subtitle(&sub, &result, &save_path))
        .await;

    let save_result = match save_result {
        Ok(r) => r,
        Err(e) => Err(e.to_string()),
    };

    match save_result {
        Ok(_) => {
            let elapsed = round2(start_time.elapsed().as_secs_f64());
            let lps = if elapsed > 0.0 {
                round1(total_lines as f64 / elapsed)
            } else {
                0.0
            };
            let row = HistoryRow {
                id: 0,
                job_id: job_id.clone(),
                source_lang: params.source_lang.clone(),
                target_lang: params.target_lang.clone(),
                is_batch: 0,
                filenames: vec![original_filename.clone()],
                output_paths: vec![output_path.to_string_lossy().into_owned()],
                total_lines: total_lines as i64,
                completed_lines: total_lines as i64,
                device: "local-llm".to_string(),
                device_used: "local-llm".to_string(),
                temperature: params.temperature,
                max_tokens: params.max_tokens as i64,
                top_p: params.top_p,
                status: "completed".to_string(),
                elapsed_seconds: elapsed,
                lines_per_second: lps,
                created_at: started_at,
                completed_at: db::now_isoformat(),
            };
            if let Err(e) = tokio::task::spawn_blocking(move || db::save_history(&row)).await {
                error!("Failed to save history for job {job_id}: {e}");
            }
        }
        Err(e) => {
            error!("Translation job {job_id} failed: {e}");
            let mut p = progress.lock().unwrap();
            *p = TranslationProgress::default();
            p.status = "failed".to_string();
            p.error = Some(e);
        }
    }
}
