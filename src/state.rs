//! In-memory job registry, port of the `jobs` dict in app/api/routes.py.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::Mutex as AsyncMutex;

use crate::llm::LlmEngine;
use crate::pipeline::TranslationProgress;

/// One uploaded file belonging to a job.
#[derive(Clone)]
pub struct FileMeta {
    /// Sanitized filename as sent by the browser (Path.name parity).
    pub original_filename: String,
    /// Path of the translated output file on disk.
    pub output_path: PathBuf,
}

/// An active (or recently finished) translation job.
pub struct Job {
    pub progress: Arc<Mutex<TranslationProgress>>,
    pub cancel: Arc<AtomicBool>,
    /// Single jobs: exactly 1 entry; batch jobs: one per uploaded file.
    pub files: Vec<FileMeta>,
    pub is_batch: bool,
    pub source_lang: String,
    pub target_lang: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub top_p: f64,
    pub start_time: Instant,
    /// datetime.now().isoformat() at job creation (history created_at).
    pub started_at: String,
    pub total_lines: usize,
}

pub type Jobs = Arc<AsyncMutex<HashMap<String, Job>>>;

#[derive(Clone)]
pub struct AppState {
    pub jobs: Jobs,
    pub status_engine: Arc<LlmEngine>,
}

impl AppState {
    pub fn new() -> AppState {
        AppState {
            jobs: Arc::new(AsyncMutex::new(HashMap::new())),
            status_engine: Arc::new(LlmEngine::new()),
        }
    }
}

/// Port of routes._cleanup_jobs: drop jobs whose status is terminal
/// (completed/failed/cancelled) and whose start_time is older than the TTL.
pub async fn cleanup_jobs(state: &AppState) {
    let ttl = crate::config::Settings::global().job_ttl_seconds;
    let mut jobs = state.jobs.lock().await;
    let expired: Vec<String> = jobs
        .iter()
        .filter(|(_, j)| {
            let st = {
                let p = j.progress.lock().unwrap();
                p.status.clone()
            };
            (st == "completed" || st == "failed" || st == "cancelled")
                && j.start_time.elapsed().as_secs() > ttl
        })
        .map(|(id, _)| id.clone())
        .collect();
    for id in expired {
        jobs.remove(&id);
    }
}
