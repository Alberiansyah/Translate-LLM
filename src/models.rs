//! JSON response models — field names must exactly match the Python Pydantic
//! models in app/api/models.py because the frontend (app/static/app.js) is
//! reused unchanged.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct TranslateResponse {
    pub job_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ProgressResponse {
    pub job_id: String,
    pub status: String,
    pub percent: f64,
    pub total_batches: i64,
    pub completed_batches: i64,
    pub error: Option<String>,
    pub elapsed_seconds: f64,
    pub last_batch_seconds: f64,
    pub device_used: String,
    pub lines_per_second: f64,
    pub total_files: i64,
    pub completed_files: i64,
    pub current_file: String,
    pub total_lines: i64,
    pub completed_lines: i64,
    pub eta_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct HistoryEntry {
    pub job_id: String,
    pub source_lang: String,
    pub target_lang: String,
    pub is_batch: bool,
    pub filenames: Vec<String>,
    pub output_paths: Vec<String>,
    pub total_lines: i64,
    pub completed_lines: i64,
    pub device: String,
    pub device_used: String,
    pub temperature: f64,
    pub max_tokens: i64,
    pub top_p: f64,
    pub status: String,
    pub elapsed_seconds: f64,
    pub lines_per_second: f64,
    pub created_at: String,
    pub completed_at: String,
}

/// Round like Python's round(x, 2) for display values.
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Round like Python's round(x, 1) for display values.
pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
