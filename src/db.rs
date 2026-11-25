//! SQLite history store, port of app/db.py using rusqlite (bundled).
//!
//! Every call opens its own connection (Python's get_conn context manager),
//! sets WAL + busy_timeout, and commits on success.

use std::path::Path;

use rusqlite::Connection;
use tracing::warn;

use crate::config::Settings;

/// One `history` row; JSON-encoded columns (filenames/output_paths) decoded.
#[derive(Debug, Clone)]
pub struct HistoryRow {
    pub id: i64,
    pub job_id: String,
    pub source_lang: String,
    pub target_lang: String,
    pub is_batch: i64,
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

fn db_path() -> &'static Path {
    Settings::global().db_path.as_path()
}

/// Port of db.get_conn: open + pragmas. Callers commit implicitly on drop of
/// the statement; rusqlite executes in autocommit mode which matches the
/// single-statement usage in db.py.
fn open_conn() -> Result<Connection, String> {
    let conn = Connection::open(db_path()).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_millis(5000))
        .map_err(|e| e.to_string())?;
    let _ = conn
        .pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| e.to_string())?;
    Ok(conn)
}

pub fn init_db() -> Result<(), String> {
    let conn = open_conn()?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            is_batch INTEGER DEFAULT 0,
            filenames TEXT NOT NULL,
            output_paths TEXT NOT NULL,
            total_lines INTEGER DEFAULT 0,
            completed_lines INTEGER DEFAULT 0,
            device TEXT DEFAULT 'local-llm',
            device_used TEXT DEFAULT 'local-llm',
            temperature REAL DEFAULT 0.3,
            max_tokens INTEGER DEFAULT 2048,
            top_p REAL DEFAULT 0.9,
            status TEXT DEFAULT 'completed',
            elapsed_seconds REAL DEFAULT 0,
            lines_per_second REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS translation_memory (
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            source_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (source_lang, target_lang, source_text)
        );",
    )
    .map_err(|e| e.to_string())?;
    tracing::info!("History DB initialized at {}", db_path().display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Translation memory (exact-match reuse of past translations)
// ---------------------------------------------------------------------------

/// Heuristic: a cached "translation" that still contains most of the
/// significant source words is suspicious (untranslated or only partially
/// translated — e.g. a weak-model echo or fragment). Suspicious entries are
/// never served from the cache; they are re-translated and replaced on the
/// next run, so stale/partial entries self-heal without manual cleanup.
pub fn source_overlap(src: &str, tgt: &str) -> bool {
    let src_words: Vec<&str> = src
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().count() >= 3)
        .collect();
    // Too few significant words to judge (names, single words).
    if src_words.len() < 2 {
        return false;
    }
    let tgt_lower = tgt.to_lowercase();
    let shared = src_words
        .iter()
        .filter(|w| tgt_lower.contains(&w.to_lowercase()))
        .count();
    shared * 100 / src_words.len() >= 70
}

/// Look up cached translations for `texts` under (source_lang, target_lang).
/// Returns a Vec parallel to `texts`: `Some(target)` when a cached translation
/// exists, `None` when the text must go to the LLM. Suspicious entries
/// (source_overlap) are treated as misses so they self-heal. Failed/echoed
/// lines are not stored, so the cache never gets poisoned by total failures.
pub fn tm_lookup(
    texts: &[String],
    source_lang: &str,
    target_lang: &str,
) -> Result<Vec<Option<String>>, String> {
    let conn = open_conn()?;
    let mut out: Vec<Option<String>> = vec![None; texts.len()];
    // Chunk the IN(...) queries to stay well under any SQLite parameter limit.
    for chunk_start in (0..texts.len()).step_by(500) {
        let chunk = &texts[chunk_start..(chunk_start + 500).min(texts.len())];
        let placeholders = (1..=chunk.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT source_text, target_text FROM translation_memory
             WHERE source_lang = ?1 AND target_lang = ?2 AND source_text IN ({placeholders})"
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
        let mut params: Vec<&str> = vec![source_lang, target_lang];
        params.extend(chunk.iter().map(|s| s.as_str()));
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (src, tgt) = row.map_err(|e| e.to_string())?;
            // Suspicious cached translation: serve nothing, let the LLM redo it.
            if source_overlap(&src, &tgt) {
                continue;
            }
            if let Some(pos) = chunk.iter().position(|t| *t == src) {
                out[chunk_start + pos] = Some(tgt);
            }
        }
    }
    Ok(out)
}

/// Store successful translations. Earlier entries win (INSERT OR IGNORE) so
/// later jobs cannot poison an established entry — EXCEPT suspicious cached
/// entries (source_overlap: still mostly the source text), which are replaced
/// by the fresh translation so stale/partial rows self-heal on the next run.
/// Runs in a single transaction; returns the number of rows written.
pub fn tm_save(
    entries: &[(String, String)],
    source_lang: &str,
    target_lang: &str,
) -> Result<usize, String> {
    if entries.is_empty() {
        return Ok(0);
    }
    let mut conn = open_conn()?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let now = now_isoformat();
    let mut written = 0usize;
    {
        let mut insert = tx
            .prepare(
                "INSERT OR IGNORE INTO translation_memory
                 (source_lang, target_lang, source_text, target_text, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )
            .map_err(|e| e.to_string())?;
        let mut replace = tx
            .prepare(
                "UPDATE translation_memory
                 SET target_text = ?4, created_at = ?5
                 WHERE source_lang = ?1 AND target_lang = ?2 AND source_text = ?3",
            )
            .map_err(|e| e.to_string())?;
        for (src, tgt) in entries {
            let n = insert
                .execute(rusqlite::params![source_lang, target_lang, src, tgt, now])
                .map_err(|e| e.to_string())?;
            if n > 0 {
                written += n;
                continue;
            }
            // Entry already exists: overwrite it only when the cached value is
            // suspicious (echo/fragment). A trustable cached translation wins.
            let cached: String = tx
                .query_row(
                    "SELECT target_text FROM translation_memory
                     WHERE source_lang = ?1 AND target_lang = ?2 AND source_text = ?3",
                    rusqlite::params![source_lang, target_lang, src],
                    |r| r.get(0),
                )
                .map_err(|e| e.to_string())?;
            if source_overlap(src, &cached) {
                written += replace
                    .execute(rusqlite::params![source_lang, target_lang, src, tgt, now])
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(written)
}

/// Port of db.save_history: INSERT OR REPLACE, 18 columns.
#[allow(clippy::too_many_arguments)]
pub fn save_history(row: &HistoryRow) -> Result<(), String> {
    let conn = open_conn()?;
    conn.execute(
        "INSERT OR REPLACE INTO history
         (job_id, source_lang, target_lang, is_batch, filenames, output_paths,
          total_lines, completed_lines, device, device_used, temperature,
          max_tokens, top_p, status, elapsed_seconds, lines_per_second,
          created_at, completed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        rusqlite::params![
            row.job_id,
            row.source_lang,
            row.target_lang,
            row.is_batch,
            serde_json::to_string(&row.filenames).unwrap_or_else(|_| "[]".into()),
            serde_json::to_string(&row.output_paths).unwrap_or_else(|_| "[]".into()),
            row.total_lines,
            row.completed_lines,
            row.device,
            row.device_used,
            row.temperature,
            row.max_tokens,
            row.top_p,
            row.status,
            row.elapsed_seconds,
            row.lines_per_second,
            row.created_at,
            row.completed_at,
        ],
    )
    .map_err(|e| e.to_string())?;
    tracing::info!("History saved: {}", row.job_id);
    Ok(())
}

fn row_to_history(row: &rusqlite::Row<'_>) -> rusqlite::Result<HistoryRow> {
    let filenames_raw: String = row.get("filenames")?;
    let output_paths_raw: String = row.get("output_paths")?;
    let filenames = parse_json_vec("filenames", &filenames_raw);
    let output_paths = parse_json_vec("output_paths", &output_paths_raw);
    Ok(HistoryRow {
        id: row.get("id")?,
        job_id: row.get("job_id")?,
        source_lang: row.get("source_lang")?,
        target_lang: row.get("target_lang")?,
        is_batch: row.get("is_batch")?,
        filenames,
        output_paths,
        total_lines: row.get("total_lines")?,
        completed_lines: row.get("completed_lines")?,
        device: row.get("device")?,
        device_used: row.get("device_used")?,
        temperature: row.get("temperature")?,
        max_tokens: row.get("max_tokens")?,
        top_p: row.get("top_p")?,
        status: row.get("status")?,
        elapsed_seconds: row.get("elapsed_seconds")?,
        lines_per_second: row.get("lines_per_second")?,
        created_at: row.get("created_at")?,
        completed_at: row.get("completed_at")?,
    })
}

/// Port of db._row_to_dict JSON decoding: corrupt JSON -> warn + empty vec.
fn parse_json_vec(key: &str, raw: &str) -> Vec<String> {
    match serde_json::from_str::<Vec<String>>(raw) {
        Ok(v) => v,
        Err(e) => {
            warn!("Corrupt {key} in history row ({e}); falling back to []");
            Vec::new()
        }
    }
}

pub fn get_history(limit: i64) -> Result<Vec<HistoryRow>, String> {
    let conn = open_conn()?;
    let mut stmt = conn
        .prepare("SELECT * FROM history ORDER BY id DESC LIMIT ?1")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![limit], row_to_history)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

pub fn get_history_entry(job_id: &str) -> Result<Option<HistoryRow>, String> {
    let conn = open_conn()?;
    let mut stmt = conn
        .prepare("SELECT * FROM history WHERE job_id = ?1")
        .map_err(|e| e.to_string())?;
    let mut rows = stmt
        .query_map(rusqlite::params![job_id], row_to_history)
        .map_err(|e| e.to_string())?;
    match rows.next() {
        Some(r) => Ok(Some(r.map_err(|e| e.to_string())?)),
        None => Ok(None),
    }
}

pub fn delete_history(job_id: &str) -> Result<bool, String> {
    let conn = open_conn()?;
    let n = conn
        .execute("DELETE FROM history WHERE job_id = ?1", rusqlite::params![job_id])
        .map_err(|e| e.to_string())?;
    let deleted = n > 0;
    if deleted {
        tracing::info!("History deleted: {job_id}");
    }
    Ok(deleted)
}

/// Python datetime.now().isoformat() parity: local time, microsecond precision.
pub fn now_isoformat() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}
