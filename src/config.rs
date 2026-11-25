//! App configuration, mirroring app/config.py. All values env-overridable.
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

pub struct Settings {
    /// Repo root (parent of rust/), same layout as the Python app.
    pub base_dir: PathBuf,
    pub upload_dir: PathBuf,
    pub output_dir: PathBuf,
    pub db_path: PathBuf,
    /// Served by the Rust binary but owned by the Python app (single source).
    pub static_dir: PathBuf,

    // Local LLM server (llama-server OpenAI-compatible API)
    pub llm_base_url: String,
    pub llm_model_name: String,

    // Upload / job lifecycle
    pub max_upload_bytes: u64,
    pub job_ttl_seconds: u64,
    pub allowed_extensions: [&'static str; 4],

    // LLM request tuning
    pub llm_timeout_secs: u64,
    pub llm_connect_timeout_secs: u64,
    pub llm_max_retries: u32,
    pub llm_concurrency: usize,
    pub max_lines_per_batch: u32,

    // Translation defaults
    pub batch_size: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: u32,

    // Efficiency / context features
    /// Persistent translation memory (SQLite) keyed by (source, target, text).
    pub tm_enabled: bool,
    /// Lines of preceding dialogue injected into each batch prompt for
    /// continuity (0 disables).
    pub context_lines: usize,
    /// One-shot context analysis pass before translating: scans the opening
    /// lines to extract characters/terms/tone into a context brief injected
    /// into the system prompt (stable per file, cache-friendly).
    pub analyze_enabled: bool,
    /// How many opening (non-empty) lines the analysis pass reads.
    pub analyze_lines: usize,
}

fn env_string(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

impl Settings {
    fn load() -> Settings {
        // Runtime base dir = the directory containing the exe, so a copied
        // exe becomes a standalone deployment (its own app/static, uploads,
        // output, data.db). Fallback: when the exe sits inside the source
        // tree (target/debug or target/release there is no app/static next
        // to it), use the compiled-in repo path like before.
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let base_dir = match exe_dir {
            Some(dir) if dir.join("app").join("static").is_dir() => dir,
            _ => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("cargo manifest dir must have a parent")
                .to_path_buf(),
        };
        let upload_dir = base_dir.join("uploads");
        let output_dir = base_dir.join("output");
        // Same as Python: created eagerly at startup.
        let _ = fs::create_dir_all(&upload_dir);
        let _ = fs::create_dir_all(&output_dir);

        Settings {
            static_dir: base_dir.join("app").join("static"),
            upload_dir,
            output_dir,
            db_path: base_dir.join("data.db"),
            base_dir,
            llm_base_url: env_string("LLM_BASE_URL", "http://127.0.0.1:8080"),
            llm_model_name: env_string("LLM_MODEL_NAME", "local-llm"),
            max_upload_bytes: env_parse("MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            job_ttl_seconds: env_parse("JOB_TTL_SECONDS", 3600u64),
            allowed_extensions: [".srt", ".ass", ".ssa", ".vtt"],
            llm_timeout_secs: env_parse("LLM_TIMEOUT", 120u64),
            llm_connect_timeout_secs: env_parse("LLM_CONNECT_TIMEOUT", 5u64),
            llm_max_retries: env_parse("LLM_MAX_RETRIES", 1u32),
            llm_concurrency: env_parse("LLM_CONCURRENCY", 1usize),
            max_lines_per_batch: 100,
            batch_size: env_parse("BATCH_SIZE", 15u32),
            temperature: env_parse("TEMPERATURE", 0.3f64),
            top_p: env_parse("TOP_P", 0.9f64),
            max_tokens: env_parse("MAX_TOKENS", 2048u32),
            tm_enabled: env_parse("TM_ENABLED", true),
            context_lines: env_parse("CONTEXT_LINES", 2usize),
            analyze_enabled: env_parse("ANALYZE_CONTEXT", true),
            analyze_lines: env_parse("ANALYZE_LINES", 150usize),
        }
    }

    pub fn global() -> &'static Settings {
        static SETTINGS: OnceLock<Settings> = OnceLock::new();
        SETTINGS.get_or_init(Self::load)
    }
}
