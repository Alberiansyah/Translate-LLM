//! App entrypoint, port of app/main.py: tracing init, DB init, router with
//! static files, job-cleanup task, HTTP serve.

mod api;
mod batcher;
mod config;
mod db;
mod glossary;
mod langs;
mod llm;
mod models;
mod pipeline;
mod resolver;
mod state;
mod subtitle;

use std::env;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tower_http::services::ServeDir;

use crate::state::AppState;

async fn index_handler() -> Response {
    let path = config::Settings::global().static_dir.join("index.html");
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, "text/html".to_string())], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Startup only, synchronous (parity with the FastAPI lifespan).
    db::init_db().expect("failed to initialize history DB");

    let state = AppState::new();
    let settings = config::Settings::global();

    // Job cleanup loop: every 300s (port of routes._cleanup_loop).
    {
        let cleanup_state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                state::cleanup_jobs(&cleanup_state).await;
            }
        });
    }

    let app = Router::new()
        .route("/", get(index_handler))
        .nest_service("/static", ServeDir::new(&settings.static_dir))
        .merge(api::router())
        // 64x headroom over the per-file limit so multipart streaming doesn't
        // 413 before the handler's per-file check runs.
        .layer(axum::extract::DefaultBodyLimit::max(
            usize::try_from(settings.max_upload_bytes.saturating_mul(64))
                .unwrap_or(usize::MAX),
        ))
        .with_state(state);

    // NOTE: if the Python app is running simultaneously, set PORT to a value
    // other than 8000 (the Python app's default).
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind {host}:{port}: {e}"));
    tracing::info!("Serving on http://{host}:{port}");
    axum::serve(listener, app).await.expect("server error");
}
