use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use hnsw_rs::prelude::*;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use rusqlite::Connection;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};
use veritas_cache::{
    build_embedder, build_index, cache_key, canonical_json, embed, increment_hits,
    increment_hits_by_rowid, init_db, lookup, lookup_by_rowid, model_by_rowid, prompt_text,
    semantic_hit, store, ChatRequest, Embedder, ErrorResponse,
};

// Return a simple health check response.
async fn health() -> &'static str {
    "ok"
}

// The application state contains the database, the HTTP client, the embedder and the ANN index.
#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
    client: reqwest::Client,
    upstream_base: String,
    semantic_threshold: f32,
    embedder: Arc<Mutex<Embedder>>,
    index: Arc<Hnsw<'static, f32, DistCosine>>,
}

// Handle POST /v1/chat/completions: exact match, semantic match, or upstream proxy.
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    // Parse the request body.
    let request: ChatRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid JSON body: {}", e),
                }),
            )
                .into_response();
        }
    };

    // Reject streaming requests because Phase 1 does not support them.
    if request.stream == Some(true) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Streaming is not supported yet.".to_string(),
            }),
        )
            .into_response();
    }

    let key = match cache_key(&request) {
        Ok(k) => k,
        Err(e) => {
            error!("Failed to compute cache key: {}", e);
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    // Check the exact-match cache first.
    let cached = {
        let conn = state.db.lock().await;
        match lookup(&conn, &key) {
            Ok(v) => v,
            Err(e) => {
                error!("Cache lookup failed: {}", e);
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        }
    };

    if let Some(response_json) = cached {
        if let Err(e) = {
            let conn = state.db.lock().await;
            increment_hits(&conn, &key)
        } {
            error!("Failed to update hit count: {}", e);
        }
        info!("cache exact hit for key {}", key);
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("x-cache", "HIT")
            .header("x-cache-match", "exact")
            .body(response_json.into())
            .unwrap();
    }

    // Embed the request prompt.
    let embedding = {
        let prompt = prompt_text(&request);
        let mut embedder = state.embedder.lock().await;
        match embed(&mut embedder, &prompt) {
            Ok(e) => e,
            Err(e) => {
                error!("Embedding failed: {}", e);
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        }
    };

    // Search the HNSW index and post-filter by model.
    let best_match = {
        let conn = state.db.lock().await;
        let mut best: Option<(i64, f32)> = None;
        for neighbour in state.index.search(&embedding, 5, 32) {
            let rowid = neighbour.d_id as i64;
            let distance = neighbour.distance;
            let similarity = 1.0f32 - distance;
            let model = match model_by_rowid(&conn, rowid) {
                Ok(Some(m)) => m,
                Ok(None) => continue,
                Err(e) => {
                    error!("Failed to read model for rowid {}: {}", rowid, e);
                    continue;
                }
            };
            if model == request.model {
                match best {
                    None => best = Some((rowid, similarity)),
                    Some((_, current)) if similarity > current => best = Some((rowid, similarity)),
                    _ => {}
                }
            }
        }
        best
    };

    // Serve a semantic cache hit if the best match clears the threshold.
    if let Some((rowid, similarity)) = best_match {
        if semantic_hit(similarity, state.semantic_threshold) {
            let response_json = {
                let conn = state.db.lock().await;
                match lookup_by_rowid(&conn, rowid) {
                    Ok(Some(r)) => r,
                    Ok(None) => {
                        error!("Missing response for semantic hit rowid {}", rowid);
                        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                    }
                    Err(e) => {
                        error!("Failed to read semantic hit response: {}", e);
                        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                    }
                }
            };

            if let Err(e) = {
                let conn = state.db.lock().await;
                increment_hits_by_rowid(&conn, rowid)
            } {
                error!("Failed to update semantic hit count: {}", e);
            }

            info!(
                "cache semantic hit for key {} with similarity {}",
                key, similarity
            );
            return Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .header("x-cache", "HIT")
                .header("x-cache-match", "semantic")
                .header("x-cache-sim", format!("{:.6}", similarity))
                .body(response_json.into())
                .unwrap();
        }
    }

    // Forward the original body to the upstream LLM API.
    let url = format!("{}/v1/chat/completions", state.upstream_base);
    let mut upstream_request = state
        .client
        .post(&url)
        .header(CONTENT_TYPE, "application/json")
        .body(body_bytes.clone());

    if let Some(auth) = headers.get(AUTHORIZATION) {
        if let Ok(value) = auth.to_str() {
            upstream_request = upstream_request.header(AUTHORIZATION, value);
        }
    }

    let upstream_response = match upstream_request.send().await {
        Ok(r) => r,
        Err(e) => {
            error!("Upstream request failed: {}", e);
            return StatusCode::BAD_GATEWAY.into_response();
        }
    };

    let status = upstream_response.status();
    let response_text = match upstream_response.text().await {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to read upstream response: {}", e);
            return StatusCode::BAD_GATEWAY.into_response();
        }
    };

    // Cache successful upstream responses only.
    if status.is_success() {
        let request_json = match serde_json::to_string(&canonical_json(
            &serde_json::to_value(&request).unwrap_or(Value::Null),
        )) {
            Ok(j) => j,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        };

        let rowid = {
            let conn = state.db.lock().await;
            match store(
                &conn,
                &key,
                &request_json,
                &response_text,
                &request.model,
                &embedding,
            ) {
                Ok(id) => id,
                Err(e) => {
                    error!("Failed to store response: {}", e);
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
            }
        };

        // Insert the new embedding into the in-memory index.
        state.index.insert((&embedding, rowid as usize));
    }

    info!("cache miss for key {}", key);
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("x-cache", "MISS")
        .body(response_text.into())
        .unwrap()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Keep dependency logs quiet. Allow an override with RUST_LOG.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "veritas_cache=info,ort=warn,hnsw_rs=warn".into()),
        )
        .init();

    let db_path = std::env::var("CACHE_DB_PATH").unwrap_or_else(|_| "cache.db".to_string());
    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    let upstream_base =
        std::env::var("UPSTREAM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());

    let semantic_threshold = std::env::var("SEMANTIC_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.85);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let embedder = Arc::new(Mutex::new(build_embedder()?));

    // Build the index after the database is ready but before serving traffic.
    let index = {
        let conn = db.lock().await;
        build_index(&conn)?
    };
    let index = Arc::new(index);

    let state = AppState {
        db,
        client,
        upstream_base,
        semantic_threshold,
        embedder,
        index,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    info!("veritas-cache listening on 127.0.0.1:8080");
    axum::serve(listener, app).await?;

    Ok(())
}
