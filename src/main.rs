use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::unfold;
use futures_util::StreamExt;
use hnsw_rs::prelude::*;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use rusqlite::Connection;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};
use veritas_cache::adaptive::Ld3Policy;
use veritas_cache::policy::{Decision, Policy};
use veritas_cache::{
    assemble_from_sse, build_embedder, build_index, cache_key, canonical_json, embed,
    increment_hits, increment_hits_by_rowid, init_db, insert_observation, judge_content_equal,
    load_observations, lookup,
    lookup_by_rowid, model_by_rowid, prompt_text, render_sse, response_content, semantic_hit,
    store, ChatRequest, Embedder, ErrorResponse,
};

// Return a simple health check response.
async fn health() -> &'static str {
    "ok"
}

async fn add_policy_header(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    if let Ok(value) = state.policy_name.parse() {
        response.headers_mut().insert("x-cache-policy", value);
    }
    response
}

// The application state contains the database, the HTTP client, the embedder and the ANN index.
#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
    client: reqwest::Client,
    upstream_base: String,
    semantic_threshold: f32,
    policy_name: String,
    adaptive_policy: Option<Arc<Mutex<Ld3Policy>>>,
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
        return serve_hit(&state, &request, response_json, "exact", None);
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

    // Decide whether the semantic neighbor is a hit.
    let semantic_decision = if state.policy_name == "ld3" {
        let mut policy = state
            .adaptive_policy
            .as_ref()
            .expect("ld3 policy")
            .lock()
            .await;
        policy.decide(best_match.map(|(rowid, similarity)| (rowid as usize, similarity)))
    } else {
        match best_match {
            Some((_, similarity)) if semantic_hit(similarity, state.semantic_threshold) => {
                Decision::Hit
            }
            _ => Decision::Miss,
        }
    };

    // Serve an exact semantic hit. Adaptive hits have a real neighbor.
    if let Some((rowid, similarity)) = best_match {
        if semantic_decision == Decision::Hit {
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
            return serve_hit(&state, &request, response_json, "semantic", Some(similarity));
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
    let streaming = request.stream == Some(true);

    // A streaming miss passes the upstream body through and caches on completion.
    if status.is_success() && streaming {
        return stream_and_cache(state, request, key, embedding, best_match, upstream_response)
            .await;
    }

    let response_text = match upstream_response.text().await {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to read upstream response: {}", e);
            return StatusCode::BAD_GATEWAY.into_response();
        }
    };

    // Cache successful upstream responses only.
    if status.is_success() {
        maybe_store(&state, &key, &request, &response_text, &embedding, best_match).await;
    }

    info!("cache miss for key {}", key);
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("x-cache", "MISS")
        .header("x-cache-policy", &state.policy_name)
        .body(response_text.into())
        .unwrap()
}

// Serve a cached response. Streaming requests get an SSE stream.
fn serve_hit(
    state: &AppState,
    request: &ChatRequest,
    response_json: String,
    match_kind: &str,
    similarity: Option<f32>,
) -> Response {
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header("x-cache", "HIT")
        .header("x-cache-match", match_kind)
        .header("x-cache-policy", &state.policy_name);
    if let Some(sim) = similarity {
        builder = builder.header("x-cache-sim", format!("{:.6}", sim));
    }
    if request.stream == Some(true) {
        match render_sse(&response_json) {
            Some(sse) => builder
                .header("content-type", "text/event-stream")
                .body(sse.into())
                .unwrap(),
            None => builder
                .header("content-type", "application/json")
                .body(response_json.into())
                .unwrap(),
        }
    } else {
        builder
            .header("content-type", "application/json")
            .body(response_json.into())
            .unwrap()
    }
}

// Cache a successful upstream response and update the adaptive policy.
async fn maybe_store(
    state: &AppState,
    key: &str,
    request: &ChatRequest,
    response_json: &str,
    embedding: &[f32],
    best_match: Option<(i64, f32)>,
) {
    let request_json = match serde_json::to_string(&canonical_json(
        &serde_json::to_value(request).unwrap_or(Value::Null),
    )) {
        Ok(j) => j,
        Err(e) => {
            error!("Failed to serialize request: {}", e);
            return;
        }
    };

    let mut should_store = true;
    if state.policy_name == "ld3" {
        if let Some((rowid, similarity)) = best_match {
            let cached_json = {
                let conn = state.db.lock().await;
                lookup_by_rowid(&conn, rowid).ok().flatten()
            };
            if let Some(cached_json) = cached_json {
                if response_content(&cached_json).is_some()
                    && response_content(response_json).is_some()
                {
                    let correct = judge_content_equal(&cached_json, response_json);
                    {
                        let mut policy = state
                            .adaptive_policy
                            .as_ref()
                            .expect("ld3 policy")
                            .lock()
                            .await;
                        policy.observe(rowid as usize, similarity, correct);
                        should_store = policy.should_insert();
                    }
                    // Persist the observation so the policy survives a restart.
                    let conn = state.db.lock().await;
                    if let Err(e) = insert_observation(&conn, rowid, similarity, correct) {
                        error!("Failed to store observation: {}", e);
                    }
                }
            }
        }
    }

    if should_store {
        let rowid = {
            let conn = state.db.lock().await;
            match store(
                &conn,
                key,
                &request_json,
                response_json,
                &request.model,
                embedding,
            ) {
                Ok(id) => id,
                Err(e) => {
                    error!("Failed to store response: {}", e);
                    return;
                }
            }
        };

        // Insert the new embedding into the in-memory index.
        state.index.insert((embedding, rowid as usize));
    }
}

// Stream an upstream SSE response to the client and cache the assembled completion.
async fn stream_and_cache(
    state: AppState,
    request: ChatRequest,
    key: String,
    embedding: Vec<f32>,
    best_match: Option<(i64, f32)>,
    upstream_response: reqwest::Response,
) -> Response {
    let policy_name = state.policy_name.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, reqwest::Error>>(64);
    let buffer: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

    let tee_buffer = buffer.clone();
    tokio::spawn(async move {
        let mut stream = upstream_response.bytes_stream();
        let mut failed = false;
        while let Some(item) = stream.next().await {
            match &item {
                Ok(bytes) => {
                    let mut buf = tee_buffer.lock().await;
                    buf.extend_from_slice(bytes);
                }
                Err(e) => {
                    error!("Upstream stream error: {}", e);
                    failed = true;
                    break;
                }
            }
            if tx.send(item).await.is_err() {
                // The client disconnected. Do not cache.
                return;
            }
        }
        if failed {
            return;
        }
        let sse_text = {
            let buf = buffer.lock().await;
            String::from_utf8_lossy(&buf).into_owned()
        };
        if let Some(assembled) = assemble_from_sse(&sse_text) {
            maybe_store(
                &state,
                &key,
                &request,
                &assembled.to_string(),
                &embedding,
                best_match,
            )
            .await;
        }
    });

    let stream = unfold(rx, |mut rx| async move { rx.recv().await.map(|item| (item, rx)) });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("x-cache", "MISS")
        .header("x-cache-policy", &policy_name)
        .body(Body::from_stream(stream))
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
    let policy_name = std::env::var("SEMANTIC_POLICY").unwrap_or_else(|_| "static".to_string());
    let adaptive_policy = match policy_name.as_str() {
        "static" => None,
        "ld3" => {
            let delta = std::env::var("ADAPTIVE_DELTA")
                .ok()
                .map(|value| value.parse::<f32>())
                .transpose()
                .map_err(|e| format!("Invalid ADAPTIVE_DELTA: {e}"))?
                .unwrap_or(0.05);
            if !(0.0..=1.0).contains(&delta) {
                return Err("ADAPTIVE_DELTA must be between 0 and 1".into());
            }
            // Observations replay from the database after the index build.
            Some(Arc::new(Mutex::new(Ld3Policy::new(delta))))
        }
        other => return Err(format!("Unknown SEMANTIC_POLICY: {other}. Use static or ld3").into()),
    };

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

    // Replay stored observations so an adaptive policy keeps its learned state.
    // Each observe refits from the full per-entry vector. The replay order is the
    // write order, so the final state matches a fresh in-memory run.
    if let Some(policy) = &adaptive_policy {
        let observations = {
            let conn = db.lock().await;
            load_observations(&conn)?
        };
        let count = observations.len();
        let mut policy = policy.lock().await;
        for (entry_rowid, similarity, correct) in observations {
            policy.observe(entry_rowid as usize, similarity, correct);
        }
        info!("replayed {} adaptive observations", count);
    }

    let state = AppState {
        db,
        client,
        upstream_base,
        semantic_threshold,
        policy_name,
        adaptive_policy,
        embedder,
        index,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, add_policy_header));

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    info!("veritas-cache listening on 127.0.0.1:{}", port);
    axum::serve(listener, app).await?;

    Ok(())
}
