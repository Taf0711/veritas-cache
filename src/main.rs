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
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};
use veritas_cache::adaptive::{AdaptivePolicy, Ld3Policy};
use veritas_cache::policy::Decision;
use veritas_cache::{
    assemble_from_sse, build_embedder, build_index, cache_key, canonical_json, count_tokens,
    embed, evict, increment_hits, increment_hits_by_rowid, init_db, insert_observation,
    migrate_shadow_log,
    judge_content_equal, load_file_config, load_observations, lookup,
    insert_shadow_row, set_shadow_fresh,
    lookup_by_rowid, model_by_rowid, parse_exact_only_models, prompt_cache_key_by_rowid,
    prompt_text, render_sse,
    response_content, semantic_hit, store, synthesize_usage, unix_now, ChatRequest, Embedder,
    ErrorResponse,
};

// Report the runtime counters as JSON.
async fn get_metrics(State(state): State<AppState>) -> Response {
    let m = &state.metrics;
    Json(json!({
        "hits_exact": m.hits_exact.load(Ordering::Relaxed),
        "hits_semantic": m.hits_semantic.load(Ordering::Relaxed),
        "misses": m.misses.load(Ordering::Relaxed),
        "stores": m.stores.load(Ordering::Relaxed),
        "evicted": m.evicted.load(Ordering::Relaxed),
    }))
    .into_response()
}

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
// Runtime counters for the metrics endpoint.
#[derive(Default)]
struct Metrics {
    hits_exact: AtomicU64,
    hits_semantic: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    evicted: AtomicU64,
}

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
    client: reqwest::Client,
    upstream_base: String,
    semantic_threshold: f32,
    policy_name: String,
    adaptive_policy: Option<Arc<Mutex<AdaptivePolicy>>>,
    embedder: Arc<Mutex<Embedder>>,
    index: Arc<Hnsw<'static, f32, DistCosine>>,
    ttl_seconds: i64,
    max_entries: i64,
    exact_only_models: std::collections::HashSet<String>,
    metrics: Arc<Metrics>,
    shadow: bool,
}

// Log one shadow-mode decision and return its row id.
async fn log_shadow(
    state: &AppState,
    key: &str,
    request: &ChatRequest,
    decision: &str,
    similarity: Option<f32>,
    would_serve_json: Option<&str>,
) -> Option<i64> {
    info!("shadow decision={} key={} sim={:?}", decision, key, similarity);
    let request_json = serde_json::to_string(&canonical_json(
        &serde_json::to_value(request).unwrap_or(Value::Null),
    ))
    .unwrap_or_default();
    let conn = state.db.lock().await;
    match insert_shadow_row(
        &conn,
        key,
        &request.model,
        decision,
        similarity,
        would_serve_json,
        &request_json,
    ) {
        Ok(id) => Some(id),
        Err(e) => {
            error!("Failed to log shadow decision: {}", e);
            None
        }
    }
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

    // In shadow mode each request logs its decision here.
    let mut shadow_row_id: Option<i64> = None;

    if let Some(response_json) = cached {
        if state.shadow {
            shadow_row_id = log_shadow(
                &state,
                &key,
                &request,
                "exact_hit",
                None,
                Some(&response_json),
            )
            .await;
        } else {
            if let Err(e) = {
                let conn = state.db.lock().await;
                increment_hits(&conn, &key)
            } {
                error!("Failed to update hit count: {}", e);
            }
            info!("cache exact hit for key {}", key);
            state.metrics.hits_exact.fetch_add(1, Ordering::Relaxed);
            return serve_hit(&state, &request, response_json, "exact", None).await;
        }
    }

    // Exact-only models skip the semantic path. They still store for exact reuse.
    // A logged shadow exact hit also skips it. Live mode would not run it either.
    let exact_only = state.exact_only_models.contains(&request.model) || shadow_row_id.is_some();

    // Embed the request prompt.
    let embedding = if exact_only {
        Vec::new()
    } else {
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
    // When the request carries a prompt cache key, keep hits inside that scope.
    let request_scope = request
        .extra
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .map(String::from);
    let best_match = if exact_only {
        None
    } else {
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
                if let Some(scope) = &request_scope {
                    match prompt_cache_key_by_rowid(&conn, rowid) {
                        Ok(Some(stored)) if &stored == scope => {}
                        Ok(_) => continue,
                        Err(e) => {
                            error!("Failed to read prompt cache key for rowid {}: {}", rowid, e);
                            continue;
                        }
                    }
                }
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
    let semantic_decision = if exact_only {
        Decision::Miss
    } else if let Some(policy) = &state.adaptive_policy {
        let mut policy = policy.lock().await;
        policy.decide(
            request_scope.as_deref(),
            best_match.map(|(rowid, similarity)| (rowid as usize, similarity)),
        )
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

            if state.shadow {
                shadow_row_id = log_shadow(
                    &state,
                    &key,
                    &request,
                    "semantic_hit",
                    Some(similarity),
                    Some(&response_json),
                )
                .await;
            } else {
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
                state.metrics.hits_semantic.fetch_add(1, Ordering::Relaxed);
                return serve_hit(&state, &request, response_json, "semantic", Some(similarity))
                    .await;
            }
        }
    }

    // Log a shadow miss. Keep the neighbor similarity when one was rejected.
    if state.shadow && shadow_row_id.is_none() {
        shadow_row_id = log_shadow(
            &state,
            &key,
            &request,
            "miss",
            best_match.map(|(_, similarity)| similarity),
            None,
        )
        .await;
    }

    // Forward the original body to the upstream LLM API.
    state.metrics.misses.fetch_add(1, Ordering::Relaxed);
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
        return stream_and_cache(
            state,
            request,
            key,
            embedding,
            best_match,
            shadow_row_id,
            upstream_response,
        )
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
        maybe_store(
            &state,
            &key,
            &request,
            &response_text,
            &embedding,
            best_match,
            shadow_row_id,
        )
        .await;
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
// Rewrite the usage first so the hit reports plausible tokens.
async fn serve_hit(
    state: &AppState,
    request: &ChatRequest,
    response_json: String,
    match_kind: &str,
    similarity: Option<f32>,
) -> Response {
    let response_json = {
        let prompt = prompt_text(request);
        let embedder = state.embedder.lock().await;
        match count_tokens(&embedder, &prompt) {
            Ok(prompt_tokens) => synthesize_usage(&response_json, prompt_tokens),
            Err(e) => {
                error!("Failed to count prompt tokens: {}", e);
                response_json
            }
        }
    };
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
// Cache a successful upstream response and update the adaptive policy.
// Attach the fresh response to the shadow row when shadow mode logged one.
async fn maybe_store(
    state: &AppState,
    key: &str,
    request: &ChatRequest,
    response_json: &str,
    embedding: &[f32],
    best_match: Option<(i64, f32)>,
    shadow_row_id: Option<i64>,
) {
    if let Some(id) = shadow_row_id {
        let conn = state.db.lock().await;
        if let Err(e) = set_shadow_fresh(&conn, id, response_json) {
            error!("Failed to set shadow fresh response: {}", e);
        }
    }
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
    if let Some(policy) = &state.adaptive_policy {
        let scope = request
            .extra
            .get("prompt_cache_key")
            .and_then(Value::as_str);
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
                        let mut policy = policy.lock().await;
                        policy.observe(scope, rowid as usize, similarity, correct);
                        should_store = policy.should_insert();
                    }
                    // Persist the observation so the policy survives a restart.
                    let conn = state.db.lock().await;
                    if let Err(e) = insert_observation(&conn, rowid, similarity, correct, scope) {
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
        if !embedding.is_empty() {
            state.index.insert((embedding, rowid as usize));
        }
        state.metrics.stores.fetch_add(1, Ordering::Relaxed);

        // Evict expired and excess entries after each store.
        let conn = state.db.lock().await;
        match evict(&conn, state.ttl_seconds, state.max_entries, unix_now()) {
            Ok(deleted) if deleted > 0 => {
                state
                    .metrics
                    .evicted
                    .fetch_add(deleted as u64, Ordering::Relaxed);
                info!("evicted {} cache entries", deleted);
            }
            Ok(_) => {}
            Err(e) => error!("Failed to evict entries: {}", e),
        }
    }
}

// Stream an upstream SSE response to the client and cache the assembled completion.
async fn stream_and_cache(
    state: AppState,
    request: ChatRequest,
    key: String,
    embedding: Vec<f32>,
    best_match: Option<(i64, f32)>,
    shadow_row_id: Option<i64>,
    upstream_response: reqwest::Response,
) -> Response {
    let policy_name = state.policy_name.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, reqwest::Error>>(64);
    let buffer: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    let tee_buffer = buffer.clone();
    tokio::spawn(async move {
        let mut stream = upstream_response.bytes_stream();
        let mut failed = false;
        let mut client_gone = false;
        while let Some(item) = stream.next().await {
            match &item {
                Ok(bytes) => {
                    let mut buf = tee_buffer.lock().unwrap();
                    buf.extend_from_slice(bytes);
                }
                Err(e) => {
                    error!("Upstream stream error: {}", e);
                    failed = true;
                    break;
                }
            }
            if !client_gone && tx.send(item).await.is_err() {
                // The client disconnected early. Real clients often close at [DONE].
                // Drain the rest of the stream and cache the complete response.
                client_gone = true;
            }
        }
        if failed {
            return;
        }
        let sse_text = {
            let buf = buffer.lock().unwrap();
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
                shadow_row_id,
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

    let file_config = load_file_config().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let db_path = std::env::var("CACHE_DB_PATH")
        .ok()
        .or(file_config.db_path)
        .unwrap_or_else(|| "cache.db".to_string());
    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;
    migrate_shadow_log(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    let upstream_base = std::env::var("UPSTREAM_BASE_URL")
        .ok()
        .or(file_config.upstream_base_url)
        .unwrap_or_else(|| "https://api.openai.com".to_string());

    let semantic_threshold = std::env::var("SEMANTIC_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .or(file_config.semantic_threshold)
        .unwrap_or(0.85);
    let policy_name = std::env::var("SEMANTIC_POLICY")
        .ok()
        .or(file_config.semantic_policy)
        .unwrap_or_else(|| "static".to_string());
    let ttl_seconds = std::env::var("CACHE_TTL_SECONDS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .or(file_config.ttl_seconds)
        .unwrap_or(0);
    let max_entries = std::env::var("CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .or(file_config.max_entries)
        .unwrap_or(0);
    let exact_only_models = match std::env::var("CACHE_EXACT_ONLY_MODELS") {
        Ok(v) => parse_exact_only_models(&v),
        Err(_) => file_config
            .exact_only_models
            .unwrap_or_default()
            .into_iter()
            .collect(),
    };
    let shadow = match std::env::var("CACHE_SHADOW") {
        Ok(v) => v == "1",
        Err(_) => file_config.shadow.unwrap_or(false),
    };
    let adaptive_policy = match policy_name.as_str() {
        "static" => None,
        "ld3" | "ld3s" => {
            let delta = std::env::var("ADAPTIVE_DELTA")
                .ok()
                .map(|value| value.parse::<f32>())
                .transpose()
                .map_err(|e| format!("Invalid ADAPTIVE_DELTA: {e}"))?
                .unwrap_or(0.05);
            if !(0.0..=1.0).contains(&delta) {
                return Err("ADAPTIVE_DELTA must be between 0 and 1".into());
            }
            if policy_name == "ld3s" {
                Some(Arc::new(Mutex::new(AdaptivePolicy::Scoped(
                    veritas_cache::adaptive::ScopedLd3Policy::new(delta),
                ))))
            } else {
                // Observations replay from the database after the index build.
                Some(Arc::new(Mutex::new(AdaptivePolicy::Ld3(Ld3Policy::new(delta)))))
            }
        }
        other => {
            return Err(format!("Unknown SEMANTIC_POLICY: {other}. Use static, ld3, or ld3s").into())
        }
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
    // Each observe refits from the full per-key vector. The replay order is the
    // write order, so the final state matches a fresh in-memory run.
    if let Some(policy) = &adaptive_policy {
        let observations = {
            let conn = db.lock().await;
            load_observations(&conn)?
        };
        let count = observations.len();
        let mut policy = policy.lock().await;
        for (entry_rowid, scope, similarity, correct) in observations {
            policy.observe(scope.as_deref(), entry_rowid as usize, similarity, correct);
        }
        info!("replayed {} adaptive observations", count);
    }

    let metrics = Arc::new(Metrics::default());

    // Evict entries that already exceed the limits before serving traffic.
    {
        let conn = db.lock().await;
        match evict(&conn, ttl_seconds, max_entries, unix_now()) {
            Ok(deleted) if deleted > 0 => {
                metrics.evicted.fetch_add(deleted as u64, Ordering::Relaxed);
                info!("evicted {} cache entries at boot", deleted);
            }
            Ok(_) => {}
            Err(e) => error!("Failed to evict entries at boot: {}", e),
        }
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
        ttl_seconds,
        max_entries,
        exact_only_models,
        metrics,
        shadow,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .route("/metrics", get(get_metrics))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, add_policy_header));

    let port = std::env::var("PORT")
        .ok()
        .or(file_config.port)
        .unwrap_or_else(|| "8080".to_string());
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    info!("veritas-cache listening on 127.0.0.1:{}", port);
    axum::serve(listener, app).await?;

    Ok(())
}
