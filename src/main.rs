use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tracing::{error, info};

// The application state contains the database and the HTTP client.
#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
    client: reqwest::Client,
    upstream_base: String,
}

// OpenAI chat completion request structure.
// This structure tolerates extra fields and explicit stream handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(flatten)]
    extra: Value,
}

// A small error response for client problems.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// Return a simple health check response.
async fn health() -> &'static str {
    "ok"
}

// Convert a JSON value into canonical form with sorted object keys.
fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::with_capacity(map.len());
            for k in keys {
                sorted.insert(k.clone(), canonical_json(&map[k]));
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

// Encode raw bytes as lower case hexadecimal characters.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

// Build a stable SHA-256 cache key for a chat request.
fn cache_key(request: &ChatRequest) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(request)?;
    let canonical = canonical_json(&value);
    let json_bytes = serde_json::to_vec(&canonical)?;
    Ok(to_hex(&Sha256::digest(&json_bytes)))
}

// Create the cache table if it does not exist and switch to WAL mode.
fn init_db(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entries (
            key_hash TEXT PRIMARY KEY,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0
        )",
        [],
    )?;
    Ok(())
}

// Look up a cached response by key.
fn lookup(conn: &Connection, key: &str) -> rusqlite::Result<Option<String>> {
    let mut stmt =
        conn.prepare("SELECT response_json FROM entries WHERE key_hash = ?1")?;
    stmt.query_row([key], |row| row.get::<_, String>(0))
        .optional()
}

// Increase the hit counter for a cached entry.
fn increment_hits(conn: &Connection, key: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE entries SET hit_count = hit_count + 1 WHERE key_hash = ?1",
        [key],
    )?;
    Ok(())
}

// Store a request and its response in the cache.
fn store(
    conn: &Connection,
    key: &str,
    request_json: &str,
    response_json: &str,
) -> rusqlite::Result<()> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR REPLACE INTO entries
         (key_hash, request_json, response_json, created_at, hit_count)
         VALUES (?1, ?2, ?3, ?4, 0)",
        params![key, request_json, response_json, now],
    )?;
    Ok(())
}

// Handle POST /v1/chat/completions: exact-match cache or upstream proxy.
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    // Reject streaming requests because Phase 1 does not support them.
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

    // Check the cache first.
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
        info!("cache hit for key {}", key);
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("x-cache", "HIT")
            .body(response_json.into())
            .unwrap();
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

    // Store the response in the cache only when the upstream call succeeded.
    // Do not cache error responses. A cached error would poison later requests.
    if status.is_success() {
        let request_json = match serde_json::to_string(&canonical_json(&serde_json::to_value(&request).unwrap_or(Value::Null))) {
            Ok(j) => j,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        };

        if let Err(e) = {
            let conn = state.db.lock().await;
            store(&conn, &key, &request_json, &response_text)
        } {
            error!("Failed to store response: {}", e);
        }
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
    tracing_subscriber::fmt::init();

    let db_path = std::env::var("CACHE_DB_PATH").unwrap_or_else(|_| "cache.db".to_string());
    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    let upstream_base = std::env::var("UPSTREAM_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let state = AppState {
        db,
        client,
        upstream_base,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_object_keys() {
        let a = json!({"z": 1, "a": 2, "m": {"b": 1, "a": 2}});
        let b = json!({"a": 2, "m": {"a": 2, "b": 1}, "z": 1});
        assert_eq!(canonical_json(&a), canonical_json(&b));
    }

    #[test]
    fn cache_key_is_stable_across_field_order() {
        let req_a: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7
        }))
        .unwrap();

        let req_b: ChatRequest = serde_json::from_value(json!({
            "temperature": 0.7,
            "messages": [{"content": "hello", "role": "user"}],
            "model": "gpt-4o-mini"
        }))
        .unwrap();

        assert_eq!(cache_key(&req_a).unwrap(), cache_key(&req_b).unwrap());
    }

    #[test]
    fn exact_match_hit_after_miss() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();

        let request = ChatRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![json!({"role": "user", "content": "what is 2+2"})],
            stream: None,
            extra: Value::Object(Default::default()),
        };

        let key = cache_key(&request).unwrap();
        assert!(lookup(&conn, &key).unwrap().is_none());

        let request_json =
            serde_json::to_string(&canonical_json(&serde_json::to_value(&request).unwrap())).unwrap();
        let response_json = r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"4"}}]}"#;
        store(&conn, &key, &request_json, response_json).unwrap();

        let cached = lookup(&conn, &key).unwrap().unwrap();
        assert!(cached.contains("chatcmpl-test"));

        increment_hits(&conn, &key).unwrap();
        let mut stmt = conn.prepare("SELECT hit_count FROM entries WHERE key_hash = ?1").unwrap();
        let count: i64 = stmt.query_row([&key], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }
}
