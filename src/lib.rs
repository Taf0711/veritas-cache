use hnsw_rs::prelude::*;
use ort::{session::Session, value::Tensor};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tracing::error;

pub mod adaptive;
pub mod policy;
pub mod replay;

// OpenAI chat completion request structure.
// This structure tolerates extra fields and explicit stream handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: Value,
}

// A small error response for client problems.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// Extract the response content used by the strict proxy judge.
pub fn response_content(response_json: &str) -> Option<String> {
    let value: Value = serde_json::from_str(response_json).ok()?;
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(|content| content.split_whitespace().collect::<Vec<_>>().join(" "))
}

// Compare two completion contents after whitespace normalization.
pub fn judge_content_equal(cached_json: &str, fresh_json: &str) -> bool {
    match (response_content(cached_json), response_content(fresh_json)) {
        (Some(cached), Some(fresh)) => cached == fresh,
        _ => false,
    }
}

// Assemble a non-streaming chat completion from buffered OpenAI SSE text.
// Return None when the stream is not valid OpenAI SSE.
pub fn assemble_from_sse(sse_text: &str) -> Option<Value> {
    let mut id: Option<String> = None;
    let mut created: Option<i64> = None;
    let mut model: Option<String> = None;
    let mut role: Option<String> = None;
    let mut content = String::new();
    // Tool calls accumulate by index.
    let mut tool_calls: Vec<(usize, Value)> = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Value> = None;

    for line in sse_text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" || data.is_empty() {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).ok()?;
        if id.is_none() {
            id = chunk.get("id").and_then(Value::as_str).map(String::from);
        }
        if created.is_none() {
            created = chunk.get("created").and_then(Value::as_i64);
        }
        if model.is_none() {
            model = chunk.get("model").and_then(Value::as_str).map(String::from);
        }
        // Skip null usage fields. Intermediate chunks carry null when
        // stream_options.include_usage is set. Only the final chunk has the object.
        if usage.is_none() {
            usage = chunk.get("usage").filter(|u| !u.is_null()).cloned();
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };
        if let Some(delta) = choice.get("delta") {
            if role.is_none() {
                role = delta.get("role").and_then(Value::as_str).map(String::from);
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                content.push_str(text);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    match tool_calls.iter_mut().find(|(i, _)| *i == index) {
                        Some((_, acc)) => merge_tool_call(acc, call),
                        None => {
                            let mut acc = json!({});
                            merge_tool_call(&mut acc, call);
                            tool_calls.push((index, acc));
                        }
                    }
                }
            }
        }
        if finish_reason.is_none() {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                if !reason.is_empty() {
                    finish_reason = Some(reason.to_string());
                }
            }
        }
    }

    let mut message = serde_json::Map::new();
    message.insert(
        "role".to_string(),
        Value::String(role.unwrap_or_else(|| "assistant".to_string())),
    );
    message.insert("content".to_string(), Value::String(content));
    if !tool_calls.is_empty() {
        tool_calls.sort_by_key(|(i, _)| *i);
        let calls: Vec<Value> = tool_calls.into_iter().map(|(_, call)| call).collect();
        message.insert("tool_calls".to_string(), Value::Array(calls));
    }

    let mut choice = serde_json::Map::new();
    choice.insert("index".to_string(), Value::from(0));
    choice.insert("message".to_string(), Value::Object(message));
    choice.insert(
        "finish_reason".to_string(),
        Value::String(finish_reason.unwrap_or_else(|| "stop".to_string())),
    );

    let mut completion = serde_json::Map::new();
    completion.insert("id".to_string(), Value::String(id.unwrap_or_default()));
    completion.insert(
        "object".to_string(),
        Value::String("chat.completion".to_string()),
    );
    completion.insert(
        "created".to_string(),
        created.map(Value::from).unwrap_or_else(|| Value::from(0)),
    );
    completion.insert(
        "model".to_string(),
        model.unwrap_or_default().into(),
    );
    completion.insert("choices".to_string(), Value::Array(vec![Value::Object(choice)]));
    if let Some(u) = usage {
        completion.insert("usage".to_string(), u);
    }
    Some(Value::Object(completion))
}

// Merge one tool-call delta fragment into the accumulated call for its index.
fn merge_tool_call(acc: &mut Value, fragment: &Value) {
    if let Some(obj) = acc.as_object_mut() {
        if let Some(id) = fragment.get("id").and_then(Value::as_str) {
            obj.insert("id".to_string(), Value::String(id.to_string()));
        }
        if let Some(call_type) = fragment.get("type").and_then(Value::as_str) {
            obj.insert("type".to_string(), Value::String(call_type.to_string()));
        }
        if let Some(func) = fragment.get("function") {
            let entry = obj
                .entry("function".to_string())
                .or_insert_with(|| json!({}));
            if let Some(fn_obj) = entry.as_object_mut() {
                if let Some(name) = func.get("name").and_then(Value::as_str) {
                    fn_obj.insert("name".to_string(), Value::String(name.to_string()));
                }
                if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                    let existing = fn_obj
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let merged = format!("{existing}{args}");
                    fn_obj.insert("arguments".to_string(), Value::String(merged));
                }
            }
        }
    }
}

// Render a stored chat completion as OpenAI SSE text.
// Return None when the stored JSON is not a valid completion.
pub fn render_sse(completion_json: &str) -> Option<String> {
    let value: Value = serde_json::from_str(completion_json).ok()?;
    let id = value
        .get("id")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let created = value.get("created").cloned().unwrap_or_else(|| Value::from(0));
    let model = value
        .get("model")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let choice = value.get("choices")?.as_array()?.first()?;
    let message = choice.get("message")?;
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("assistant")
        .to_string();
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let tool_calls = message.get("tool_calls").cloned();
    let finish_reason = choice
        .get("finish_reason")
        .cloned()
        .unwrap_or_else(|| Value::String("stop".to_string()));
    let usage = value.get("usage").cloned();

    let mut delta = serde_json::Map::new();
    delta.insert("role".to_string(), Value::String(role));
    delta.insert("content".to_string(), Value::String(content));
    if let Some(calls) = tool_calls {
        delta.insert("tool_calls".to_string(), calls);
    }

    let mut out = String::new();
    let first = json!({
        "id": id.clone(),
        "object": "chat.completion.chunk",
        "created": created.clone(),
        "model": model.clone(),
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
    });
    out.push_str(&format!("data: {first}\n\n"));
    let second = json!({
        "id": id.clone(),
        "object": "chat.completion.chunk",
        "created": created.clone(),
        "model": model.clone(),
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
    });
    out.push_str(&format!("data: {second}\n\n"));
    if let Some(u) = usage {
        let third = json!({
            "id": id.clone(),
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": u
        });
        out.push_str(&format!("data: {third}\n\n"));
    }
    out.push_str("data: [DONE]\n\n");
    Some(out)
}

// Embedding model and tokenizer.
// Session::run needs mut self, so embedder access uses a Mutex.
pub struct Embedder {
    tokenizer: Tokenizer,
    session: Session,
}

// Convert a JSON value into canonical form with sorted object keys.
pub fn canonical_json(value: &Value) -> Value {
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
pub fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

// Build a stable SHA-256 cache key for a chat request.
// The stream fields do not change the response content.
// Streaming and non-streaming variants share one cache entry.
pub fn cache_key(request: &ChatRequest) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(request)?;
    let canonical = canonical_json(&value);
    let mut cleaned = canonical;
    if let Some(obj) = cleaned.as_object_mut() {
        obj.remove("stream");
        obj.remove("stream_options");
    }
    let json_bytes = serde_json::to_vec(&cleaned)?;
    Ok(to_hex(&Sha256::digest(&json_bytes)))
}

// Join all messages into one embedding string.
// Each message becomes one line in the form "role: content".
pub fn prompt_text(request: &ChatRequest) -> String {
    let mut lines = Vec::with_capacity(request.messages.len());
    for msg in &request.messages {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();
        let content = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        lines.push(format!("{}: {}", role, content));
    }
    lines.join("\n")
}

// Create the embedder from model files on disk.
// Exit with a clear error if the files are missing.
pub fn build_embedder() -> Result<Embedder, Box<dyn std::error::Error + Send + Sync>> {
    let model_path = "models/model.onnx";
    let tokenizer_path = "models/tokenizer.json";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tokenizer_path).exists()
    {
        eprintln!(
            "Model files are missing. Run ./scripts/fetch_model.sh to download them. \
             Expected files: {} and {}",
            model_path, tokenizer_path
        );
        std::process::exit(1);
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let session = Session::builder()?.commit_from_file(model_path)?;

    Ok(Embedder { tokenizer, session })
}

// Count the tokens of a text with the embedding tokenizer.
pub fn count_tokens(embedder: &Embedder, text: &str) -> Result<usize, String> {
    let encoding = embedder.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
    Ok(encoding.get_ids().len())
}

// Parse a comma-separated list of exact-only model names.
pub fn parse_exact_only_models(value: &str) -> std::collections::HashSet<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

// Optional JSON config file. Every key is optional.
// Environment variables win over these values.
#[derive(Debug, Default, Deserialize)]
pub struct FileConfig {
    pub port: Option<String>,
    pub db_path: Option<String>,
    pub upstream_base_url: Option<String>,
    pub semantic_policy: Option<String>,
    pub semantic_threshold: Option<f32>,
    pub ttl_seconds: Option<i64>,
    pub max_entries: Option<i64>,
    pub exact_only_models: Option<Vec<String>>,
    pub shadow: Option<bool>,
}

// Parse the JSON text of a config file.
pub fn parse_file_config(text: &str) -> Result<FileConfig, String> {
    serde_json::from_str(text).map_err(|e| format!("Invalid config JSON: {}", e))
}

// Load the config file named by CACHE_CONFIG. Return defaults when unset.
pub fn load_file_config() -> Result<FileConfig, String> {
    let Ok(path) = std::env::var("CACHE_CONFIG") else {
        return Ok(FileConfig::default());
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read CACHE_CONFIG file {}: {}", path, e))?;
    parse_file_config(&text)
        .map_err(|e| format!("Failed to parse CACHE_CONFIG file {}: {}", path, e))
}

// Rewrite the usage of a cached response for a new request.
// Prompt tokens come from the new request. Completion tokens stay from the stored response.
// Return the input unchanged when the response has no usage object.
pub fn synthesize_usage(response_json: &str, prompt_tokens: usize) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(response_json) else {
        return response_json.to_string();
    };
    let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) else {
        return response_json.to_string();
    };
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    usage.insert("prompt_tokens".to_string(), Value::from(prompt_tokens as u64));
    usage.insert(
        "total_tokens".to_string(),
        Value::from(prompt_tokens as u64 + completion),
    );
    value.to_string()
}

// Compute a sentence embedding.
// Tokenize the prompt, run the ONNX model, mean pool the last hidden state, and L2-normalize.
pub fn embed(
    embedder: &mut Embedder,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let encoding = embedder
        .tokenizer
        .encode(text, true)
        .map_err(|e| e.to_string())?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as i64)
        .collect();
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    let len = input_ids.len();

    let outputs = embedder.session.run(ort::inputs! {
        "input_ids" => Tensor::from_array(([1, len], input_ids))?,
        "attention_mask" => Tensor::from_array(([1, len], attention_mask.clone()))?,
        "token_type_ids" => Tensor::from_array(([1, len], token_type_ids))?,
    })?;

    let (shape, raw) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    if shape.len() != 3 {
        return Err("Unexpected hidden state shape".into());
    }
    let seq_len = shape[1];
    let hidden_dim = shape[2];
    if raw.len() != seq_len * hidden_dim {
        return Err("Hidden state size mismatch".into());
    }

    // Mean pooling over sequence positions weighted by the attention mask.
    let mut pooled = vec![0.0f32; hidden_dim];
    let mut mask_sum = 0.0f32;
    for i in 0..seq_len {
        let mask = attention_mask[i] as f32;
        mask_sum += mask;
        for j in 0..hidden_dim {
            pooled[j] += raw[i * hidden_dim + j] * mask;
        }
    }
    if mask_sum > 0.0 {
        for val in &mut pooled {
            *val /= mask_sum;
        }
    }

    // L2-normalize the pooled vector.
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut pooled {
            *val /= norm;
        }
    }

    Ok(pooled)
}

// Convert an embedding vector into a little-endian f32 blob.
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for &v in embedding {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    blob
}

// Convert a little-endian f32 blob back into a vector.
pub fn embedding_from_blob(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.is_empty() || blob.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(bytes));
    }
    Some(out)
}

// Cosine similarity between two L2-normalized vectors.
// The result is the dot product because both inputs are unit length.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Decide whether a semantic hit is above the configured threshold.
pub fn semantic_hit(similarity: f32, threshold: f32) -> bool {
    similarity >= threshold
}

// Return the current unix time in seconds.
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// Create the cache tables if they do not exist and switch to WAL mode.
// Add the last_accessed_at column to databases that predate it.
pub fn init_db(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entries (
            key_hash TEXT PRIMARY KEY,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            last_accessed_at INTEGER NOT NULL DEFAULT 0
        )",
        [],
    )?;
    let has_column = {
        let mut stmt = conn.prepare("PRAGMA table_info(entries)")?;
        let mut rows = stmt.query([])?;
        let mut found = false;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == "last_accessed_at" {
                found = true;
            }
        }
        found
    };
    if !has_column {
        conn.execute(
            "ALTER TABLE entries ADD COLUMN last_accessed_at INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    conn.execute(
        "CREATE TABLE IF NOT EXISTS observations (
            entry_rowid INTEGER NOT NULL,
            similarity REAL NOT NULL,
            correct INTEGER NOT NULL,
            scope TEXT
        )",
        [],
    )?;
    // Add the scope column to databases that predate it.
    let has_scope = {
        let mut stmt = conn.prepare("PRAGMA table_info(observations)")?;
        let mut rows = stmt.query([])?;
        let mut found = false;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == "scope" {
                found = true;
            }
        }
        found
    };
    if !has_scope {
        conn.execute("ALTER TABLE observations ADD COLUMN scope TEXT", [])?;
    }
    conn.execute(
        "CREATE TABLE IF NOT EXISTS shadow_log (
            id INTEGER PRIMARY KEY,
            ts INTEGER NOT NULL,
            key_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            decision TEXT NOT NULL,
            similarity REAL,
            would_serve_json TEXT,
            fresh_json TEXT,
            request_json TEXT
        )",
        [],
    )?;
    migrate_shadow_log(conn)?;
    Ok(())
}

// Look up a cached response by key.
pub fn lookup(conn: &Connection, key: &str) -> rusqlite::Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT response_json FROM entries WHERE key_hash = ?1")?;
    stmt.query_row([key], |row| row.get::<_, String>(0))
        .optional()
}

// Load a response by SQLite rowid.
pub fn lookup_by_rowid(conn: &Connection, rowid: i64) -> rusqlite::Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT response_json FROM entries WHERE rowid = ?1")?;
    stmt.query_row([rowid], |row| row.get::<_, String>(0))
        .optional()
}

// Load the model name by SQLite rowid.
pub fn model_by_rowid(conn: &Connection, rowid: i64) -> rusqlite::Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT model FROM entries WHERE rowid = ?1")?;
    stmt.query_row([rowid], |row| row.get::<_, String>(0))
        .optional()
}

// Read the prompt cache key of a stored entry. Agent clients like Splice send
// this field to scope one run and stage. Semantic hits must stay inside one scope.
pub fn prompt_cache_key_by_rowid(
    conn: &Connection,
    rowid: i64,
) -> rusqlite::Result<Option<String>> {
    let mut stmt = conn.prepare(
        "SELECT json_extract(request_json, '$.prompt_cache_key') FROM entries WHERE rowid = ?1",
    )?;
    stmt.query_row([rowid], |row| row.get::<_, Option<String>>(0))
        .optional()
        .map(|row| row.flatten())
}

// Increase the hit counter for a cached entry and record the access time.
pub fn increment_hits(conn: &Connection, key: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE entries SET hit_count = hit_count + 1, last_accessed_at = ?2 WHERE key_hash = ?1",
        params![key, unix_now()],
    )?;
    Ok(())
}

// Increase the hit counter for a semantic match and record the access time.
pub fn increment_hits_by_rowid(conn: &Connection, rowid: i64) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE entries SET hit_count = hit_count + 1, last_accessed_at = ?2 WHERE rowid = ?1",
        params![rowid, unix_now()],
    )?;
    Ok(())
}

// Append one adaptive-policy observation for a cache entry.
// The scope is the prompt cache key of the request when one exists.
pub fn insert_observation(
    conn: &Connection,
    entry_rowid: i64,
    similarity: f32,
    correct: bool,
    scope: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO observations (entry_rowid, similarity, correct, scope) VALUES (?1, ?2, ?3, ?4)",
        params![entry_rowid, similarity, correct as i64, scope],
    )?;
    Ok(())
}

// Load every observation in write order for a boot-time policy replay.
// Return (entry_rowid, scope, similarity, correct) per row.
pub fn load_observations(
    conn: &Connection,
) -> rusqlite::Result<Vec<(i64, Option<String>, f32, bool)>> {
    let mut stmt = conn.prepare(
        "SELECT entry_rowid, scope, similarity, correct FROM observations ORDER BY rowid",
    )?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let correct: i64 = row.get(3)?;
        out.push((row.get(0)?, row.get(1)?, row.get(2)?, correct != 0));
    }
    Ok(out)
}

// Record one shadow-mode decision. Return the row id.
// The fresh response lands later through set_shadow_fresh.
pub fn insert_shadow_row(
    conn: &Connection,
    key_hash: &str,
    model: &str,
    decision: &str,
    similarity: Option<f32>,
    would_serve_json: Option<&str>,
    request_json: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO shadow_log
         (ts, key_hash, model, decision, similarity, would_serve_json, fresh_json, request_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7)",
        params![unix_now(), key_hash, model, decision, similarity, would_serve_json, request_json],
    )?;
    Ok(conn.last_insert_rowid())
}

// Add the request column to shadow logs that predate it.
pub fn migrate_shadow_log(conn: &Connection) -> rusqlite::Result<()> {
    let has_column = {
        let mut stmt = conn.prepare("PRAGMA table_info(shadow_log)")?;
        let mut rows = stmt.query([])?;
        let mut found = false;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == "request_json" {
                found = true;
            }
        }
        found
    };
    if !has_column {
        conn.execute("ALTER TABLE shadow_log ADD COLUMN request_json TEXT", [])?;
    }
    Ok(())
}

// Attach the fresh upstream response to a shadow row.
pub fn set_shadow_fresh(conn: &Connection, id: i64, fresh_json: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE shadow_log SET fresh_json = ?2 WHERE id = ?1",
        params![id, fresh_json],
    )?;
    Ok(())
}

// Store a request, its response, the model and the embedding in the cache.
// Return the new SQLite rowid.
pub fn store(
    conn: &Connection,
    key: &str,
    request_json: &str,
    response_json: &str,
    model: &str,
    embedding: &[f32],
) -> rusqlite::Result<i64> {
    let now = unix_now();
    conn.execute(
        "INSERT OR REPLACE INTO entries
         (key_hash, request_json, response_json, created_at, hit_count, model, embedding, last_accessed_at)
         VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6, ?4)",
        params![
            key,
            request_json,
            response_json,
            now,
            model,
            embedding_to_blob(embedding)
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

// Delete expired and least recently used entries from the cache.
// A ttl_seconds of 0 disables expiry. A max_entries of 0 disables the cap.
// Delete the observations of each removed entry as well.
// Return the number of deleted entries.
pub fn evict(
    conn: &Connection,
    ttl_seconds: i64,
    max_entries: i64,
    now_unix: i64,
) -> rusqlite::Result<usize> {
    let mut dead: Vec<i64> = Vec::new();
    if ttl_seconds > 0 {
        let cutoff = now_unix - ttl_seconds;
        let mut stmt = conn.prepare("SELECT rowid FROM entries WHERE created_at < ?1")?;
        let mut rows = stmt.query([cutoff])?;
        while let Some(row) = rows.next()? {
            dead.push(row.get(0)?);
        }
    }
    delete_entries(conn, &dead)?;
    let mut deleted = dead.len();

    if max_entries > 0 {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))?;
        let excess = count - max_entries;
        if excess > 0 {
            let mut victims: Vec<i64> = Vec::new();
            let mut stmt = conn.prepare(
                "SELECT rowid FROM entries ORDER BY last_accessed_at ASC, rowid ASC LIMIT ?1",
            )?;
            let mut rows = stmt.query([excess])?;
            while let Some(row) = rows.next()? {
                victims.push(row.get(0)?);
            }
            delete_entries(conn, &victims)?;
            deleted += victims.len();
        }
    }
    Ok(deleted)
}

// Delete the given entries and their observations by rowid.
fn delete_entries(conn: &Connection, rowids: &[i64]) -> rusqlite::Result<()> {
    for rowid in rowids {
        conn.execute("DELETE FROM entries WHERE rowid = ?1", [rowid])?;
        conn.execute("DELETE FROM observations WHERE entry_rowid = ?1", [rowid])?;
    }
    Ok(())
}

// Build the in-memory HNSW index from stored embeddings.
pub fn build_index(conn: &Connection) -> rusqlite::Result<Hnsw<'static, f32, DistCosine>> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))?;
    let capacity = (count as usize).max(100);
    let index = Hnsw::new(16, capacity, 16, 100, DistCosine {});
    let mut stmt = conn.prepare("SELECT rowid, embedding FROM entries")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let rowid: i64 = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        if let Some(embedding) = embedding_from_blob(&blob) {
            index.insert((&embedding, rowid as usize));
        } else {
            error!("Skipping invalid embedding blob for rowid {}", rowid);
        }
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn judge_content_ignores_whitespace_and_ids() {
        let cached = r#"{"id":"old","choices":[{"message":{"content":"hello   world"}}]}"#;
        let fresh = r#"{"id":"new","choices":[{"message":{"content":" hello world "}}]}"#;
        assert!(judge_content_equal(cached, fresh));
    }

    #[test]
    fn judge_content_rejects_different_content() {
        let cached = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let fresh = r#"{"choices":[{"message":{"content":"goodbye"}}]}"#;
        assert!(!judge_content_equal(cached, fresh));
    }

    #[test]
    fn judge_content_rejects_invalid_json() {
        let valid = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        assert!(!judge_content_equal("invalid", valid));
        assert!(!judge_content_equal(valid, "invalid"));
    }

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
    fn parse_file_config_reads_full_and_partial_json() {
        let full = parse_file_config(
            r#"{"port": "18080", "db_path": "x.db", "upstream_base_url": "http://u",
                "semantic_policy": "ld3", "semantic_threshold": 0.9, "ttl_seconds": 60,
                "max_entries": 100, "exact_only_models": ["gpt-4o-mini"]}"#,
        )
        .unwrap();
        assert_eq!(full.port.as_deref(), Some("18080"));
        assert_eq!(full.db_path.as_deref(), Some("x.db"));
        assert_eq!(full.upstream_base_url.as_deref(), Some("http://u"));
        assert_eq!(full.semantic_policy.as_deref(), Some("ld3"));
        assert_eq!(full.semantic_threshold, Some(0.9));
        assert_eq!(full.ttl_seconds, Some(60));
        assert_eq!(full.max_entries, Some(100));
        assert_eq!(full.exact_only_models.as_ref().unwrap().len(), 1);

        let partial = parse_file_config(r#"{"port": "18081"}"#).unwrap();
        assert_eq!(partial.port.as_deref(), Some("18081"));
        assert!(partial.semantic_policy.is_none());

        assert!(parse_file_config("not json").is_err());
    }

    #[test]
    fn embedding_from_blob_rejects_empty_and_misaligned_blobs() {
        assert!(embedding_from_blob(&[]).is_none());
        assert!(embedding_from_blob(&[0, 1, 2]).is_none());
        assert!(embedding_from_blob(&[0, 0, 0, 0]).is_some());
    }

    #[test]
    fn cache_key_differs_by_tool_choice() {
        let forced: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "call the tool"}],
            "tool_choice": {"type": "function", "function": {"name": "run"}}
        }))
        .unwrap();
        let auto: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "call the tool"}],
            "tool_choice": "auto"
        }))
        .unwrap();
        let plain: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "call the tool"}]
        }))
        .unwrap();

        let forced_key = cache_key(&forced).unwrap();
        assert_ne!(forced_key, cache_key(&auto).unwrap());
        assert_ne!(forced_key, cache_key(&plain).unwrap());
    }

    #[test]
    fn synthesize_usage_rewrites_prompt_and_total() {
        let stored = json!({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let rewritten = synthesize_usage(&stored.to_string(), 42);
        let value: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(value["usage"]["prompt_tokens"], 42);
        assert_eq!(value["usage"]["completion_tokens"], 5);
        assert_eq!(value["usage"]["total_tokens"], 47);

        // A response without usage stays unchanged.
        let bare = json!({"id": "chatcmpl-2", "choices": []});
        assert_eq!(synthesize_usage(&bare.to_string(), 42), bare.to_string());
    }

    #[test]
    fn parse_exact_only_models_trims_and_drops_empty() {
        let set = parse_exact_only_models(" gpt-4o-mini , ,gpt-4o,");
        assert!(set.contains("gpt-4o-mini"));
        assert!(set.contains("gpt-4o"));
        assert_eq!(set.len(), 2);
        assert!(parse_exact_only_models("").is_empty());
    }

    #[test]
    fn cache_key_ignores_stream_fields() {
        let streaming: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .unwrap();

        let plain: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        assert_eq!(cache_key(&streaming).unwrap(), cache_key(&plain).unwrap());
    }

    // Build one SSE data line from a chunk value.
    fn sse_line(chunk: Value) -> String {
        format!("data: {}\n\n", chunk)
    }

    #[test]
    fn assemble_from_sse_builds_completion_with_tool_calls() {
        let mut sse = String::new();
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            })
        };
        sse.push_str(&sse_line(chunk(
            json!({"role": "assistant", "content": ""}),
            None,
        )));
        sse.push_str(&sse_line(chunk(json!({"content": "Hello"}), None)));
        sse.push_str(&sse_line(chunk(
            json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{\"cmd\":"}}]}), 
            None,
        )));
        sse.push_str(&sse_line(chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"ls\"}"}}]}), 
            None,
        )));
        sse.push_str(&sse_line(chunk(json!({}), Some("tool_calls"))));
        let usage_chunk = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        sse.push_str(&sse_line(usage_chunk));
        sse.push_str("data: [DONE]\n\n");

        let assembled = assemble_from_sse(&sse).unwrap();
        assert_eq!(assembled["id"], "chatcmpl-1");
        assert_eq!(assembled["model"], "gpt-4o-mini");
        assert_eq!(assembled["created"], 123);
        assert_eq!(assembled["object"], "chat.completion");
        let choice = &assembled["choices"][0];
        assert_eq!(choice["message"]["role"], "assistant");
        assert_eq!(choice["message"]["content"], "Hello");
        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "run");
        assert_eq!(call["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(assembled["usage"]["total_tokens"], 15);
    }

    #[test]
    fn render_sse_roundtrips_through_assemble() {
        let completion = json!({
            "id": "chatcmpl-9",
            "object": "chat.completion",
            "created": 456,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "four"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        });

        let sse = render_sse(&completion.to_string()).unwrap();
        assert!(sse.ends_with("data: [DONE]\n\n"));

        let rebuilt = assemble_from_sse(&sse).unwrap();
        assert_eq!(rebuilt["id"], "chatcmpl-9");
        assert_eq!(rebuilt["model"], "gpt-4o-mini");
        assert_eq!(rebuilt["created"], 456);
        assert_eq!(rebuilt["choices"][0]["message"]["content"], "four");
        assert_eq!(rebuilt["choices"][0]["finish_reason"], "stop");
        assert_eq!(rebuilt["usage"]["total_tokens"], 3);
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
            serde_json::to_string(&canonical_json(&serde_json::to_value(&request).unwrap()))
                .unwrap();
        let response_json = r#"{"id":"chatcmpl-test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"4"}}]}"#;
        let embedding = vec![0.1f32; 384];
        store(
            &conn,
            &key,
            &request_json,
            response_json,
            &request.model,
            &embedding,
        )
        .unwrap();

        let cached = lookup(&conn, &key).unwrap().unwrap();
        assert!(cached.contains("chatcmpl-test"));

        increment_hits(&conn, &key).unwrap();
        let mut stmt = conn
            .prepare("SELECT hit_count FROM entries WHERE key_hash = ?1")
            .unwrap();
        let count: i64 = stmt.query_row([&key], |row| row.get(0)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn prompt_cache_key_by_rowid_reads_the_scope() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let request = json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "session-1:code_writer"
        });
        let rowid = store(
            &conn,
            "key-scope",
            &request.to_string(),
            "{}",
            "mock",
            &[1.0, 0.0],
        )
        .unwrap();
        let scoped = prompt_cache_key_by_rowid(&conn, rowid).unwrap();
        assert_eq!(scoped.as_deref(), Some("session-1:code_writer"));

        // An entry without the field reads as None.
        let bare = json!({"model": "mock", "messages": []});
        let rowid2 = store(&conn, "key-bare", &bare.to_string(), "{}", "mock", &[1.0, 0.0])
            .unwrap();
        assert_eq!(prompt_cache_key_by_rowid(&conn, rowid2).unwrap(), None);
    }

    #[test]
    fn shadow_row_roundtrip() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();

        let id = insert_shadow_row(&conn, "key-a", "mock", "exact_hit", None, Some("{\"old\":1}"), "{}")
            .unwrap();
        set_shadow_fresh(&conn, id, "{\"new\":2}").unwrap();

        let row: (i64, String, String, String, Option<f32>, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT ts, key_hash, model, decision, similarity, would_serve_json, fresh_json
                 FROM shadow_log WHERE id = ?1",
                [id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .unwrap();
        assert!(row.0 > 0);
        assert_eq!(row.1, "key-a");
        assert_eq!(row.2, "mock");
        assert_eq!(row.3, "exact_hit");
        assert_eq!(row.4, None);
        assert_eq!(row.5.as_deref(), Some("{\"old\":1}"));
        assert_eq!(row.6.as_deref(), Some("{\"new\":2}"));

        // A miss row carries a similarity when a neighbor was rejected.
        let id2 = insert_shadow_row(&conn, "key-b", "mock", "miss", Some(0.5), None, "{}").unwrap();
        let sim: Option<f32> = conn
            .query_row(
                "SELECT similarity FROM shadow_log WHERE id = ?1",
                [id2],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(sim, Some(0.5));
    }

    #[test]
    fn observations_roundtrip_in_write_order() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();

        insert_observation(&conn, 7, 0.91, true, Some("sess-1:code_writer")).unwrap();
        insert_observation(&conn, 7, 0.72, false, Some("sess-2:code_writer")).unwrap();
        insert_observation(&conn, 11, 0.88, true, None).unwrap();

        let rows = load_observations(&conn).unwrap();
        assert_eq!(rows.len(), 3);

        let for_seven: Vec<&(i64, Option<String>, f32, bool)> =
            rows.iter().filter(|r| r.0 == 7).collect();
        assert_eq!(for_seven.len(), 2);
        assert!((for_seven[0].2 - 0.91).abs() < 1e-6);
        assert!(for_seven[0].3);
        assert!((for_seven[1].2 - 0.72).abs() < 1e-6);
        assert!(!for_seven[1].3);

        let for_eleven: Vec<&(i64, Option<String>, f32, bool)> =
            rows.iter().filter(|r| r.0 == 11).collect();
        assert_eq!(for_eleven.len(), 1);
        assert!((for_eleven[0].2 - 0.88).abs() < 1e-6);
        assert!(for_eleven[0].3);

        // Scope roundtrip: two sessions of one stage group together. NULL stays NULL.
        let mut by_scope: std::collections::HashMap<Option<String>, usize> =
            std::collections::HashMap::new();
        for row in &rows {
            let scope = row.1.as_deref().map(|s| {
                let key = s.rsplit_once(':').map(|(_, stage)| stage).unwrap_or(s);
                key.to_string()
            });
            *by_scope.entry(scope).or_default() += 1;
        }
        assert_eq!(by_scope.get(&Some("code_writer".to_string())), Some(&2));
        assert_eq!(by_scope.get(&None), Some(&1));
    }

    // Insert one entry with explicit timestamps. Return its rowid.
    fn insert_entry(conn: &Connection, key: &str, created_at: i64, last_accessed_at: i64) -> i64 {
        conn.execute(
            "INSERT INTO entries
             (key_hash, request_json, response_json, created_at, hit_count, model, embedding, last_accessed_at)
             VALUES (?1, '{}', '{}', ?2, 0, 'mock', ?3, ?4)",
            params![key, created_at, embedding_to_blob(&[1.0, 0.0]), last_accessed_at],
        )
        .unwrap();
        conn.last_insert_rowid()
    }

    #[test]
    fn evict_expires_old_entries_and_their_observations() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let now = 1_000_000;
        let old = insert_entry(&conn, "old", 1_000, 1_000);
        let fresh = insert_entry(&conn, "fresh", 999_999, 999_999);
        insert_observation(&conn, old, 0.9, true, None).unwrap();
        insert_observation(&conn, fresh, 0.8, false, None).unwrap();

        let deleted = evict(&conn, 3600, 0, now).unwrap();
        assert_eq!(deleted, 1);
        assert!(lookup_by_rowid(&conn, old).unwrap().is_none());
        assert!(lookup_by_rowid(&conn, fresh).unwrap().is_some());
        let observations = load_observations(&conn).unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].0, fresh);
    }

    #[test]
    fn evict_keeps_the_most_recently_accessed_entries() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let oldest = insert_entry(&conn, "a", 100, 100);
        let middle = insert_entry(&conn, "b", 100, 200);
        let newest = insert_entry(&conn, "c", 100, 300);
        insert_observation(&conn, oldest, 0.9, true, None).unwrap();

        let deleted = evict(&conn, 0, 2, 1_000).unwrap();
        assert_eq!(deleted, 1);
        assert!(lookup_by_rowid(&conn, oldest).unwrap().is_none());
        assert!(lookup_by_rowid(&conn, middle).unwrap().is_some());
        assert!(lookup_by_rowid(&conn, newest).unwrap().is_some());
        assert!(load_observations(&conn).unwrap().is_empty());
    }

    #[test]
    fn init_db_adds_last_accessed_at_to_old_databases() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE entries (
                key_hash TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL
            )",
            [],
        )
        .unwrap();

        init_db(&conn).unwrap();
        let rowid = insert_entry(&conn, "a", 5, 7);
        assert!(lookup_by_rowid(&conn, rowid).unwrap().is_some());
        let accessed: i64 = conn
            .query_row(
                "SELECT last_accessed_at FROM entries WHERE rowid = ?1",
                [rowid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(accessed, 7);
    }

    #[test]
    fn evict_with_zero_limits_is_a_no_op() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let rowid = insert_entry(&conn, "a", 1, 1);

        let deleted = evict(&conn, 0, 0, 1_000_000).unwrap();
        assert_eq!(deleted, 0);
        assert!(lookup_by_rowid(&conn, rowid).unwrap().is_some());
    }

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn threshold_decision_is_pure_function() {
        assert!(semantic_hit(0.9, 0.85));
        assert!(!semantic_hit(0.84, 0.85));
    }

    #[test]
    fn embedding_blob_roundtrip() {
        let v: Vec<f32> = (0..384).map(|i| i as f32 / 100.0).collect();
        let blob = embedding_to_blob(&v);
        assert_eq!(blob.len(), v.len() * 4);
        let decoded = embedding_from_blob(&blob).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn embedding_blob_with_bad_length_returns_none() {
        assert!(embedding_from_blob(&[0, 1, 2]).is_none());
    }

    #[test]
    #[ignore = "requires model files in models/"]
    fn paraphrase_embeddings_are_similar() {
        let mut embedder = build_embedder().unwrap();
        let a = embed(&mut embedder, "How big is the universe?").unwrap();
        let b = embed(&mut embedder, "What is the size of the cosmos?").unwrap();
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim > 0.7,
            "expected paraphrase similarity > 0.7, got {}",
            sim
        );
    }

    // Write CSV rows to a file under bench/results. Create the directory.
    fn write_results_csv(filename: &str, rows: &[String]) {
        use std::io::Write;
        let dir = std::path::Path::new("bench/results");
        std::fs::create_dir_all(dir).unwrap();
        let mut handle = std::fs::File::create(dir.join(filename)).unwrap();
        for row in rows {
            writeln!(handle, "{}", row).unwrap();
        }
    }

    #[test]
    #[ignore = "requires model files and trace"]
    fn trace_similarity_separation() {
        use std::collections::{BTreeMap, HashMap, HashSet};

        let trace_path = "bench/trace.jsonl";
        let model_path = "models/model.onnx";
        if !std::path::Path::new(trace_path).exists() || !std::path::Path::new(model_path).exists()
        {
            eprintln!("Skipping. Missing {} or {}.", trace_path, model_path);
            return;
        }

        // Group the trace prompts by class id.
        let mut classes: BTreeMap<i64, Vec<String>> = BTreeMap::new();
        for line in std::fs::read_to_string(trace_path).unwrap().lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            let prompt = value["prompt"].as_str().unwrap().to_string();
            let class_id = value["class_id"].as_i64().unwrap();
            classes.entry(class_id).or_default().push(prompt);
        }

        // Sample same-class pairs. Take two members from each class in order.
        let mut same_pairs: Vec<(String, String)> = Vec::new();
        for members in classes.values() {
            for window in members.chunks(2) {
                if window.len() == 2 {
                    same_pairs.push((window[0].clone(), window[1].clone()));
                }
                if same_pairs.len() == 300 {
                    break;
                }
            }
            if same_pairs.len() == 300 {
                break;
            }
        }

        // Sample cross-class pairs. Use a fixed stride over the class order.
        let class_ids: Vec<i64> = classes.keys().cloned().collect();
        let stride = 57;
        let mut cross_pairs: Vec<(String, String)> = Vec::new();
        let mut index = 0;
        while cross_pairs.len() < 300 && index < class_ids.len() {
            let other = (index + stride) % class_ids.len();
            cross_pairs.push((
                classes[&class_ids[index]][0].clone(),
                classes[&class_ids[other]][0].clone(),
            ));
            index += 1;
        }

        // Embed every unique prompt exactly once.
        let mut unique: HashSet<&str> = HashSet::new();
        for (a, b) in same_pairs.iter().chain(cross_pairs.iter()) {
            unique.insert(a);
            unique.insert(b);
        }
        let mut embedder = build_embedder().unwrap();
        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        for text in unique {
            embeddings.insert(text.to_string(), embed(&mut embedder, text).unwrap());
        }

        // Compute the cosine similarity for every pair.
        let same_sims: Vec<f32> = same_pairs
            .iter()
            .map(|(a, b)| cosine_similarity(&embeddings[a], &embeddings[b]))
            .collect();
        let cross_sims: Vec<f32> = cross_pairs
            .iter()
            .map(|(a, b)| cosine_similarity(&embeddings[a], &embeddings[b]))
            .collect();

        // Write the raw pair similarities for the Python chart script.
        let mut rows: Vec<String> = Vec::with_capacity(same_sims.len() + cross_sims.len() + 1);
        rows.push("group,sim".to_string());
        for sim in &same_sims {
            rows.push(format!("same,{:.6}", sim));
        }
        for sim in &cross_sims {
            rows.push(format!("cross,{:.6}", sim));
        }
        write_results_csv("separation.csv", &rows);

        fn percentile(sorted: &[f32], fraction: f64) -> f32 {
            if sorted.is_empty() {
                return 0.0;
            }
            let position = ((sorted.len() - 1) as f64 * fraction) as usize;
            sorted[position]
        }

        fn stats(label: &str, sims: &[f32]) {
            let mut sorted = sims.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
            println!(
                "{}: mean {:.4} p5 {:.4} p50 {:.4} p95 {:.4} (n {})",
                label,
                mean,
                percentile(&sorted, 0.05),
                percentile(&sorted, 0.50),
                percentile(&sorted, 0.95),
                sorted.len()
            );
        }

        stats("same-class", &same_sims);
        stats("cross-class", &cross_sims);

        let same_mean = same_sims.iter().sum::<f32>() / same_sims.len() as f32;
        let cross_mean = cross_sims.iter().sum::<f32>() / cross_sims.len() as f32;
        assert!(
            same_mean - cross_mean >= 0.1,
            "expected same-class mean to exceed cross-class mean by 0.1, got difference {}",
            same_mean - cross_mean
        );
    }

    #[test]
    #[ignore = "requires model files and trace; slow"]
    fn trace_nearest_neighbor_difficulty() {
        let trace_path = "bench/trace.jsonl";
        if !std::path::Path::new(trace_path).exists()
            || !std::path::Path::new("models/model.onnx").exists()
            || !std::path::Path::new("models/tokenizer.json").exists()
        {
            eprintln!("Skipping. Missing {} or model files.", trace_path);
            return;
        }

        // Load the trace. Keep the original record order.
        let mut records: Vec<(String, i64)> = Vec::new();
        for line in std::fs::read_to_string(trace_path).unwrap().lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            let prompt = value["prompt"].as_str().unwrap().to_string();
            let class_id = value["class_id"].as_i64().unwrap();
            records.push((prompt, class_id));
        }
        let count = records.len();

        // Embed every prompt exactly once.
        let mut embedder = build_embedder().unwrap();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(count);
        for (prompt, _) in &records {
            embeddings.push(embed(&mut embedder, prompt).unwrap());
        }

        // Sample every 10th record as a query. This is deterministic.
        let query_indices: Vec<usize> = (0..count).step_by(10).collect();
        let queries = query_indices.len();

        // Scan every entry for each query. Track the best same-class and the
        // best different-class neighbor, and the overall best neighbor.
        let mut best_same: Vec<f32> = Vec::with_capacity(queries);
        let mut best_cross: Vec<f32> = Vec::with_capacity(queries);
        let mut best_neighbor_sim: Vec<f32> = Vec::with_capacity(queries);
        let mut best_neighbor_correct: Vec<bool> = Vec::with_capacity(queries);
        for &query_index in &query_indices {
            let query_class = records[query_index].1;
            let mut same_best = -2.0f32;
            let mut cross_best = -2.0f32;
            let mut neighbor_sim = -2.0f32;
            let mut neighbor_correct = false;
            for (index, (_, class_id)) in records.iter().enumerate() {
                if index == query_index {
                    continue;
                }
                let sim = cosine_similarity(&embeddings[query_index], &embeddings[index]);
                if *class_id == query_class {
                    if sim > same_best {
                        same_best = sim;
                    }
                } else if sim > cross_best {
                    cross_best = sim;
                }
                if sim > neighbor_sim {
                    neighbor_sim = sim;
                    neighbor_correct = *class_id == query_class;
                }
            }
            best_same.push(same_best);
            best_cross.push(cross_best);
            best_neighbor_sim.push(neighbor_sim);
            best_neighbor_correct.push(neighbor_correct);
        }

        // Write the raw per-query values for the Python chart script.
        let mut rows: Vec<String> = Vec::with_capacity(queries + 1);
        rows.push("query_index,class_id,best_same,best_cross,nn_sim,nn_same_class".to_string());
        for (i, &query_index) in query_indices.iter().enumerate() {
            rows.push(format!(
                "{},{},{:.6},{:.6},{:.6},{}",
                query_index,
                records[query_index].1,
                best_same[i],
                best_cross[i],
                best_neighbor_sim[i],
                best_neighbor_correct[i]
            ));
        }
        write_results_csv("nn_difficulty.csv", &rows);

        fn percentile(sorted: &[f32], fraction: f64) -> f32 {
            if sorted.is_empty() {
                return 0.0;
            }
            let position = ((sorted.len() - 1) as f64 * fraction) as usize;
            sorted[position]
        }

        fn stats(label: &str, sims: &[f32]) {
            let mut sorted = sims.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
            println!(
                "{}: mean {:.4} p5 {:.4} p50 {:.4} p95 {:.4} (n {})",
                label,
                mean,
                percentile(&sorted, 0.05),
                percentile(&sorted, 0.50),
                percentile(&sorted, 0.95),
                sorted.len()
            );
        }

        stats("best-same", &best_same);
        stats("best-cross", &best_cross);
        let mut sorted_same = best_same.clone();
        sorted_same.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sorted_cross = best_cross.clone();
        sorted_cross.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Report the share of queries with a confusable different-class neighbor.
        for threshold in [0.80f32, 0.85, 0.90] {
            let count = sorted_cross.iter().filter(|&&sim| sim >= threshold).count();
            println!(
                "confusable share: {:.2}% of queries have best_cross >= {:.2}",
                100.0 * count as f32 / queries as f32,
                threshold
            );
        }

        // Compute the static-threshold curve over the full cache.
        // A query is a hit when its best neighbor clears the threshold.
        println!("static-threshold curve (n={} queries):", queries);
        for threshold in [0.30f32, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99] {
            let mut hits = 0;
            let mut wrong = 0;
            for index in 0..queries {
                if best_neighbor_sim[index] >= threshold {
                    hits += 1;
                    if !best_neighbor_correct[index] {
                        wrong += 1;
                    }
                }
            }
            let hit_rate = hits as f64 / queries as f64;
            let false_hit_rate = wrong as f64 / queries as f64;
            println!(
                "t={:.2} hit={:.4} false={:.4} misses={}",
                threshold,
                hit_rate,
                false_hit_rate,
                queries - hits
            );
        }

        let same_p50 = percentile(&sorted_same, 0.50);
        let cross_p95 = percentile(&sorted_cross, 0.95);
        assert!(
            same_p50 > cross_p95,
            "expected best-same p50 {} to exceed best-cross p95 {}",
            same_p50,
            cross_p95
        );
    }
}
