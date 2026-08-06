use hnsw_rs::prelude::*;
use ort::{session::Session, value::Tensor};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
pub fn cache_key(request: &ChatRequest) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(request)?;
    let canonical = canonical_json(&value);
    let json_bytes = serde_json::to_vec(&canonical)?;
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
    if blob.len() % 4 != 0 {
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

// Create the cache table if it does not exist and switch to WAL mode.
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
            embedding BLOB NOT NULL
        )",
        [],
    )?;
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

// Increase the hit counter for a cached entry.
pub fn increment_hits(conn: &Connection, key: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE entries SET hit_count = hit_count + 1 WHERE key_hash = ?1",
        [key],
    )?;
    Ok(())
}

// Increase the hit counter for a semantic match.
pub fn increment_hits_by_rowid(conn: &Connection, rowid: i64) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE entries SET hit_count = hit_count + 1 WHERE rowid = ?1",
        [rowid],
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
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR REPLACE INTO entries
         (key_hash, request_json, response_json, created_at, hit_count, model, embedding)
         VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6)",
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
