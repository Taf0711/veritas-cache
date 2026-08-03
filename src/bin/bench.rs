use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;
use veritas_cache::policy::StaticPolicy;
use veritas_cache::replay::replay;
use veritas_cache::{build_embedder, embed};

const EMBEDDING_DIM: usize = 384;
const EMBEDDING_MAGIC: &[u8] = b"VERITAS_EMBEDDINGS_V1\0";
const DEFAULT_THRESHOLDS: &[f32] = &[0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99];

#[derive(Debug, Deserialize)]
struct TraceRecord {
    prompt: String,
    class_id: i64,
}

fn read_trace(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<TraceRecord>, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        records.push(serde_json::from_str::<TraceRecord>(&line)?);
        if limit.is_some_and(|value| records.len() >= value) {
            break;
        }
    }
    if records.is_empty() {
        return Err("trace has no records".into());
    }
    Ok(records)
}

fn read_u64(reader: &mut &[u8]) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    if reader.len() < 8 {
        return Err("embedding cache is truncated".into());
    }
    let value = u64::from_le_bytes(reader[..8].try_into()?);
    *reader = &reader[8..];
    Ok(value)
}

fn load_embeddings(
    path: &Path,
    requested: usize,
) -> Result<Option<Vec<Vec<f32>>>, Box<dyn std::error::Error + Send + Sync>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let mut rest = bytes.as_slice();
    if rest.len() < EMBEDDING_MAGIC.len() || &rest[..EMBEDDING_MAGIC.len()] != EMBEDDING_MAGIC {
        return Ok(None);
    }
    rest = &rest[EMBEDDING_MAGIC.len()..];
    let count = read_u64(&mut rest)? as usize;
    let dimension = read_u64(&mut rest)? as usize;
    if dimension != EMBEDDING_DIM || count < requested {
        return Ok(None);
    }
    let needed = count
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(4))
        .ok_or("embedding cache size overflow")?;
    if rest.len() != needed {
        return Ok(None);
    }
    let mut embeddings = Vec::with_capacity(requested);
    for row in rest[..requested * dimension * 4].chunks_exact(dimension * 4) {
        let mut embedding = Vec::with_capacity(dimension);
        for value in row.chunks_exact(4) {
            embedding.push(f32::from_le_bytes(value.try_into()?));
        }
        embeddings.push(embedding);
    }
    Ok(Some(embeddings))
}

fn write_embeddings(
    path: &Path,
    embeddings: &[Vec<f32>],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    file.write_all(EMBEDDING_MAGIC)?;
    file.write_all(&(embeddings.len() as u64).to_le_bytes())?;
    file.write_all(&(EMBEDDING_DIM as u64).to_le_bytes())?;
    for embedding in embeddings {
        if embedding.len() != EMBEDDING_DIM {
            return Err("embedding dimension does not match cache format".into());
        }
        for value in embedding {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_embed_times(
    path: &Path,
    count: usize,
) -> Result<Option<Vec<u64>>, Box<dyn std::error::Error + Send + Sync>> {
    if !path.exists() {
        return Ok(None);
    }
    let values: Vec<u64> = fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if values.len() < count {
        return Ok(None);
    }
    Ok(Some(values[..count].to_vec()))
}

fn write_embed_times(
    path: &Path,
    values: &[u64],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    for value in values {
        writeln!(file, "{value}")?;
    }
    Ok(())
}

fn load_or_build_embeddings(
    records: &[TraceRecord],
    cache_path: &Path,
    times_path: &Path,
) -> Result<(Vec<Vec<f32>>, Vec<u64>), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(embeddings) = load_embeddings(cache_path, records.len())? {
        if let Some(times) = read_embed_times(times_path, records.len())? {
            println!(
                "Loaded {} embeddings from {}.",
                records.len(),
                cache_path.display()
            );
            return Ok((embeddings, times));
        }
    }

    println!("Embedding {} trace prompts.", records.len());
    let mut embedder = build_embedder()?;
    let mut embeddings = Vec::with_capacity(records.len());
    let mut embed_times = Vec::with_capacity(records.len());
    for record in records {
        let started = Instant::now();
        embeddings.push(embed(&mut embedder, &record.prompt)?);
        embed_times.push(started.elapsed().as_micros() as u64);
    }
    write_embeddings(cache_path, &embeddings)?;
    write_embed_times(times_path, &embed_times)?;
    Ok((embeddings, embed_times))
}

fn normal_sample(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn load_miss_latencies(path: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
    if path.exists() {
        let values: Vec<f64> = fs::read_to_string(path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if !values.is_empty() {
            println!(
                "Loaded {} miss latencies from {}.",
                values.len(),
                path.display()
            );
            return Ok(values);
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut values = Vec::with_capacity(20_000);
    for _ in 0..20_000 {
        values.push((800.0f64.ln() + 0.6 * normal_sample(&mut rng)).exp());
    }
    let mut file = File::create(path)?;
    for value in &values {
        writeln!(file, "{value:.6}")?;
    }
    println!("Miss latency uses a seeded lognormal model, not a measurement.");
    Ok(values)
}

fn thresholds() -> Vec<f32> {
    if let Ok(value) = std::env::var("BENCH_THRESHOLDS") {
        value
            .split(',')
            .filter_map(|item| item.trim().parse().ok())
            .collect()
    } else {
        DEFAULT_THRESHOLDS.to_vec()
    }
}

fn percentile(histogram: &hdrhistogram::Histogram<u64>, percentile: f64) -> f64 {
    histogram.value_at_percentile(percentile) as f64
}

fn write_static_results(
    path: &Path,
    rows: &[(f32, veritas_cache::replay::ReplayResult)],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    writeln!(file, "threshold,queries,hits,hit_rate,false_hits,false_hit_rate,false_misses,false_miss_rate,p50_lookup_us,p99_lookup_us,p50_total_ms,p99_total_ms")?;
    for (threshold, result) in rows {
        let queries = result.queries as f64;
        writeln!(
            file,
            "{threshold:.2},{},{},{:.6},{},{:.6},{},{:.6},{:.0},{:.0},{:.3},{:.3}",
            result.queries,
            result.hits,
            result.hits as f64 / queries,
            result.false_hits,
            result.false_hits as f64 / queries,
            result.false_misses,
            result.false_misses as f64 / queries,
            percentile(&result.lookup_us, 50.0),
            percentile(&result.lookup_us, 99.0),
            percentile(&result.total_us, 50.0) / 1_000.0,
            percentile(&result.total_us, 99.0) / 1_000.0,
        )?;
    }
    Ok(())
}

fn write_reference_log(
    path: &Path,
    result: &veritas_cache::replay::ReplayResult,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    writeln!(file, "id,sim,decision,correct,lookup_us")?;
    for event in &result.events {
        let sim = event.similarity.unwrap_or(-1.0);
        let decision = match event.decision {
            veritas_cache::policy::Decision::Hit => "HIT",
            veritas_cache::policy::Decision::Miss => "MISS",
        };
        let correct = event
            .correct
            .map(|value| value.to_string())
            .unwrap_or_default();
        writeln!(
            file,
            "{},{sim:.6},{decision},{correct},{}",
            event.id, event.lookup_us
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limit = std::env::var("TRACE_LIMIT")
        .ok()
        .map(|value| value.parse())
        .transpose()?;
    let records = read_trace(Path::new("bench/trace.jsonl"), limit)?;
    let count = records.len();
    fs::create_dir_all("bench/cache")?;
    fs::create_dir_all("bench/results")?;

    let (embeddings, embed_us) = load_or_build_embeddings(
        &records,
        Path::new("bench/cache/embeddings.bin"),
        Path::new("bench/cache/embed_times.txt"),
    )?;
    let classes: Vec<i64> = records.iter().map(|record| record.class_id).collect();
    let miss_latencies = load_miss_latencies(Path::new("bench/miss_latencies.txt"))?;
    let mut results = Vec::new();
    for threshold in thresholds() {
        let mut policy = StaticPolicy { threshold };
        results.push((
            threshold,
            replay(
                &classes,
                &embeddings,
                &embed_us,
                &mut policy,
                &miss_latencies,
            ),
        ));
    }
    write_static_results(Path::new("bench/results/stream_static.csv"), &results)?;

    if let Some((_, result)) = results
        .iter()
        .find(|(threshold, _)| (*threshold - 0.85).abs() < f32::EPSILON)
    {
        write_reference_log(Path::new("bench/results/stream_log.csv"), result)?;
    } else {
        let mut policy = StaticPolicy { threshold: 0.85 };
        let result = replay(
            &classes,
            &embeddings,
            &embed_us,
            &mut policy,
            &miss_latencies,
        );
        write_reference_log(Path::new("bench/results/stream_log.csv"), &result)?;
    }

    println!(
        "threshold | hit rate | false-hit rate | false-miss rate | p50 lookup us | p99 total ms"
    );
    for (threshold, result) in &results {
        let queries = result.queries as f64;
        println!(
            "{threshold:.2}      | {:.4}    | {:.4}          | {:.4}           | {:.0}          | {:.3}",
            result.hits as f64 / queries,
            result.false_hits as f64 / queries,
            result.false_misses as f64 / queries,
            percentile(&result.lookup_us, 50.0),
            percentile(&result.total_us, 99.0) / 1_000.0,
        );
    }
    println!("Replayed {count} prompts.");
    Ok(())
}
