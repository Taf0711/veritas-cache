# Build the proxy in a full Rust toolchain image.
FROM rust:1-trixie AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Run the proxy in a slim image. The model files download from Hugging Face.
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /data/models \
    && curl -L --fail -o /data/models/model.onnx \
       https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx \
    && curl -L --fail -o /data/models/tokenizer.json \
       https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json
COPY --from=build /app/target/release/veritas-cache /usr/local/bin/veritas-cache

ENV HOST=0.0.0.0 \
    VERITAS_MODEL_DIR=/data/models \
    CACHE_DB_PATH=/data/cache.db \
    PORT=8080 \
    UPSTREAM_BASE_URL=https://api.openai.com \
    SEMANTIC_POLICY=ld3s
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["veritas-cache"]
