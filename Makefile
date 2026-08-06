# Build targets for veritas-cache.
# The default target prints the available commands.
.PHONY: bench test smoke help

help:
	@echo "make bench   Run the benchmark harness and build the charts."
	@echo "make test    Run the Rust tests and the trace regression tests."
	@echo "make smoke   Run the proxy end-to-end smoke test."

bench:
	cargo run --release --bin bench
	python3 scripts/make_charts.py

test:
	cargo test
	python3 scripts/test_trace.py

smoke:
	scripts/smoke_proxy.sh
