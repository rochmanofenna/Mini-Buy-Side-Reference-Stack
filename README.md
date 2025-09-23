# AI Trading Infrastructure (Reference Stack)

[![Performance CI](https://github.com/rochmanofenna/Mini-Buy-Side-Reference-Stack/actions/workflows/perf.yml/badge.svg)](https://github.com/rochmanofenna/Mini-Buy-Side-Reference-Stack/actions/workflows/perf.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducible Evidence](https://img.shields.io/badge/evidence-backtests%20%7C%20k6%20%7C%20prom-snapshot-blue)](#reproducible-evidence)

Production-style **AI trading infrastructure** for research/recruiting: FastAPI router, GPU/CPU backtesting, ingestion, monitoring, and **reproducible evidence**.
**Not for live trading.**

---

## TL;DR (What this proves)

- **Backtests (demo dataset):** **Sharpe 2.95**, **MaxDD −0.36%**, **Ann. return ~2%** (walk-forward).
  Evidence: `evidence/backtests/metrics_summary.json` + charts.

- **Router SLOs:** **p99 ≈ 18–20 ms @ ~14k rps** under k6 load.
  Evidence: `evidence/benchmarks/k6_http_*.json` + Grafana PNGs; Prometheus TSDB snapshot in `evidence/prometheus/`.

- **Signals ledger:** Published in `evidence/signals/` (demo + full). Recompute hit-rate/PnL from CSV.

> This is a **production-style reference** stack with **public artifacts**. Use it to audit methodology & results. Do **not** connect to real brokers.

---

## Quickstart

```bash
# python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-unified.txt
pip install -e .

# deterministic backtest
python -m src.strategy.backtest_runner \
  --config config/backtest_config.yaml \
  --seed 1337 \
  --output out/backtest_results.parquet

python scripts/evaluate_backtest.py \
  --input out/backtest_results.parquet \
  --outdir out/eval_demo
```

Expected artifacts:

* `out/eval_demo/metrics_summary.json` (≈ `evidence/backtests/metrics_summary.json`)
* `out/eval_demo/portfolio_equity.png`, `return_distribution.png`

---

## Run the stack + bench the router

```bash
# bring up infra (Prometheus, Grafana, router, etc.)
cp .env.example .env
docker compose -f infra/compose.yaml up -d --build

# health check
curl -sf http://localhost:8080/healthz && echo "router OK"

# k6 (machine-readable export)
k6 run --vus 200 --duration 60s \
  -e BASE_URL=http://localhost:8080 \
  --summary-export out/k6_http_ci.json \
  benchmarks/bench_http_k6.js

# Prometheus snapshot (requires admin API enabled in compose)
PROM=$(docker ps --filter "ancestor=prom/prometheus:latest" -q | head -n1)
docker exec "$PROM" wget -qO- "http://localhost:9090/api/v1/admin/tsdb/snapshot?skip_head=true"
SNAP=$(docker exec "$PROM" sh -lc "ls -1 /prometheus/snapshots | tail -n1" | tr -d '\r')
docker cp "$PROM":/prometheus/snapshots/$SNAP out/prom_snapshot
```

Optional PNG export (proves Grafana renderer & panel IDs):

```bash
GRAFANA_PASS=${GRAFANA_PASSWORD:-admin}
curl -s -u "admin:$GRAFANA_PASS" \
 "http://localhost:3000/render/d-solo/fusionalpha-performance/Performance?orgId=1&panelId=2&width=1600&height=900" \
 --output out/router_latency_p99.png
```

---

## Reproducible evidence

* Backtests: [`evidence/backtests/`](evidence/backtests/)
* Router perf: [`evidence/benchmarks/`](evidence/benchmarks/)
* Prometheus snapshot: [`evidence/prometheus/`](evidence/prometheus/)
* Signals ledger: [`evidence/signals/`](evidence/signals/)

> We pin seeds, versions, and configs. See `docs/METHODOLOGY.md` for walk-forward protocol, TC/slippage, risk caps, and leakage guards.

---

## Architecture (high level)

* **Router:** FastAPI + micro-batching + Redis caching
* **Backtesting:** GBM/Monte-Carlo engine (CPU/GPU), risk & position sizing, walk-forward
* **Ingestion:** NATS JetStream (at-least-once), S3 Parquet writers
* **Observability:** Prometheus (metrics), Grafana (dashboards, PNG exports)
* **Perf:** k6 scripts for HTTP + WebSocket

```
clients → FastAPI Router → Strategy/Models
   │            │
   │            ├─ Prometheus (metrics) → Grafana (dashboards)
   └─ k6 load ──┘
data → NATS → S3 Parquet → Backtests → Evidence/
```

---

## Methodology & Data

* Full protocol: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
* Data sources & license: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
* **Disclaimer:** This repo is for research/recruiting. Not investment advice.

---

## Dev notes

* CI: Performance smoke test builds the stack, runs k6 with `--summary-export`, uploads JSON and Prom snapshot artifacts.
* Strict guard: CI fails on any stray code placeholders.
* License: MIT (see [LICENSE](LICENSE)).

---