# 🏛️ Sovereign AI Trading Engine (v3.5 - Institutional Grade)

An institutional-grade quantitative screening and scoring ecosystem designed for the Indian (NSE) and Global (US) markets. This system bridges the gap between raw unstructured market data and high-conviction investment signals through a service-oriented validation pipeline.

---

## 📦 Two Editions

**Sovereign Lite** (default — this is what runs in the preview and is recommended for a single user on SQLite + yFinance):

- SQLite only (5 core tables + score history + quarterly results + MB candidate tracking + 5y financial history)
- yFinance-only data layer (`lite/data.py`) — no API keys, no Redis/Celery
- 5-screen React dashboard: Dashboard · Screener · Elite Picks · Portfolio · Backtest
- 5-factor regime-aware scoring (Quality/Growth/Momentum/Valuation/Risk) with **data-confidence dampening** (partial fundamentals can't rank high), multi-factor RS (1/3/6/12M percentiles), **institutional quality** (5y stability), earnings-revision proxy, sector rotation (RS + growth + breadth + momentum), conviction-weighted portfolio construction, 100-Bagger Detector with MB v3 (incl. 5y **compounder score**), one-click backtest + **walk-forward validation** (3×12-month folds)
- 5y price history stored, so backtests and walk-forward folds cover real multi-year windows
- Entry point: `python3 lite_main.py` · deps: `requirements-lite.txt`

**Sovereign Pro** (optional, legacy enterprise stack) — moved to `pro/` (`pro/modules/`, `pro/worker/`, `pro/monitoring/`, `pro/api/`). XGBoost/SHAP hybrid scoring, Redis + Celery distributed workers, Prometheus observability, Alpha Vantage multi-source ingestion. Loaded only if you explicitly run the Pro entry points and install its deps — it is never imported by the Lite app:

```bash
pip install -r pro/requirements.txt
python3 pro/main.py          # legacy FastAPI engine (serves pro/web-ui)
python3 pro/sovereign-cli.py # ops CLI
```

---

## 💎 Core Investment Philosophy

The Sovereign Engine is built on the **"Quality at a Reasonable Price" (QARP)** principle, enhanced by **Momentum Alpha**. It filters out 99% of the market noise to find stocks exhibiting high return on capital, robust cash flows, and accelerating earnings momentum.

## 🧠 Hardened Scoring Methodology

The heart of the system is the `calculate_institutional_score` function in `modules/scoring.py`. This is a dynamic, regime-aware weighting engine that adapts to market conditions.

### 1. Factor Normalization & Splines
- **Sigmoid Normalization**: Every raw metric is passed through a `sigmoid-based normalization` (0-100) to prevent step-cliff biases.
- **Smooth Graduation Splines**: Replaced binary disqualifiers with continuous splines. A stock with 15.1% ROE no longer scores 20 points higher than one with 14.9%.
- **Deterministic Tie-Breaking**: Implemented a symbol-hash based microscopic epsilon (5-decimal precision) to ensure stable rankings for stocks with identical fundamentals.

### 2. The 8-Factor Model
| Factor | Default Weight | Why it Matters |
| --- | --- | --- |
| **Sales Growth** | 0.15 | Verifies top-line demand expansion. |
| **ROE Stability** | 0.15 | Measures capital efficiency and moat strength. |
| **Cash conversion** | 0.10 | Detects accounting red flags (CFO/PAT). |
| **Valuation Gap** | 0.15 | Graham / PEG Margin of Safety. |
| **EPS Velocity** | 0.10 | Identifies profit inflection points. |
| **F-Score** | 0.10 | 9-Pt Piotroski business health. |
| **Leverage** | 0.10 | Debt/Equity penalties (Sect-weighted). |
| **Momentum** | 0.15 | Relative Strength and technical trend. |

---

## 📈 Market Regime Architecture

The Sovereign Engine automatically switches between **four market regimes**, redistributing weights to match the environment:
- **BULL (Momentum)**: Aggressive growth priority (w_mom=0.35, w_eps=0.40).
- **BEAR (Value)**: Extreme focus on Graham floor and cash conversion (w_val=0.30).
- **SIDEWAYS (Balanced)**: Balanced focus on ROE and consistent sales.
- **QUALITY**: Prioritizes F-Score and CFO efficiency above all else.

---

## 📡 Observability & Monitoring

The engine is now fully instrumented for production-grade observability:
- **Prometheus Metrics**: High-resolution business metrics exposed via `/metrics`.
- **Latency Tracking**: Every Celery task is instrumented with timers to monitor pipeline throughput and data ingestion lag.
- **Data Quality (DQ) Guard**: Real-time tracking of fundamental data coverage and LLM thesis fallbacks.

---

## 🏗️ System Architecture

The trading engine follows a decoupled, **Service-Oriented Design** for maximum scalability.

```mermaid
graph TD
    A[ticker_list.py] --> B[TaskQueueCoordinator]
    B --> C[IngestionService]
    B --> D[ScoringService]
    B --> M[MonitoringService]
    
    C -->|Multi-Source| E[PNSEA / yfinance / nsepython]
    C -->|Validate| F[Pydantic Models]
    
    D -->|8-Factor| G[Institutional Scoring]
    D -->|ML| H[MLOpsService / XGBoost]
    D -->|Swarm Validation| S[MiroFish Client Service]
    
    S -->|REST API| Z[(MiroFish Prediction Engine)]
    
    M -->|Metrics| P[(Prometheus Export)]
    
    F --> I[DataStoreService]
    I --> J[(stocks.db / PostgreSQL)]
    I --> K[(pit_store.db)]
    
    J --> L[FastAPI / main.py]
    L --> N[Web UI Dashboard]
```

### Swarm Intelligence Validation (MiroFish Integration)
Before taking a final position, high-conviction picks generated by the QARP/Momentum pipeline can be passed to our **Multi-Agent Simulation Layer**. Running `python scan_swarm.py --tickers ...` triggers a Swarm Intelligence debate via the MiroFish engine. Multiple AI agents parse contemporary news and fundamental data to battle-test the thesis, returning a finalized "Swarm Conviction Score."

---

## 🛡️ Security & Integrity

- **Environment Isolation**: API keys (Alpha Vantage) are never hardcoded. Missing keys trigger a hard-warning or disable downstream modules.
- **CORS Whitelisting**: The API serves only trusted origins defined in `.env`.
- **Database Abstraction**: Supports both **SQLite** (local dev) and **PostgreSQL** (production) via `DATABASE_URL` dynamic routing.
- **100% Test Coverage**: The core scoring engine passes a comprehensive unit test suite ([tests/test_scoring_engine.py](tests/test_scoring_engine.py)) with 35+ edge-case scenarios.

---

## 🛠️ Operational CLI (`sovereign-cli.py`)

- `health`: Run deep-forensic checks on env, deps, and connectivity.
- `ml-ops`: Monitor and update ML models (`--retrain`, `--update`).
- `tune-db`: One-click optimization for all SQLite databases.
- `db-stats`: Instant table audit and health overview.
- `regime`: Real-time diagnostic of market regime voting.

---

## ⚙️ Advanced Configuration (`config.py`)

| Key | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | env | Dynamic DB routing (PostgreSQL/SQLite). |
| `OLLAMA_URL` | env | Remote LLM gateway for thesis generation. |
| `CORS_ALLOWED_ORIGINS` | env | Comma-separated trusted web origins. |
| `MAX_SECTOR_EXPOSURE` | 0.25 | Prevents portfolio over-indexing. |
| `HARD_KILL_SWITCH_VIX` | 35.0 | Stops execution during extreme volatility. |

---

## 🚀 Getting Started

**Sovereign Lite (recommended — the default path):**

1. **Install**: `pip install -r requirements-lite.txt`
2. **Run**: `python3 lite_main.py` — dashboard at `:9005`, API at `/api/*`
3. **First scan**: hit **⟳ Run Scan** on the dashboard (fetches ~155 NSE names from yFinance, ~10s when cached)

**Sovereign Pro (legacy enterprise stack, optional):**

1. `pip install -r pro/requirements.txt`
2. `python3 pro/main.py` — legacy dashboard at `:9005`
3. Health check: `python3 pro/sovereign-cli.py health`

**Note for Contributors:** The Lite dashboard is a single file (`lite/web/index.html`, React via CDN). The legacy `pro/web-ui` frontend's `node_modules` is gitignored — run `cd pro/web-ui && npm install` if you touch it. Generated reports (`reports/`, `reports_cache/`) are gitignored.
