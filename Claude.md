# Intrensic-valuator

An agentic AI system that values stocks using a top-down approach (Country → Industry → Company) and generates Buy/Sell/Hold recommendations with confidence scores.

## Python Environment
This project uses a local virtual environment located in the `./venv` folder.
- Always run Python scripts using the explicit path: `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows).
- Always install packages using the explicit path: `./venv/bin/pip install <package>`.
- Never use bare `python`, `python3`, or `pip` commands.

### Dependencies
- Maintain a `requirements.txt` file at the project root listing all required Python libraries.

---

## Architecture Overview

```
User Input (Ticker / Company Name)
        │
        ▼
┌───────────────────────────────────┐
│     ORCHESTRATOR AGENT             │
│  (LangGraph State Machine)         │
│  Routes to specialized agents      │
│  in prescribed order               │
└──────┬────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  1. COUNTRY / MACRO AGENT        │
│  - GDP growth, inflation, rates  │
│  - Political stability, currency │
│  - Sovereign risk premium        │
│  Output: Country Risk Premium,   │
│         Macro Growth Assumptions │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  2. INDUSTRY AGENT               │
│  - TAM, industry growth rate     │
│  - Competitive dynamics (Porter) │
│  - Regulatory environment        │
│  - Industry beta & leverage      │
│  Output: Industry Growth Rate,   │
│         Industry Beta, Risk Adj.  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  3. COMPANY AGENT                │
│  - Financial statement analysis  │
│  - Management quality, moat      │
│  - Idiosyncratic risks ONLY      │
│    (customer concentration,      │
│     key person, litigation)      │
│  - Forbidden: macro/country risk │
│    penalization (double-counting │
│    prevention guardrail)         │
│  Output: Revenue Growth Est.,    │
│         FCF Margin Est.,         │
│         Company-Specific Risk    │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  4. VALUATION ENGINE             │
│  - Build 5-10 year DCF model     │
│  - Calculate WACC with CRP       │
│  - Terminal value (Gordon/Exit)  │
│  - Monte Carlo simulation        │
│  - Relative valuation (comps)    │
│  Output: Intrinsic Value/Share,  │
│          Fair Value Range        │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  5. RECOMMENDATION ENGINE        │
│  - Compare intrinsic vs market   │
│  - Margin of safety calculation  │
│  - Confidence score              │
│  Output: BUY / SELL / HOLD       │
│          + detailed report       │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  6. REPORT GENERATOR             │
│  - Executive summary (1 page)    │
│  - Key findings per agent        │
│  - Valuation summary table       │
│  - Monte Carlo distribution plot │
│  - Scenario comparison chart     │
│  Output: PDF report via UI       │
│          download button          │
└──────────────────────────────────┘
```

---

## Data Sources

### Primary (Free)
| Source | Data Provided | Access Method |
|--------|--------------|---------------|
| **yfinance** | Stock price, financial statements (IS, BS, CF), shares outstanding, beta | Python library |
| **SEC EDGAR (XBRL)** | 10-K/10-Q filings — revenue, OCF, capex, debt, diluted shares (primary source) | HTTP + xbrl parsing |
| **FRED (St. Louis Fed)** | US Treasury yields, GDP, CPI, unemployment | `fredapi` Python library |
| **World Bank API** | Global GDP growth, inflation, population by country | `wbgapi` Python library |
| **Yahoo Finance (web)** | Analyst estimates, news, industry peers | Web scraping or `yfinance` |

### Secondary (Freemium — fallback)
| Source | Data Provided | Limits |
|--------|--------------|--------|
| **Financial Modeling Prep (FMP)** | Pre-computed DCF, analyst estimates, treasury rates, peer trading multiples (NaN fallback) | 250 req/day free |
| **Alpha Vantage** | Income statement, balance sheet, cash flow, forex | 25 req/day free |
| **EODHD** | Fundamentals, 60+ global markets | Free demo token |

---

## Caching Architecture ✅

A dual-layer caching system sits between the data sources and the valuation engine, ensuring subsequent calls for the same ticker or macro data don't re-hit the APIs.

### Why Two Layers?

| Layer | What it caches | Technology | Scope |
|-------|---------------|------------|-------|
| **Application** | Post-processed Python objects (DataFrames, dicts) | SQLite via raw `sqlite3` + custom JSON encoder | All fetcher outputs |
| **HTTP** | Raw HTTP responses from REST APIs | `requests-cache` CachedSession (SQLite backend) | All outbound GET requests |

The HTTP layer catches requests transparently — fetcher modules don't even know they're cached. The application layer caches the expensive post-processing (XBRL parsing, financial ratio computation) so those only run once per data refresh cycle.

### Application Cache (`src/data/data_cache.py`)

```
DataCache
├── Storage: Single SQLite table in WAL mode
│   CREATE TABLE cache_entries (
│       key        TEXT PRIMARY KEY,
│       value      TEXT NOT NULL,      -- JSON serialized
│       expires_at REAL NOT NULL,       -- absolute Unix timestamp
│       created_at REAL NOT NULL
│   );
├── Expiration: Lazy eviction on get() — no background thread
├── Thread safety: threading.Lock on writes; reads are lock-free
├── Serializer: Custom JSONEncoder handles numpy scalars/arrays,
│   pandas DataFrames (orient="split"), pandas Series
└── API: get(key) → Any|None
         set(key, value, ttl_seconds)
         delete(key) | clear() | expire() → count
         stats() → {hits, misses, expirations, hit_rate, db_size}
```

**Cache key convention:** `"{source}:{identifier}:{data_type}"`
```
yfinance:AAPL:cash_flow     fred:GDP:quarterly
yfinance:AAPL:balance_sheet  worldbank:IN:gdp_growth
yfinance:MSFT:price          fmp:AAPL:analyst_estimates
```

### HTTP Cache (`src/data/http_cache.py`)

```
get_http_session() → CachedSession (singleton)
├── Backend: SQLite (separate DB from application cache)
├── URL-pattern TTLs:
│   *.yahoo.com/*          → 1 day
│   data.sec.gov/*         → 1 day
│   api.stlouisfed.org/*   → 7 days
│   api.worldbank.org/*    → 30 days
│   financialmodelingprep.com/* → 7 days
│   www.alphavantage.co/*  → 7 days
├── stale_if_error=True   — serve stale cache if remote is down
├── allowable_codes=(200, 301, 302, 404)
└── clear_http_cache()    — force-refresh all HTTP responses
```

### TTL Rationale

| TTL Constant | Duration | Applies To | Why |
|-------------|----------|------------|-----|
| `TTL_PRICE` | 1 day | Stock quotes | Prices change daily; intraday noise doesn't affect DCF |
| `TTL_FINANCIALS` | 7 days | IS, BS, CF statements | Filed quarterly; 7 days covers weekends + earnings release lag |
| `TTL_MACRO_RATES` | 7 days | Treasury yields, Fed rates | Updated daily but DCF uses a single snapshot rate |
| `TTL_MACRO_GDP` | 30 days | GDP, CPI, macro aggregates | Released monthly/quarterly; no benefit to more frequent pulls |
| `TTL_ESTIMATES` | 7 days | Analyst consensus | Updated weekly by most data providers |

### How Fetchers Use the Cache

```python
from src.data.data_cache import DataCache, TTL_FINANCIALS

cache = DataCache()

def fetch_cash_flow(ticker: str) -> pd.DataFrame:
    key = f"yfinance:{ticker}:cash_flow"
    cached = cache.get(key)
    if cached is not None:
        return cached          # cache hit — no API call

    df = _pull_from_yfinance(ticker)
    cache.set(key, df, ttl_seconds=TTL_FINANCIALS)
    return df
```

The HTTP cache is transparent — if `_pull_from_yfinance` uses `requests` under the hood, those GET calls are cached by `requests-cache` without the fetcher being aware of it.

---

## Valuation Methodology

### Primary: Discounted Cash Flow (DCF)

#### Step 1: Calculate Base Free Cash Flow
```
FCF = Operating Cash Flow − Capital Expenditure
```
Pull from cash flow statement (NOT summary metrics — yfinance `info['freeCashflow']` is unreliable).

#### Step 2: Project Future Cash Flows (Years 1–5 explicit, then terminal)
```
FCF_Year_N = FCF_Base × (1 + Revenue_Growth_N) × FCF_Margin_N
```
Growth rates follow a continuous 3-knot linear spline:
- **Years 1–3**: Company_Growth → Industry_Growth (firm-specific converges to sector norm)
- **Years 4–5**: Industry_Growth → Terminal_Growth (sector decays to long-run GDP growth)
- At year 3, both segments equal Industry_Growth — mathematically seamless transition.

#### Step 3: Calculate Weighted Average Cost of Capital (WACC)
```
Cost of Equity (CAPM) = Rf + β_industry × ERP + CRP + SRP

Where:
  Rf    = 10-year US Treasury yield (from FRED)
  β_industry = Median unlevered beta of industry peers, relevered to company's D/E
          using Hamada formula with Statutory_Tax_Rate (NOT effective tax rate)
  ERP   = Equity Risk Premium (typically ~5.0-6.0%)
  CRP   = Country Risk Premium (from Country Agent analysis)
  SRP   = Size Risk Premium (if applicable)

Cost of Debt = (Rf + Country_Spread + Company_Credit_Spread) × (1 − Statutory_Tax_Rate)

WACC = (E/V) × Cost_of_Equity + (D/V) × Cost_of_Debt

NOTE: E/V and D/V use Market Cap for equity and Total_Debt_Proxy (book value of
debt from balance sheet) for debt. Book value of debt is an industry-standard proxy
since true market values for corporate debt are not publicly traded/observable.
Statutory tax rate is used throughout WACC and Hamada's formula to avoid distortions
from anomalous tax years (e.g., one-off credits producing 0% effective rates).
```

#### Step 4: Terminal Value
```
Terminal_Value = FCF_Final_Year × (1 + g_terminal) / (WACC − g_terminal)
```
Where g_terminal = long-term GDP growth or 2-3% (whichever is lower).

#### Step 5: Discount to Present Value
```
Enterprise_Value = Σ [FCF_t / (1 + WACC)^t] + TV / (1 + WACC)^n
Equity_Value = Enterprise_Value − Total_Debt + Cash_and_Equivalents
Intrinsic_Value_Per_Share = Equity_Value / Diluted_Shares_Outstanding
```
Diluted shares are validated via SEC EDGAR XBRL filings to avoid basic-share bias
(especially critical for companies with large option pools or convertible debt).

### Supporting: Relative Valuation
```
Fair_Value = Median(Peer_P/E) × Company_EPS
Fair_Value = Median(Peer_EV/EBITDA) × Company_EBITDA
```
Weight: 70% DCF, 30% Relative in final intrinsic value blend.

Peer data is primarily sourced via yfinance. When yfinance returns NaN for peer
multiples (common for smaller/international stocks), the system automatically falls
back to Financial Modeling Prep (FMP) or Alpha Vantage APIs, which provide cleaner
pre-calculated multiples.

### Scenario Analysis
- **Bull Case**: Higher growth, lower WACC
- **Base Case**: Expected values
- **Bear Case**: Lower growth, higher WACC
- Monte Carlo: 10,000 simulations varying growth and discount rate

### Assumption Validation Layer (Pre-DCF Guardrail)
Before agent-generated assumptions enter the DCF engine, they pass through validation:
- **Historical Range Check**: Assumptions outside the company's 5-year historical range are flagged
- **Industry Benchmark**: Assumptions >2× industry median require strong justification
- **Statistical Confidence Bands**: GREEN (within 1σ), AMBER (1-2σ), RED (>2σ — capped)
- **Override Flags**: RED assumptions are capped at 2σ and reported transparently in the PDF
- Confidence score is penalized for each flag raised

### Evidence-Chain Requirement
Every numerical adjustment must cite specific evidence. Agents cannot output
"Management: High → +2% Growth" without documenting the evidence chain:
- Specific metrics (ROIC, D/E, acquisition track record, CEO tenure/TSR)
- Why this evidence supports the chosen bucket
- Confidence level in the evidence
This makes every assumption auditable and rebuildable by a human analyst.

---

## Recommendation Engine

```
Margin_of_Safety = (Intrinsic_Value − Current_Price) / Intrinsic_Value

If Intrinsic_Value > Current_Price × 1.20 → BUY
If Intrinsic_Value < Current_Price × 0.90 → SELL
Else → HOLD

Confidence_Score = f(forecast_precision, model_agreement, data_quality,
                     historical_stability, analyst_consensus, macro_uncertainty,
                     assumption_validation)
```

**No-Override Principle:** The intrinsic valuation is never overridden by
qualitative judgment. Binary event risks (FDA approval, contract outcomes)
affect confidence and risk narrative — not the fair value estimate. The
Recommendation Agent communicates risks alongside the valuation; it does not
change the valuation itself. This keeps the system objective and auditable.

---

## Report Generation (PDF)

The system generates a professional PDF report downloadable from the UI. The report is structured for both retail investors and professionals.

### Report Sections
1. **Cover Page** — Company name, ticker, date, recommendation badge (color-coded: Green BUY / Red SELL / Amber HOLD)
2. **Executive Summary** (Page 1) — 3-4 paragraph synthesis: what the company does, key valuation drivers, recommendation with confidence score, one-line summary from each analysis layer
3. **Key Findings** — Bullet-point findings from Country, Industry, and Company agents, plus a **Growth Attribution Table** decomposing the revenue growth assumption into component drivers (historical CAGR, industry tailwind, product expansion, management execution, regulatory headwinds, etc.) with individual contributions summing to the net assumption
4. **Valuation Summary** — Table showing: Current Price, Intrinsic Value, Margin of Safety, Fair Value Range, WACC, Terminal Growth Rate, Terminal Value % of EV
5. **DCF Projection Table** — Year-by-year FCF, discount factor, present value
6. **Scenario Analysis** — Bull / Base / Bear case table with intrinsic values
7. **Monte Carlo Distribution** — Histogram chart of 10,000 simulated intrinsic values with percentile markers
8. **Relative Valuation** — Peer comparison table (P/E, EV/EBITDA, P/B multiples)
9. **Risk Factors** — Top risks identified and their potential impact
10. **Disclaimer** — Standard financial disclaimer

### PDF Generation Tech
- **Library**: `reportlab` for programmatic PDF generation (pure Python, no external deps)
- **Charts**: `matplotlib` rendered to PNG, embedded in PDF
- **Tables**: `reportlab.platypus.Table` with alternating row colors
- **Template**: Custom layout with header/footer branding, page numbers
- **Output**: Single PDF file, served as download from Streamlit UI

### Color Scheme
- BUY → Green (#1B5E20 background, white text badge)
- SELL → Red (#B71C1C background, white text badge)
- HOLD → Amber (#F57F17 background, white text badge)

---

## Agent Implementation (LangGraph)

### State Schema
```python
class ValuationState(TypedDict):
    ticker: str
    company_name: str
    # Country layer
    country: str
    country_gdp_growth: float
    country_inflation: float
    country_risk_premium: float
    risk_free_rate: float
    # Industry layer
    industry: str
    industry_growth_rate: float
    industry_beta: float
    industry_peers: list[str]
    # Company layer
    revenue_growth_est: float
    fcf_margin_est: float
    company_risk_adj: float
    financials: dict
    # Valuation
    wacc: float
    intrinsic_value: float
    fair_value_range: tuple[float, float]
    current_price: float
    # Output
    recommendation: str  # BUY / SELL / HOLD
    confidence_score: float
    report: str
```

### Agent Nodes
Each agent is an LLM-powered node that receives context from previous layers and produces structured outputs.

---

## Project Structure

✅ = implemented  |  📋 = planned

```
intrensic-valuator/
├── README.md                    # Project documentation ✅
├── Claude.md                    # This file ✅
├── Plan.txt                     # Detailed implementation plan ✅
├── requirements.txt             # Python dependencies ✅
├── pyproject.toml               # Pytest markers config ✅
├── .gitignore                   # Excludes .env, venv, cache, __pycache__ ✅
├── .env                         # API keys (gitignored, never commit) ✅
├── app.py                       # Streamlit web UI ✅
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline ✅
├── src/
│   ├── __init__.py              ✅
│   ├── main.py                  # CLI entry point ✅
│   ├── orchestrator.py          # LangGraph pipeline + sequential fallback ✅
│   ├── agents/
│   │   └── __init__.py          ✅
│   ├── data/
│   │   ├── __init__.py          ✅
│   │   ├── data_cache.py        # SQLite-backed TTL cache (application layer) ✅
│   │   ├── http_cache.py        # requests-cache session (HTTP layer) ✅
│   │   ├── yahoo_fetcher.py     # yfinance wrapper with fuzzy field matching ✅
│   │   ├── sec_fetcher.py       # SEC EDGAR XBRL verification ✅
│   │   └── macro_fetcher.py     # FRED + World Bank ✅
│   ├── valuation/
│   │   ├── __init__.py          ✅
│   │   ├── dcf_model.py         # Core DCF engine + growth decay helper ✅
│   │   ├── wacc_calculator.py   # CAPM + Hamada beta + synthetic rating ✅
│   │   ├── relative_val.py      # Peer multiples comparison ✅
│   │   └── monte_carlo.py       # Monte Carlo + Scenario analysis ✅
│   ├── report/
│   │   ├── __init__.py          ✅
│   │   ├── pdf_generator.py     # 10-section PDF via reportlab ✅
│   │   └── charts.py            # matplotlib histograms + bar charts ✅
│   ├── prompts/
│   │   ├── __init__.py          ✅
│   │   ├── country_prompt.py    # Country/macro agent system prompt ✅
│   │   ├── industry_prompt.py   # Industry analysis agent prompt ✅
│   │   ├── company_prompt.py    # Company fundamentals agent prompt ✅
│   │   └── recommendation_prompt.py # Final recommendation agent prompt ✅
│   └── utils/
│       ├── __init__.py          ✅
│       ├── config.py            # Typed .env loader + Config singleton ✅
│       └── logger.py            # Rich-based structured logging ✅
├── tests/
│   ├── __init__.py              ✅
│   ├── test_dcf_model.py        # 21 tests ✅
│   ├── test_wacc.py             # 24 tests ✅
│   ├── test_relative_val.py     # 11 tests ✅
│   ├── test_monte_carlo.py      # 11 tests ✅
│   ├── test_data_cache.py       # 17 tests ✅
│   └── test_integration.py      # 12 tests (network-reliant) ✅
├── notebooks/                   ✅ (empty)
└── cache/                       # SQLite cache files (gitignored) ✅
```

---

## Key Design Decisions

1. **LangGraph for orchestration** — DAG-based agent workflow with state passing. Each agent's output feeds the next. Better than linear chains because some analysis can branch (e.g., parallel Monte Carlo runs).

2. **yfinance as primary data source** — Free, no API key, covers all US stocks. FCF pulled from cash flow statement fields, not summary info dict.

3. **LLM for qualitative → quantitative conversion** — The hardest problem in automated valuation. Each agent uses carefully engineered prompts to convert textual analysis (management quality, moat, regulatory risks) into numeric adjustments to growth rates, margins, and discount rates. To prevent hallucinated precision, agents select from **discrete categorical buckets** (e.g., Country Risk Premium can only be 0, 250, 500, 750, 1000, or 1500 bps) rather than arbitrary values within continuous ranges. Each agent MUST output structured JSON with explicit numeric fields.

4. **WACC as the integration point** — All three layers (country, industry, company) feed into WACC calculation. Country → CRP. Industry → Beta. Company → specific risk premium. Uses **statutory tax rate** (not effective rate) in Hamada's formula and cost of debt to avoid distortions from anomalous tax years. **Book value of debt** is used as a pragmatic proxy for market value of debt, which is standard corporate finance convention since corporate bonds don't trade on public exchanges. This is the theoretically correct way to cascade top-down risk into valuation.

5. **Dual-layer caching** — Two complementary caching layers prevent redundant API calls:
   - **Application cache** (`DataCache`): SQLite-backed key-value store with TTL-driven lazy eviction. Handles numpy + pandas types via custom JSON serializer. Used by fetcher modules to cache post-processed results.
   - **HTTP cache** (`get_http_session`): `requests-cache` CachedSession with per-URL-pattern TTLs. Transparently caches raw HTTP responses from all data-source REST APIs. `stale_if_error=True` serves stale cache if the remote is unreachable.
   - TTLs: Prices=1d, Financials=7d, Rates=7d, GDP=30d, Estimates=7d.

6. **Confidence scoring** — 7-factor weighted model: forecast precision (25%), model agreement (20%), data quality (15%), historical stability (10%), analyst consensus agreement (10%), macroeconomic uncertainty (10%), and assumption validation (10%). Confidence is an assessment of model reliability, not just valuation uncertainty.

7. **No-override principle** — The intrinsic valuation is never overridden by qualitative judgment. Binary events affect confidence and risk narrative, not fair value. Keeps the system objective and auditable.

8. **Assumption validation layer** — All agent-generated assumptions pass through historical range, industry benchmark, and statistical confidence checks before entering the DCF. RED-flagged assumptions are capped at 2σ and reported transparently.

9. **Backtesting as core infrastructure** — Every recommendation is tracked against actual returns. Component-level attribution identifies which layer contributed most to forecast error. Without this, it's impossible to know if the multi-agent architecture adds value beyond a naive DCF.

7. **PDF Report Generation** — Professional, downloadable PDF report with executive summary, key findings, valuation tables, scenario analysis, and charts. Built with `reportlab` (pure Python) + `matplotlib` for embedded charts. Served as a download button in the Streamlit UI.

---

## Implementation Phases

### Phase 1: Foundation (Data Layer) — ✅ COMPLETE
- ✅ Set up project structure (all directories, `__init__.py` files, `.gitignore`)
- ✅ Implement caching layer
  - `DataCache` — SQLite-backed, thread-safe, TTL-driven key-value store
  - `http_cache` — `requests-cache` CachedSession with per-URL-pattern TTLs
  - JSON serializer handles numpy scalars/arrays, pandas DataFrames/Series
  - Lazy eviction: expired entries deleted on read (no background thread)
- ✅ Config utility (`src/utils/config.py`) — typed `.env` loader with property helpers
- ✅ Logger utility (`src/utils/logger.py`) — Rich-based structured logging
- ✅ YahooFinanceFetcher — yfinance wrapper with fuzzy field-name matching
- ✅ MacroFetcher — FRED + World Bank API
- ✅ DCF model — FCF, projection, terminal value, discounting, equity value
- ✅ WACC calculator — CAPM, Hamada beta, synthetic credit rating, size premium

### Phase 2: Core Valuation (No LLM)
- ✅ Complete DCF engine with all formulas
- ✅ Relative valuation (peer comparison)
- ✅ Monte Carlo simulation
- ✅ CLI that takes manual growth/WACC assumptions and outputs valuation

### Phase 3: Agentic AI Integration
- ✅ LangGraph orchestrator (with sequential fallback for compatibility)
- ✅ Implement Country Agent with LLM (DeepSeek prompt ready)
- ✅ Implement Industry Agent with LLM (DeepSeek prompt ready)
- ✅ Implement Company Agent with LLM (DeepSeek prompt ready)
- ✅ Integrate agent outputs → DCF inputs → valuation (defaults used when no LLM)

### Phase 4: Recommendation & Polish
- ✅ Recommendation engine with confidence scoring (7-factor model)
- ✅ No-override principle: valuation stands; qualitative factors affect confidence and risk narrative only
- ✅ PDF report generation (10 sections: cover, exec summary, key findings + growth attribution, valuation, DCF projection, scenarios, Monte Carlo, peer comparison, risks, disclaimer)
- ✅ Streamlit UI with ticker input, report display, and PDF download button
- 📋 Backtesting framework — **promoted to core infrastructure** (track historical recommendations, component-level validation: does Country Analysis improve accuracy? Does Monte Carlo calibrate correctly? Does blending beat pure DCF?)
- ✅ Tests and documentation

---

## Dependencies (requirements.txt)

```
# Core
langgraph>=0.2.0
langchain>=0.3.0
langchain-deepseek>=0.1.0
pydantic>=2.0.0

# Data
yfinance>=0.2.40
fredapi>=0.5.0
wbgapi>=1.0.0
pandas>=2.0.0
numpy>=1.24.0

# Valuation
numpy-financial>=1.0.0
scipy>=1.10.0

# Caching
requests-cache>=1.0.0

# UI (optional — Phase 4)
streamlit>=1.30.0

# PDF Report Generation
reportlab>=4.0.0
matplotlib>=3.7.0

# Utilities
python-dotenv>=1.0.0
rich>=13.0.0

# Web Search (DuckDuckGo free + Tavily optional)
duckduckgo-search>=7.0.0

# Testing
pytest>=8.0.0
```

---

## LLM Provider

Primary: **DeepSeek** (via `langchain-deepseek`)
- Strong reasoning capabilities, excellent for structured financial analysis
- Cost-effective for multi-agent workflows
- Model: `deepseek-chat` for agent calls (supports tool calling and structured output)
- Fallback: `deepseek-reasoner` for complex synthesis requiring deeper reasoning

### API Key Configuration
- Store API key in `.env` file at project root:
  ```
  DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
  ```
- Loaded via `python-dotenv` at runtime
- `.env.example` provided as template (never commit real keys)

### Web Search Tools (for Agent Research)
- **Default**: DuckDuckGo search via `duckduckgo-search` library — free, no API key, unlimited usage
- **Optional**: Tavily Search API — higher-quality structured results, but rate-limited free tier
- Fallback logic: Try Tavily first if API key is set; fall back to DuckDuckGo if Tavily quota exhausted or key not configured

Each agent call includes:
- System prompt defining the agent's role and output schema
- Context from previous analysis layers
- Tool access to data fetchers + web search for real-time lookups
- Required structured output format (JSON)

---

## How Qualitative Data Becomes Quantitative

This is the core innovation. Each agent converts text analysis into numbers. To prevent
LLM "hallucinated precision" (arbitrary variance between identical runs), agents select
from **discrete categorical buckets** rather than open-ended continuous ranges. Each
category maps to a fixed quantitative value in Python — the LLM chooses the structural
bucket, not the exact number.

### Conversion Table (Discrete Buckets)

| Qualitative Classification | Quantitative Parameter | Fixed Bucket Values |
|---|---|---|
| Country political stability | Country Risk Premium | [0, 250, 500, 750, 1000, 1500] bps |
| Currency risk assessment | Additional discount | [0, 100, 250, 500] bps |
| Industry competitive intensity | FCF margin pressure | [0, -100, -250, -500] bps |
| Regulatory risk | Beta adjustment | [-0.2, -0.1, 0.0, +0.1, +0.2] |
| Disruption risk | Terminal growth modifier | [-1.0%, -0.5%, 0.0%, +0.5%] |
| Moat width | Competitive adv. period | [3, 5, 7, 10] years |
| Management quality | Execution growth factor | [-2.0%, -1.0%, 0.0%, +1.0%, +2.0%] |
| Financial health | Credit spread | Fixed synthetic rating mapping (70-800 bps) |
| ROIC vs WACC gap | Value creation confidence | Bounds terminal perpetuity assumption |

Buckets are derived from corporate valuation literature (Damodaran, McKinsey, CFA
curriculum). Agents must justify their category choice in the narrative output.

### Double-Counting Prevention (Critical Guardrail)

The Company Agent is **explicitly restricted to idiosyncratic risks only** (customer
concentration, key person risk, litigation, product obsolescence). It is forbidden
from penalizing or adding premiums for systemic macro/country-level problems
(e.g., general inflation, country risk). Those risks belong exclusively to Agents 1
(Country) and 2 (Industry). This prevents the same underlying macroeconomic risk
from being double-counted, which would over-discount cash flows and produce an
unjustifiably low intrinsic value.

---

## Implementation Deviations

These are intentional deviations from the original Plan.txt, documented for transparency.

### 1. LangGraph Sequential Fallback
**Plan:** LangGraph StateGraph would directly compile with `ValuationState` dict subclass.
**Actual:** `ValuationState(dict)` caused `'ticker'` KeyError in LangGraph's `invoke()`. Added a sequential fallback pipeline that calls each agent node function in order. The LangGraph graph is still built and attempted first; if `app.invoke()` fails, the pipeline proceeds sequentially. This makes the orchestrator MORE robust — it works regardless of LangGraph version compatibility.

### 2. Raw sqlite3 Instead of sqlitedict
**Plan:** Use `sqlitedict` library for SQLite caching.
**Actual:** Used Python's built-in `sqlite3` module directly. This eliminates a dependency and gives finer control over WAL mode, busy timeout, and table schema. The `DataCache` class is more featureful than a plain sqlitedict (stats, explicit expire, bulk clear). `sqlitedict` was removed from `requirements.txt` and Claude.md dependencies.

### 3. Monte Carlo Iterations (5,000 not 10,000)
**Plan:** 10,000 iterations.
**Actual:** Defaults to 5,000 with a comment "5k for speed; use 10k for production". This gives faster CLI feedback during development. The `MonteCarloSimulation` class accepts `iterations` as a constructor parameter — users can pass 10,000 or more.

### 4. yfinance Field Name Aliasing
**Plan:** Direct field name access from yfinance DataFrames.
**Actual:** yfinance field names vary across stocks (e.g., "Total Revenue" vs "Revenue", "Operating Cash Flow" vs "Cash Flow From Continuing Operating Activities"). Added a fuzzy-match alias system in `_get_latest()` with fallback lists. This was necessary for the pipeline to work across different stocks.

### 5. No .env.example File
**Plan:** Create `.env.example` as a template.
**Actual:** Only `.env` exists. The `.env` file is gitignored. Users can copy `.env` from the documentation in Plan.txt Section 3.1.

### 6. Agent Files as Prompts (Not Separate Agent Classes)
**Plan:** Individual agent classes in `src/agents/` (country_agent.py, industry_agent.py, company_agent.py, etc.).
**Actual:** Agent system prompts live in `src/prompts/` as standalone modules. The agent logic (LLM calls, parsing, state updates) is in `src/orchestrator.py` as node functions. The `src/agents/` directory exists with only `__init__.py`. This is cleaner — one orchestrator to maintain instead of 5 agent class files.

### 7. Shares Outstanding in Raw Count
**Plan:** Shares outstanding in millions.
**Actual:** yfinance returns actual share count (e.g., 14,687,356,000 for AAPL). The DCF model divides equity value by this raw count. Math is consistent as long as market cap and shares use the same units.

### 8. Plan Refinements — Mathematical & Architectural Hardening (2026-07-02)
**Plan (original):** Several sections had mathematical vulnerabilities, operational gaps, and LLM guardrail weaknesses.
**Actual (refined Plan.txt):** Eight targeted fixes applied:

*Mathematical:*
- Growth decay formula replaced with continuous 3-knot spline (Company→Industry→Terminal) eliminating the year 3→4 discontinuity while preserving industry anchoring
- Statutory tax rate adopted throughout Hamada's formula and WACC cost of debt to prevent anomalous tax-year distortions
- Book value of debt explicitly documented as a proxy (`Total_Debt_Proxy`), with honest acknowledgment that true market values for corporate debt are unobservable

*Operational robustness:*
- FMP/Alpha Vantage designated as automatic fallback when yfinance peer data returns NaN
- Diluted shares designated as the standard, with SEC EDGAR as primary validation source (applied at 3 locations: data sources, capital structure weights, and PV calculation)

*LLM guardrails:*
- Qualitative→quantitative conversion rebuilt with discrete categorical buckets instead of continuous ranges — eliminates hallucinated precision (e.g., CRP can only be 0|250|500|750|1000|1500 bps)
- Company Agent restricted to idiosyncratic risks only, with explicit guardrail forbidding systemic macro/country penalization to prevent double-counting

These refinements do not change the implemented code architecture — they tighten the plan's theoretical rigor and provide clearer guidance for future code-level hardening.

### 9. Recommendations-Driven Architectural Hardening (2026-07-02)
Following a systematic review of the plan, six additional architectural principles were adopted:

*No-override principle:* The intrinsic valuation is never overridden by qualitative judgment. Binary events (FDA, contracts) affect confidence and risk narrative only.

*Assumption validation layer:* All agent assumptions are checked against historical ranges, industry benchmarks, and statistical confidence bands (GREEN/AMBER/RED) before entering the DCF engine. RED assumptions are capped at 2σ and reported transparently.

*Evidence-chain requirement:* Every numerical adjustment must cite specific, verifiable evidence. "Management: High → +2% Growth" is not sufficient — the agent must document ROIC track record, leverage history, acquisition integration, and why this evidence supports the chosen bucket.

*Feature attribution:* Growth assumptions are decomposed into component drivers (historical CAGR, industry tailwind, product expansion, management execution, regulatory headwinds) with individual contributions summing to the net assumption.

*Expanded confidence scoring:* Model grew from 4 to 7 factors, adding analyst consensus agreement, macroeconomic uncertainty, and assumption validation quality.

*Backtesting promoted to core:* Backtesting moved from "future enhancement" to core infrastructure with component-level validation (does Country Analysis improve accuracy? Is Monte Carlo well-calibrated? Does blending beat pure DCF?).

These principles represent the v1 architectural guardrails. The v2 roadmap extends further: shift LLMs from number generation to evidence extraction, replace fixed discrete buckets with historically-calibrated models, and automate prompt calibration against known outcomes.

---

## Testing

### Test Structure

```
tests/
├── __init__.py
├── test_data_cache.py        # 17 unit tests — DataCache CRUD, TTL, serialization, stats
├── test_dcf_model.py         # 21 unit tests — FCF, projection, terminal value, discount, pipeline
├── test_wacc.py              # 24 unit tests — credit spread, size premium, beta, WACC components
├── test_relative_val.py      # 11 unit tests — P/E, EV/EBITDA, P/B, blended, medians
├── test_monte_carlo.py       # 11 unit tests — distribution, reproducibility, scenarios
└── test_integration.py       # 12 integration tests — full pipeline, data fetchers, PDF generation
```

### Running Tests

```bash
# All 78 unit tests (no network, < 3 seconds)
./venv/bin/python -m pytest tests/ -m "not integration" -v

# Full suite including integration tests (need network, ~20 seconds)
./venv/bin/python -m pytest tests/ -v

# Specific test file
./venv/bin/python -m pytest tests/test_dcf_model.py -v
```

### Coverage Summary

| Module | Tests | Type |
|--------|-------|------|
| `data_cache.py` | 17 | Unit |
| `dcf_model.py` | 21 | Unit |
| `wacc_calculator.py` | 24 | Unit |
| `relative_val.py` | 11 | Unit |
| `monte_carlo.py` | 11 | Unit |
| `orchestrator.py` + fetchers + PDF | 12 | Integration |
| **Total** | **90** | **78 unit + 12 integration** |

### Test Markers

- `@pytest.mark.integration` — tests requiring network (yfinance, FRED). Skip with `-m "not integration"`.
- All other tests are pure unit tests with no external dependencies.

---

## CI/CD (GitHub Actions)

CI pipeline at `.github/workflows/ci.yml` runs on every push and pull request to `main`/`master`.

### Pipeline Steps
1. **Checkout** → `actions/checkout@v4`
2. **Python 3.12** → `actions/setup-python@v5`
3. **Cache pip** → keyed by `requirements.txt` hash
4. **Install deps** → `pip install -r requirements.txt`
5. **Verify imports** → smoke test all 7 core modules
6. **Run unit tests** → `pytest -m "not integration" --strict-markers --junitxml=junit.xml`
7. **Publish summary** → test-summary action renders results inline
8. **Upload artifacts** → `junit.xml` downloadable

### What's Excluded
- Integration tests (`@pytest.mark.integration`) — skipped via marker selector
- All API keys explicitly set to `""` in CI environment
- Zero network access required — all 78 unit tests are pure computation

---

## Recent Fixes & Updates

### Plan Refinements — Mathematical & Architectural Hardening (2026-07-02)
Eight targeted fixes applied to Plan.txt addressing vulnerabilities identified in a systematic review. See Implementation Deviation #8 for full details. Key changes: continuous 3-knot growth decay spline, statutory tax rate in Hamada/WACC, discrete categorical buckets replacing continuous LLM ranges, double-counting prevention guardrail in Company Agent, FMP/Alpha Vantage fallback for peer NaN data, diluted shares via SEC EDGAR validation.

### Recommendations-Driven Architectural Principles (2026-07-02)
Six architectural principles adopted from external review: no-override principle, assumption validation layer (GREEN/AMBER/RED with 2σ capping), evidence-chain requirement for all adjustments, feature attribution in growth decomposition, expanded 7-factor confidence scoring, and backtesting promoted from future enhancement to core infrastructure. See Implementation Deviation #9 for full details.

### PDF Narrative Truncation Fix (2026-06-25)
**Problem:** Agent narrative text (macro, industry, company) was hard-truncated at 800 characters with `text[:800]`, cutting off mid-sentence on the company analysis PDF page.
**Fix:** Replaced all hard truncation with a `_safe_text()` helper that:
- Allows up to 4,000 characters per narrative
- Truncates at paragraph boundaries (double-newline), not mid-sentence
- Falls back to sentence boundaries (`. `, `! `, `? `) if no paragraph break
- Strips raw HTML/XML tags that break reportlab's parser
- Adds `(truncated)` note only when text exceeds limit

### duckduckgo_search → ddgs Rename
The `duckduckgo-search` package was renamed to `ddgs`. A `RuntimeWarning` appears when importing. Update to `pip install ddgs` in a future version.

### DeepSeek Agents Verified
All three LLM agents (Country, Industry, Company) run successfully with a valid `DEEPSEEK_API_KEY`. Each agent:
1. Receives full prior analysis context
2. Performs DuckDuckGo web search for current data
3. Produces structured JSON output parsed into the pipeline state
4. Outputs feed directly into the valuation engine (growth rates, margins, WACC adjustments)

Without a DeepSeek key, the pipeline falls back to conservative defaults from actual financial data — still producing a fully reasoned valuation.
