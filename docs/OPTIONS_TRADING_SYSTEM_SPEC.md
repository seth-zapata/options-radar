# Options Trading Recommendation System — v2 Specification

**Project Codename:** OptionsRadar  
**Version:** 2.0  
**Last Updated:** December 27, 2025  
**Purpose:** Implementation specification for Claude Code

---

## 1. System Overview

A local, display-only options trading recommendation system for a curated watchlist of tech/AI stocks. The system streams real-time options data, applies gating logic, and outputs trade recommendations with full transparency into data freshness, liquidity, and reasoning.

**Core Principle:** The system ABSTAINS by default. A recommendation only fires when ALL gates pass.

### 1.1 Scope Boundaries

| In Scope | Out of Scope |
|----------|--------------|
| Display recommendations | Trade execution |
| Portfolio visibility (read-only) | Portfolio modification |
| Simple puts/calls | Spreads, multi-leg strategies |
| Curated watchlist + daily discovery | Full market coverage |
| Local deployment (localhost) | Cloud/containerized deployment |

### 1.2 Budget Tiers

| Tier | Monthly Cost | Data Quality | Use Case |
|------|--------------|--------------|----------|
| **Prototype** | $30-80 | 15-min delayed, limited Greeks | Development, UI testing |
| **Production** | $300-400 | Real-time OPRA, full Greeks | Live recommendations |

**Prototype Stack:** Polygon Starter ($29) + Tradier (free w/ funded account) + Finnhub free tier  
**Production Stack:** ORATS Live ($199) + Alpaca Algo Trader Plus ($99) + Quiver ($20-75)

---

## 2. Canonical Option ID Specification

All internal logic uses a single canonical representation. Vendor-specific symbols are derived deterministically.

### 2.1 Internal Representation

```typescript
interface CanonicalOptionId {
  underlying: string;      // e.g., "NVDA"
  expiry: string;          // ISO 8601 date: "2025-01-17"
  right: "C" | "P";        // Call or Put
  strike: number;          // Float with 2 decimal precision: 500.00
  multiplier: number;      // Default 100, adjust for minis/weeklies
}

// Example
{
  underlying: "NVDA",
  expiry: "2025-01-17",
  right: "C",
  strike: 500.00,
  multiplier: 100
}
```

### 2.2 Vendor Symbol Mapping

```typescript
function toOCC(opt: CanonicalOptionId): string {
  // OCC format: NVDA  250117C00500000 (padded)
  const exp = opt.expiry.replace(/-/g, '').slice(2); // YYMMDD
  const strike = Math.round(opt.strike * 1000).toString().padStart(8, '0');
  return `${opt.underlying.padEnd(6)}${exp}${opt.right}${strike}`;
}

function toAlpaca(opt: CanonicalOptionId): string {
  // Alpaca: NVDA250117C00500000 (no padding on symbol)
  const exp = opt.expiry.replace(/-/g, '').slice(2);
  const strike = Math.round(opt.strike * 1000).toString().padStart(8, '0');
  return `${opt.underlying}${exp}${opt.right}${strike}`;
}

function toORATS(opt: CanonicalOptionId): object {
  // ORATS uses query params
  return {
    ticker: opt.underlying,
    expirDate: opt.expiry,
    strike: opt.strike,
    callPut: opt.right === "C" ? "call" : "put"
  };
}
```

### 2.3 Corporate Action Handling

```typescript
interface CorporateActionAdjustment {
  originalId: CanonicalOptionId;
  adjustedId: CanonicalOptionId;
  adjustmentType: "split" | "reverse_split" | "special_dividend" | "merger";
  effectiveDate: string;
  multiplierChange?: number;
  strikeAdjustment?: number;
}

// TODO: Implement adjustment detection via ORATS corporate actions endpoint
// For MVP: Log warning if contract has non-standard multiplier
```

---

## 3. Data Sources and Refresh Cadence

### 3.1 Cadence Table

| Field | Vendor | Cadence | Stale After | Fallback Action |
|-------|--------|---------|-------------|-----------------|
| `bid` / `ask` | Alpaca WebSocket | Real-time stream | 5 seconds | Mark stale, cap confidence to 50% |
| `last_price` | Alpaca WebSocket | Real-time stream | 10 seconds | Use mid-price estimate |
| `delta` | ORATS REST | 30 seconds | 90 seconds | Disable recommendation |
| `gamma` | ORATS REST | 30 seconds | 90 seconds | Disable recommendation |
| `theta` | ORATS REST | 30 seconds | 90 seconds | Disable recommendation |
| `vega` | ORATS REST | 30 seconds | 90 seconds | Disable recommendation |
| `iv` | ORATS REST | 30 seconds | 90 seconds | Disable recommendation |
| `iv_rank` | ORATS REST | 5 minutes | 15 minutes | Use last known, flag in explanation |
| `iv_percentile` | ORATS REST | 5 minutes | 15 minutes | Use last known, flag in explanation |
| `open_interest` | Alpaca REST | 60 seconds | 5 minutes | Use last known |
| `volume` | Alpaca REST | 60 seconds | 5 minutes | Use last known |
| `underlying_price` | Alpaca WebSocket | Real-time stream | 2 seconds | CRITICAL: Halt all recommendations |
| `news_sentiment` | Finnhub REST | 60 seconds | 10 minutes | Exclude from signal, note in explanation |
| `social_sentiment` | Quiver REST | 5 minutes | 30 minutes | Exclude from signal, note in explanation |
| `portfolio_cash` | Alpaca REST | 30 seconds | 2 minutes | Use last known, flag staleness |
| `portfolio_positions` | Alpaca REST | 30 seconds | 2 minutes | Use last known, flag staleness |

### 3.2 Rate Limit Budget (ORATS: 100,000 requests/month)

**Daily budget:** ~3,200 requests/day

| Operation | Frequency | Requests/Day | Monthly Total |
|-----------|-----------|--------------|---------------|
| Greeks refresh (10 symbols × 40 strikes) | Every 30s during market hours (6.5 hrs) | 400 × 780 / 30 = 10,400 | Too high! |
| **Optimized:** Greeks refresh (10 symbols × 20 strikes) | Every 60s | 200 × 390 = 78,000 | ~78,000 ✓ |
| IV rank/percentile (10 symbols) | Every 5 min | 10 × 78 = 780 | ~23,400 ✓ |
| Daily scan (50 symbols) | 3x/day | 150 | ~4,500 ✓ |

**Implementation:** Fetch Greeks for ±10 strikes around ATM only. Expand on demand if user scrolls.

### 3.3 Data Contracts

```typescript
interface QuoteData {
  canonicalId: CanonicalOptionId;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  last: number | null;
  timestamp: string;          // ISO 8601 UTC
  receiveTimestamp: string;   // Local receive time
  source: "alpaca" | "tradier" | "polygon";
}

interface GreeksData {
  canonicalId: CanonicalOptionId;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  iv: number;
  theoreticalValue: number;
  timestamp: string;
  source: "orats" | "calculated";
}

interface UnderlyingData {
  symbol: string;
  price: number;
  ivRank: number;           // 0-100
  ivPercentile: number;     // 0-100
  timestamp: string;
}
```

---

## 4. Recommendation Contract Schema

Every recommendation is a complete, self-describing object. If any required field cannot be populated, the system outputs an ABSTAIN instead.

### 4.1 Recommendation Object

```typescript
interface Recommendation {
  // Identity
  id: string;                           // UUID
  generatedAt: string;                  // ISO 8601 UTC
  
  // Trade Parameters
  action: "BUY_CALL" | "BUY_PUT" | "SELL_CALL" | "SELL_PUT";
  canonicalId: CanonicalOptionId;
  
  entryPrice: {
    limit: number;                      // Recommended limit price
    currentMid: number;                 // Mid at generation time
    currentSpread: number;              // Ask - Bid
    spreadPercent: number;              // Spread / Mid * 100
  };
  
  positionSize: {
    contracts: number;
    maxRiskDollars: number;
    percentOfPortfolio: number;
    reasoning: string;                  // e.g., "2% risk rule, $500 max loss"
  };
  
  targets: {
    profitTargetPercent: number;        // e.g., 50 for 50% profit
    profitTargetPrice: number;
    stopLossPercent: number;            // e.g., -100 for full loss acceptable
    stopLossPrice: number;
    timeStop: string | null;            // ISO date to exit regardless (e.g., 21 DTE)
  };
  
  // Strategy Context
  strategyType: "0DTE" | "WEEKLY" | "MONTHLY" | "45DTE" | "LEAP";
  itmOtmReasoning: string;              // e.g., "OTM 0.30 delta for probability play"
  
  // Data Freshness
  dataFreshness: {
    quoteAge: number;                   // Seconds since last quote
    greeksAge: number;                  // Seconds since last Greeks update
    sentimentAge: number | null;        // Seconds, or null if not used
    underlyingAge: number;              // Seconds since last underlying price
    allFresh: boolean;                  // True if all within thresholds
  };
  
  // Liquidity Metrics
  liquidity: {
    spreadPercent: number;
    openInterest: number;
    dailyVolume: number;
    bidSize: number;
    askSize: number;
    liquidityScore: number;             // 0-100 composite
    liquidityGrade: "A" | "B" | "C" | "F";
  };
  
  // Greeks Snapshot
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    iv: number;
  };
  
  // Risk Context
  risk: {
    maxLossDollars: number;
    ivRank: number;
    ivPercentile: number;
    daysToExpiry: number;
    earningsWithinExpiry: boolean;
    exDivWithinExpiry: boolean;
  };
  
  // Portfolio Context
  portfolio: {
    availableCash: number;
    currentPositionInUnderlying: boolean;
    netDeltaExposure: number;           // Portfolio-wide
    correlatedExposure: string[];       // Other symbols with similar exposure
  };
  
  // Explanation
  explanation: {
    primaryDriver: string;              // e.g., "High IV rank (78%) favors selling"
    supportingFactors: string[];
    cautionFlags: string[];
    confidenceScore: number;            // 0-100
  };
  
  // Gating Results
  gatesStatus: {
    allPassed: boolean;
    gates: GateResult[];
  };
}

interface GateResult {
  gateName: string;
  passed: boolean;
  value: number | string | boolean;
  threshold: number | string | boolean;
  message: string;
}
```

### 4.2 Abstain Object

When the system cannot or should not recommend:

```typescript
interface Abstain {
  id: string;
  generatedAt: string;
  underlying: string;
  reason: AbstainReason;
  failedGates: GateResult[];
  dataFreshness: Recommendation["dataFreshness"];
  resumeCondition: string;              // e.g., "Spread must narrow to < 5%"
}

type AbstainReason = 
  | "STALE_DATA"
  | "LIQUIDITY_INSUFFICIENT"
  | "SPREAD_TOO_WIDE"
  | "FEED_DEGRADED"
  | "GATES_FAILED"
  | "NO_CLEAR_SIGNAL"
  | "PORTFOLIO_CONSTRAINT"
  | "KILL_SWITCH_ACTIVE";
```

---

## 5. Gating Pipeline

The pipeline processes in order. Any gate failure triggers ABSTAIN.

### 5.1 Pipeline Stages

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  1. DATA    │ → │ 2. LIQUIDITY│ → │ 3. STRATEGY │ → │ 4. PORTFOLIO│ → │ 5. EXPLAIN  │
│  FRESHNESS  │   │   SCORING   │   │   FIT       │   │ CONSTRAINTS │   │   PAYLOAD   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
      │                  │                 │                 │                 │
      ▼                  ▼                 ▼                 ▼                 ▼
   ABSTAIN           ABSTAIN           ABSTAIN           ABSTAIN        RECOMMENDATION
   if stale         if illiquid       if no fit        if blocked
```

### 5.2 Gate Definitions

```typescript
interface GateConfig {
  name: string;
  evaluate: (context: GateContext) => GateResult;
  severity: "hard" | "soft";  // hard = instant ABSTAIN, soft = cap confidence
}

const GATES: GateConfig[] = [
  // Stage 1: Data Freshness
  {
    name: "underlying_price_fresh",
    severity: "hard",
    evaluate: (ctx) => ({
      passed: ctx.underlyingAge <= 2,
      value: ctx.underlyingAge,
      threshold: 2,
      message: ctx.underlyingAge <= 2 ? "OK" : "Underlying price stale"
    })
  },
  {
    name: "quote_fresh",
    severity: "hard",
    evaluate: (ctx) => ({
      passed: ctx.quoteAge <= 5,
      value: ctx.quoteAge,
      threshold: 5,
      message: ctx.quoteAge <= 5 ? "OK" : "Quote data stale"
    })
  },
  {
    name: "greeks_fresh",
    severity: "hard",
    evaluate: (ctx) => ({
      passed: ctx.greeksAge <= 90,
      value: ctx.greeksAge,
      threshold: 90,
      message: ctx.greeksAge <= 90 ? "OK" : "Greeks data stale"
    })
  },
  
  // Stage 2: Liquidity
  {
    name: "spread_acceptable",
    severity: "hard",
    evaluate: (ctx) => ({
      passed: ctx.spreadPercent <= 10,
      value: ctx.spreadPercent,
      threshold: 10,
      message: ctx.spreadPercent <= 10 ? "OK" : `Spread ${ctx.spreadPercent.toFixed(1)}% exceeds 10%`
    })
  },
  {
    name: "open_interest_sufficient",
    severity: "hard",
    evaluate: (ctx) => ({
      passed: ctx.openInterest >= 100,
      value: ctx.openInterest,
      threshold: 100,
      message: ctx.openInterest >= 100 ? "OK" : `OI ${ctx.openInterest} below 100`
    })
  },
  {
    name: "volume_sufficient",
    severity: "soft",
    evaluate: (ctx) => ({
      passed: ctx.volume >= 50,
      value: ctx.volume,
      threshold: 50,
      message: ctx.volume >= 50 ? "OK" : `Volume ${ctx.volume} is thin`
    })
  },
  
  // Stage 3: Strategy Fit
  {
    name: "iv_rank_appropriate",
    severity: "soft",
    evaluate: (ctx) => {
      const buyingPremium = ctx.action.includes("BUY");
      const ivAppropriate = buyingPremium ? ctx.ivRank < 50 : ctx.ivRank > 30;
      return {
        passed: ivAppropriate,
        value: ctx.ivRank,
        threshold: buyingPremium ? "< 50" : "> 30",
        message: ivAppropriate ? "OK" : `IV rank ${ctx.ivRank} not ideal for ${ctx.action}`
      };
    }
  },
  {
    name: "delta_in_range",
    severity: "hard",
    evaluate: (ctx) => {
      const deltaAbs = Math.abs(ctx.delta);
      const inRange = deltaAbs >= 0.10 && deltaAbs <= 0.80;
      return {
        passed: inRange,
        value: deltaAbs,
        threshold: "0.10 - 0.80",
        message: inRange ? "OK" : `Delta ${deltaAbs.toFixed(2)} outside tradeable range`
      };
    }
  },
  
  // Stage 4: Portfolio Constraints
  {
    name: "cash_available",
    severity: "hard",
    evaluate: (ctx) => {
      const requiredCash = ctx.contracts * ctx.premium * 100;
      const hasEnough = ctx.availableCash >= requiredCash;
      return {
        passed: hasEnough,
        value: ctx.availableCash,
        threshold: requiredCash,
        message: hasEnough ? "OK" : `Need $${requiredCash}, have $${ctx.availableCash}`
      };
    }
  },
  {
    name: "position_size_limit",
    severity: "hard",
    evaluate: (ctx) => {
      const positionValue = ctx.contracts * ctx.premium * 100;
      const percentOfPortfolio = (positionValue / ctx.portfolioValue) * 100;
      const withinLimit = percentOfPortfolio <= 5;
      return {
        passed: withinLimit,
        value: percentOfPortfolio,
        threshold: 5,
        message: withinLimit ? "OK" : `Position ${percentOfPortfolio.toFixed(1)}% exceeds 5% limit`
      };
    }
  },
  {
    name: "sector_concentration",
    severity: "soft",
    evaluate: (ctx) => {
      const sectorExposure = ctx.currentSectorExposurePercent + ctx.newPositionPercent;
      const withinLimit = sectorExposure <= 25;
      return {
        passed: withinLimit,
        value: sectorExposure,
        threshold: 25,
        message: withinLimit ? "OK" : `Sector exposure ${sectorExposure.toFixed(1)}% high`
      };
    }
  }
];
```

### 5.3 Sentiment Gating (0DTE vs 45DTE)

```typescript
interface SentimentGate {
  mode: "0DTE" | "STANDARD";
  config: SentimentGateConfig;
}

const SENTIMENT_GATES = {
  "0DTE": {
    // Sentiment can ARM a recommendation
    sentimentCanTrigger: true,
    // But market must CONFIRM before FIRE
    requireMarketConfirmation: true,
    confirmationGates: [
      { gate: "spread_percent", threshold: 3 },     // Tighter for 0DTE
      { gate: "volume_last_5min", threshold: 20 },  // Recent activity
      { gate: "price_momentum_aligned", threshold: true },
      { gate: "feed_latency_ms", threshold: 500 }   // Fresh data critical
    ],
    // If any confirmation fails, HOLD (don't fire, don't fully abstain)
    onConfirmationFail: "HOLD"
  },
  
  "STANDARD": {
    // Sentiment is a confidence modifier only
    sentimentCanTrigger: false,
    // Used to adjust position size
    sentimentWeight: 0.2,
    // High sentiment disagreement = reduce size
    disagreementPenalty: 0.5
  }
};
```

---

## 6. Evaluation and Logging

### 6.1 Shadow Mode Logging Schema

Every recommendation (including ABSTAINs) is logged for later analysis:

```typescript
interface RecommendationLog {
  // Input State
  timestamp: string;
  underlying: string;
  underlyingPrice: number;
  
  // Full recommendation or abstain object
  output: Recommendation | Abstain;
  
  // Snapshot of all inputs
  inputSnapshot: {
    quotes: QuoteData[];
    greeks: GreeksData[];
    sentiment: SentimentData | null;
    portfolio: PortfolioSnapshot;
  };
  
  // Outcome (filled in later)
  outcome?: {
    recordedAt: string;
    priceAt15Min: number;
    priceAt1Hr: number;
    priceAtClose: number;
    optionPriceAt15Min: number;
    optionPriceAt1Hr: number;
    optionPriceAtClose: number;
    wouldHaveProfited: boolean;
    theoreticalPnL: number;
  };
}
```

### 6.2 Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| `recommendation_rate` | Recommendations / Hour | Context-dependent |
| `abstention_rate` | Abstains / Total Signals | 30-70% (healthy caution) |
| `spread_at_rec_time` | Average spread % when recommending | < 5% |
| `time_to_signal` | Seconds from catalyst to recommendation | < 30s for 0DTE |
| `regret_rate` | Abstains that would have been profitable | Track, minimize over time |
| `false_positive_rate` | Recommendations that would have lost | Track, target < 50% |
| `data_staleness_events` | Times we abstained due to stale data | Minimize |
| `gate_failure_distribution` | Which gates fail most often | Identify bottlenecks |

### 6.3 Offline Replay Format

```typescript
interface ReplaySession {
  sessionId: string;
  startTime: string;
  endTime: string;
  symbols: string[];
  
  // Time-series of market state
  marketTicks: MarketTick[];
  
  // Expected outputs at each decision point
  expectedDecisions: ExpectedDecision[];
}

interface MarketTick {
  timestamp: string;
  quotes: Map<string, QuoteData>;
  greeks: Map<string, GreeksData>;
  underlying: Map<string, UnderlyingData>;
  sentiment: Map<string, SentimentData>;
}
```

---

## 7. Production Hygiene

### 7.1 Secrets Management

```bash
# .env file (NOT committed to git)
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
ORATS_API_TOKEN=xxx
FINNHUB_API_KEY=xxx
QUIVER_API_KEY=xxx

# Load in application
from dotenv import load_dotenv
load_dotenv()
```

**Requirements:**
- `.env` in `.gitignore`
- Separate `.env.paper` and `.env.live` for different modes
- Rotate keys quarterly

### 7.2 Kill Switch

```typescript
interface KillSwitch {
  enabled: boolean;
  reason: string;
  triggeredAt: string;
  autoResetAt: string | null;
}

const KILL_SWITCH_TRIGGERS = [
  {
    name: "feed_latency_spike",
    condition: (metrics) => metrics.avgLatencyMs > 2000,
    autoReset: true,
    resetAfterMs: 60000
  },
  {
    name: "error_rate_spike",
    condition: (metrics) => metrics.errorRatePer5Min > 10,
    autoReset: true,
    resetAfterMs: 300000
  },
  {
    name: "spread_anomaly",
    condition: (metrics) => metrics.avgSpreadPercent > 20,
    autoReset: true,
    resetAfterMs: 60000
  },
  {
    name: "manual_halt",
    condition: () => false,  // Only triggered manually
    autoReset: false
  }
];
```

### 7.3 Observability

**Minimum metrics to expose:**

```typescript
interface SystemMetrics {
  // Latency
  quoteLatencyP50Ms: number;
  quoteLatencyP99Ms: number;
  greeksFetchLatencyMs: number;
  
  // Throughput
  quotesPerSecond: number;
  recommendationsPerHour: number;
  
  // Health
  feedConnected: boolean;
  lastQuoteTimestamp: string;
  lastGreeksTimestamp: string;
  apiErrorsLast5Min: number;
  
  // Rate limits
  oratsRequestsRemaining: number;
  oratsRequestsUsedToday: number;
}
```

**Logging levels:**
- `ERROR`: Gate failures, API errors, kill switch triggers
- `WARN`: Soft gate failures, staleness events, approaching rate limits
- `INFO`: Recommendations generated, abstains
- `DEBUG`: All quote updates, Greeks refreshes (disable in production)

---

## 8. MVP Vertical Slice

**Goal:** Prove data plumbing + contract + gating works before strategy sophistication.

### 8.1 MVP Scope

| Component | MVP Scope |
|-----------|-----------|
| Symbols | 1 symbol: NVDA |
| Expirations | 2 buckets: nearest weekly + ~45 DTE |
| Strikes | ±10 strikes around ATM |
| Data | Streaming quotes + 60s Greeks refresh |
| Recommendations | Single type: "WATCH/ARM" signals only |
| Gating | All hard gates implemented |
| UI | Chain display, freshness indicators, abstain reasons |

### 8.2 MVP Validates

- [ ] Symbol mapping works across Alpaca → ORATS
- [ ] Quote/Greeks alignment (timestamps reconcile)
- [ ] Rate limit reality (staying within ORATS budget)
- [ ] UI thrash behavior (updates don't cause flicker)
- [ ] "ABSTAIN by default" posture feels usable
- [ ] Logging captures all decision inputs

### 8.3 MVP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chain View  │  │  Status Bar │  │  Abstain/Watch Panel    │  │
│  │ (strikes,   │  │  (freshness,│  │  (why not recommending) │  │
│  │  Greeks)    │  │   latency)  │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  WebSocket  │  │   Gating    │  │    Data Aggregator      │  │
│  │  Manager    │  │   Engine    │  │  (merge quotes+Greeks)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────┐                   ┌─────────────────────────┐
│ Alpaca WebSocket│                   │      ORATS REST         │
│ (quotes stream) │                   │   (Greeks polling)      │
└─────────────────┘                   └─────────────────────────┘
```

### 8.4 File Structure (MVP)

```
options-radar/
├── backend/
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # Environment loading
│   ├── models/
│   │   ├── canonical.py        # CanonicalOptionId
│   │   ├── recommendation.py   # Recommendation, Abstain
│   │   └── market_data.py      # QuoteData, GreeksData
│   ├── data/
│   │   ├── alpaca_client.py    # WebSocket + REST
│   │   ├── orats_client.py     # REST client
│   │   └── aggregator.py       # Merge data sources
│   ├── engine/
│   │   ├── gates.py            # Gate definitions
│   │   ├── pipeline.py         # Gating pipeline
│   │   └── recommender.py      # Recommendation logic
│   ├── websocket/
│   │   └── manager.py          # Connection management
│   └── logging/
│       └── shadow_logger.py    # Recommendation logging
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── store/
│   │   │   └── optionsStore.ts # Zustand store
│   │   ├── components/
│   │   │   ├── ChainView.tsx
│   │   │   ├── StatusBar.tsx
│   │   │   └── AbstainPanel.tsx
│   │   └── hooks/
│   │       └── useOptionsStream.ts
│   └── package.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 9. Development Phases

### Phase 1: Data Foundation
- [ ] Canonical ID implementation + vendor mappings
- [ ] Alpaca WebSocket connection (quotes stream)
- [ ] ORATS REST client (Greeks polling)
- [ ] Data aggregator with timestamp alignment
- [ ] Staleness detection

### Phase 2: Gating Engine
- [ ] Gate framework implementation
- [ ] All hard gates
- [ ] Pipeline orchestration
- [ ] Abstain generation

### Phase 3: MVP UI
- [ ] FastAPI WebSocket endpoint
- [ ] React chain view
- [ ] Freshness indicators
- [ ] Abstain panel

### Phase 4: Recommendation Logic
- [ ] Strike selection algorithms
- [ ] Position sizing
- [ ] Profit/loss targets
- [ ] Full recommendation generation

### Phase 5: Evaluation Infrastructure
- [ ] Shadow mode logging
- [ ] Outcome recording
- [ ] Metrics dashboard
- [ ] Offline replay framework

### Phase 6: Expansion
- [ ] Additional symbols
- [ ] Sentiment integration
- [ ] Daily scanner
- [ ] Portfolio integration (Alpaca read-only)

---

## 10. Appendix: Quick Reference

### API Endpoints

| Service | Endpoint | Auth |
|---------|----------|------|
| Alpaca Options Stream | `wss://stream.data.alpaca.markets/v1beta1/opra` | Header: `APCA-API-KEY-ID`, `APCA-API-SECRET-KEY` |
| Alpaca Options REST | `https://data.alpaca.markets/v1beta1/options` | Same |
| ORATS Live | `https://api.orats.io/datav2/live/strikes` | Query: `token=xxx` |
| ORATS IV | `https://api.orats.io/datav2/live/ivrank` | Query: `token=xxx` |
| Finnhub News | `https://finnhub.io/api/v1/news-sentiment` | Query: `token=xxx` |

### Key Thresholds (Defaults)

| Parameter | Value | Adjustable |
|-----------|-------|------------|
| Max spread % | 10% | Yes |
| Min open interest | 100 | Yes |
| Min volume | 50 | Yes |
| Quote staleness | 5s | Yes |
| Greeks staleness | 90s | Yes |
| Max position % of portfolio | 5% | Yes |
| Max sector exposure | 25% | Yes |
| Target delta range | 0.10 - 0.80 | Yes |

### MsgPack Handling (Alpaca)

```python
import msgpack

def decode_alpaca_message(raw_bytes: bytes) -> dict:
    return msgpack.unpackb(raw_bytes, raw=False)
```

---

**End of Specification**

*This document should be updated as implementation reveals gaps. Version control all changes.*
