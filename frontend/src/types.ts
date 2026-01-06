/**
 * TypeScript types for OptionsRadar frontend.
 */

export interface CanonicalOptionId {
  underlying: string;
  expiry: string;
  right: 'C' | 'P';
  strike: number;
}

export interface OptionData {
  canonicalId: CanonicalOptionId;
  bid: number | null;
  ask: number | null;
  bidSize: number | null;
  askSize: number | null;
  last: number | null;
  mid: number | null;
  spread: number | null;
  spreadPercent: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
  iv: number | null;
  theoreticalValue: number | null;
  quoteTimestamp: string | null;
  greeksTimestamp: string | null;
  quoteAge: number | null;
  greeksAge: number | null;
}

export interface UnderlyingData {
  symbol: string;
  price: number;
  ivRank: number;
  ivPercentile: number;
  timestamp: string;
  age: number;
}

export interface GateResult {
  name: string;
  passed: boolean;
  value: unknown;
  threshold: unknown;
  message: string;
}

export interface AbstainData {
  reason: string;
  resumeCondition: string;
  failedGates: GateResult[];
}

export type TradeAction = 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT';

export interface RecommendationGateResult {
  name: string;
  passed: boolean;
  message: string;
}

export interface Recommendation {
  id: string;
  generatedAt: string;
  underlying: string;
  action: TradeAction;
  strike: number;
  expiry: string;
  right: 'C' | 'P';
  contracts: number;
  premium: number;
  totalCost: number;
  confidence: number;
  rationale: string;
  gateResults: RecommendationGateResult[];
  quoteAge: number | null;
  greeksAge: number | null;
  underlyingAge: number | null;
  validUntil: string;
}

export interface SessionStatus {
  sessionId: string;
  startedAt: string;
  recommendationCount: number;
  totalExposure: number;
  exposureRemaining: number;
  exposurePercent: number;
  isAtLimit: boolean;
  isWarning: boolean;
  recommendationsBySymbol: Record<string, number>;
  lastRecommendationAt: string | null;
}

export type PositionStatus = 'open' | 'closed' | 'exit_signal';

export interface TrackedPosition {
  id: string;
  recommendationId: string;
  openedAt: string;
  underlying: string;
  expiry: string;
  strike: number;
  right: string;
  action: TradeAction;
  contracts: number;
  fillPrice: number;
  entryCost: number;
  currentPrice: number | null;
  currentValue: number | null;
  pnl: number;
  pnlPercent: number;
  dte: number | null;
  delta: number | null;
  status: PositionStatus;
  exitReason: string | null;
  closedAt: string | null;
  closePrice: number | null;
}

export interface ExitSignal {
  positionId: string;
  reason: string;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  urgency: 'low' | 'medium' | 'high';
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// Scanner types
export interface WSBTrending {
  symbol: string;
  mentions24h: number;
  sentiment: number;
  sentimentScore: number;
  rank: number;
  buzzLevel: string;
  isBullish: boolean;
}

export interface ScanResult {
  symbol: string;
  score: number;
  direction: string;
  isOpportunity: boolean;
  isStrongOpportunity: boolean;
  signals: string[];
  sentiment: {
    scores: {
      news: number;
      wsb: number;
      combined: number;
    };
    signal: string;
    strength: string;
    flags: {
      newsBuzzing: boolean;
      wsbTrending: boolean;
      wsbBullish: boolean;
      sourcesAligned: boolean;
    };
  } | null;
}

export interface HotPicks {
  wsbTrending: WSBTrending[];
  topOpportunities: ScanResult[];
}

// Regime strategy types
export type RegimeType = 'strong_bullish' | 'moderate_bullish' | 'moderate_bearish' | 'strong_bearish' | 'neutral';

export interface ActiveRegime {
  type: RegimeType;
  is_bullish: boolean;
  is_bearish: boolean;
  triggered_at: string;
  expires_at: string;
  sentiment_value: number;
  days_remaining: number;
  is_active: boolean;
}

export interface RegimeConfig {
  strong_bullish_threshold: number;
  moderate_bullish_threshold: number;
  moderate_bearish_threshold: number;
  strong_bearish_threshold: number;
  regime_window_days: number;
  pullback_threshold: number;
  bounce_threshold: number;
  target_dte: number;
  enabled_symbols: string[];
}

export interface RegimeStatus {
  symbol: string;
  active_regime: ActiveRegime | null;
  signal_generator: {
    last_signal_time: string | null;
    signals_generated_today: number;
  };
  config: RegimeConfig;
  timestamp: string;
}

export interface RegimeSignalOption {
  strike: number;
  expiry: string;
  dte: number;
  bid: number;
  ask: number;
  mid: number;
  delta: number | null;
  open_interest: number;
  volume: number;
  suggested_contracts: number;
  total_cost: number;
}

export interface RegimeSignal {
  id: string;
  symbol: string;
  signal_type: 'BUY_CALL' | 'BUY_PUT';
  regime_type: RegimeType;
  trigger_reason: string;
  trigger_pct: number;
  entry_price: number;
  generated_at: string;
  option?: RegimeSignalOption;
}

export interface PriceMonitoring {
  high: number;
  low: number;
  current: number;
  pullback_pct: number;
  bounce_pct: number;
}

// Auto-execution types
export interface TradingStatus {
  configured: boolean;
  enabled: boolean;
  auto_execution: boolean;
  simulation_mode: boolean;
  scalping_enabled: boolean;
  open_positions: number;
  max_positions: number;
  position_size_pct: number;
  exit_monitor_running: boolean;
  error?: string;
}

export interface SimulationStatus {
  active: boolean;
  running: boolean;
  speed: number;
  currentRegime: string;
  sentiment: number;
  prices: Record<string, { price: number; high: number; low: number }>;
  mockPortfolio: {
    cash: number;
    equity: number;
    buyingPower: number;
    openPositions: number;
    unrealizedPl: number;
    closedPl: number;
    totalPl: number;
    positions: Array<{
      symbol: string;
      qty: number;
      entryPrice: number;
      currentPrice: number;
      pl: number;
      plPercent: number;
    }>;
  };
  error?: string;
}

export interface AlpacaAccount {
  equity: number;
  buying_power: number;
  cash: number;
  portfolio_value: number;
  positions_count: number;
  day_trade_count: number;
  pattern_day_trader: boolean;
}

export interface AlpacaPosition {
  symbol: string;
  qty: number;
  market_value: number;
  cost_basis: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  current_price: number;
  avg_entry_price: number;
}

export interface AlpacaOrder {
  id: string;
  symbol: string;
  qty: number;
  side: 'buy' | 'sell';
  type: string;
  status: string;
  filled_qty: number;
  filled_avg_price: number | null;
  submitted_at: string;
}

// Scalping types (0DTE/1DTE intraday momentum)
export type ScalpSignalType = 'SCALP_CALL' | 'SCALP_PUT';
export type ScalpTrigger = 'momentum_burst' | 'vwap_bounce' | 'vwap_rejection' | 'breakout';
export type ScalpExitReason = 'take_profit' | 'stop_loss' | 'time_exit';

export interface ScalpSignal {
  id: string;
  timestamp: string;
  symbol: string;
  signal_type: ScalpSignalType;
  trigger: ScalpTrigger;
  underlying_price: number;
  velocity_pct: number;
  volume_ratio: number;
  option_symbol: string;
  strike: number;
  expiry: string;
  delta: number;
  dte: number;
  bid_price: number;
  ask_price: number;
  entry_price: number;
  spread_pct: number;
  take_profit_pct: number;
  stop_loss_pct: number;
  max_hold_minutes: number;
  confidence: number;
  suggested_contracts: number;
}

export interface ScalpPosition {
  signal_id: string;
  symbol: string;
  signal_type: ScalpSignalType;
  trigger: string;
  confidence: number;
  option_symbol: string;
  strike: number;
  expiry: string;
  delta: number;
  dte: number;
  entry_time: string;
  entry_price: number;
  underlying_at_entry: number;
  contracts: number;
  take_profit_pct: number;
  stop_loss_pct: number;
  max_hold_seconds: number;
  current_price: number | null;
  current_pnl_pct: number;
  hold_seconds: number;
  max_gain_pct: number;
  max_drawdown_pct: number;
}

export interface ScalpExecutionResult {
  success: boolean;
  signal: ScalpSignal;
  position: ScalpPosition | null;
  order_id: string | null;
  error: string | null;
  fill_price: number | null;
}

export interface ScalpExitResult {
  signal_id: string;
  symbol: string;
  signal_type: ScalpSignalType;
  option_symbol: string;
  exit_reason: ScalpExitReason;
  entry_price: number;
  exit_price: number;
  pnl_dollars: number;
  pnl_pct: number;
  hold_seconds: number;
  max_gain_pct: number;
  max_drawdown_pct: number;
  order_id: string | null;
}

export interface WebSocketMessage {
  type: 'option_update' | 'underlying_update' | 'gate_status' | 'abstain' | 'connection_status' | 'error' | 'ping' | 'recommendation' | 'session_status' | 'position_opened' | 'position_closed' | 'position_updated' | 'exit_signal' | 'symbol_changed' | 'regime_signal' | 'regime_status' | 'scalp_signal' | 'scalp_position_opened' | 'scalp_position_closed' | 'scalp_position_update';
  data?: unknown;
  timestamp?: string;
}

// Helper to create option key for Map
export function optionKey(id: CanonicalOptionId): string {
  return `${id.underlying}-${id.expiry}-${id.strike}-${id.right}`;
}
