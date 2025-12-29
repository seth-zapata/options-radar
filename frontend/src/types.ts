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

export interface WebSocketMessage {
  type: 'option_update' | 'underlying_update' | 'gate_status' | 'abstain' | 'connection_status' | 'error' | 'ping' | 'recommendation' | 'session_status' | 'position_opened' | 'position_closed' | 'position_updated' | 'exit_signal' | 'symbol_changed';
  data?: unknown;
  timestamp?: string;
}

// Helper to create option key for Map
export function optionKey(id: CanonicalOptionId): string {
  return `${id.underlying}-${id.expiry}-${id.strike}-${id.right}`;
}
