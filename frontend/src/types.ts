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

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: 'option_update' | 'underlying_update' | 'gate_status' | 'abstain' | 'connection_status' | 'error';
  data: unknown;
  timestamp: string;
}

// Helper to create option key for Map
export function optionKey(id: CanonicalOptionId): string {
  return `${id.underlying}-${id.expiry}-${id.strike}-${id.right}`;
}
