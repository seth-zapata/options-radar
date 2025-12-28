/**
 * Zustand store for options data state management.
 */

import { create } from 'zustand';
import type { OptionData, UnderlyingData, AbstainData, ConnectionStatus, GateResult, Recommendation, SessionStatus, TrackedPosition, ExitSignal } from '../types';
import { optionKey } from '../types';

export interface EvaluatedOption {
  strike: number;
  right: string;
  expiry: string;
  premium: number | null;
}

interface OptionsState {
  // Connection
  connectionStatus: ConnectionStatus;
  lastMessageTime: number | null;

  // Market Data
  options: Map<string, OptionData>;
  underlying: UnderlyingData | null;

  // Gating
  abstain: AbstainData | null;
  gateResults: GateResult[];
  evaluatedOption: EvaluatedOption | null;

  // Recommendations
  recommendations: Recommendation[];
  sessionStatus: SessionStatus | null;

  // Positions
  positions: TrackedPosition[];
  exitSignals: ExitSignal[];

  // Actions
  setConnectionStatus: (status: ConnectionStatus) => void;
  updateOption: (option: OptionData) => void;
  updateUnderlying: (underlying: UnderlyingData) => void;
  setAbstain: (abstain: AbstainData | null) => void;
  setGateResults: (results: GateResult[], evaluatedOption?: EvaluatedOption) => void;
  addRecommendation: (rec: Recommendation) => void;
  setSessionStatus: (status: SessionStatus) => void;
  addPosition: (position: TrackedPosition) => void;
  updatePosition: (position: TrackedPosition) => void;
  addExitSignal: (signal: ExitSignal) => void;
  clearExitSignal: (positionId: string) => void;
  clearAll: () => void;
}

export const useOptionsStore = create<OptionsState>((set) => ({
  // Initial state
  connectionStatus: 'disconnected',
  lastMessageTime: null,
  options: new Map(),
  underlying: null,
  abstain: null,
  gateResults: [],
  evaluatedOption: null,
  recommendations: [],
  sessionStatus: null,
  positions: [],
  exitSignals: [],

  // Actions
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  updateOption: (option) => set((state) => {
    const newOptions = new Map(state.options);
    newOptions.set(optionKey(option.canonicalId), option);
    return {
      options: newOptions,
      lastMessageTime: Date.now(),
    };
  }),

  updateUnderlying: (underlying) => set({
    underlying,
    lastMessageTime: Date.now(),
  }),

  setAbstain: (abstain) => set({ abstain }),

  setGateResults: (results, evaluatedOption) => set({
    gateResults: results,
    evaluatedOption: evaluatedOption ?? null,
  }),

  addRecommendation: (rec) => set((state) => {
    // Check if recommendation already exists (by ID)
    const exists = state.recommendations.some((r) => r.id === rec.id);
    if (exists) return state;

    // Add new recommendation to the front, keep last 20
    const newRecs = [rec, ...state.recommendations].slice(0, 20);
    return { recommendations: newRecs };
  }),

  setSessionStatus: (status) => set({ sessionStatus: status }),

  addPosition: (position) => set((state) => {
    const exists = state.positions.some((p) => p.id === position.id);
    if (exists) return state;
    return { positions: [position, ...state.positions] };
  }),

  updatePosition: (position) => set((state) => ({
    positions: state.positions.map((p) =>
      p.id === position.id ? position : p
    ),
  })),

  addExitSignal: (signal) => set((state) => {
    const exists = state.exitSignals.some((s) => s.positionId === signal.positionId);
    if (exists) return state;
    return { exitSignals: [...state.exitSignals, signal] };
  }),

  clearExitSignal: (positionId) => set((state) => ({
    exitSignals: state.exitSignals.filter((s) => s.positionId !== positionId),
  })),

  clearAll: () => set({
    options: new Map(),
    underlying: null,
    abstain: null,
    gateResults: [],
    evaluatedOption: null,
    recommendations: [],
    sessionStatus: null,
    positions: [],
    exitSignals: [],
  }),
}));

// Selector helpers
export const selectOptionsByExpiry = (state: OptionsState) => {
  const byExpiry = new Map<string, OptionData[]>();

  state.options.forEach((option) => {
    const expiry = option.canonicalId.expiry;
    if (!byExpiry.has(expiry)) {
      byExpiry.set(expiry, []);
    }
    byExpiry.get(expiry)!.push(option);
  });

  // Sort by strike within each expiry
  byExpiry.forEach((options) => {
    options.sort((a, b) => a.canonicalId.strike - b.canonicalId.strike);
  });

  return byExpiry;
};

export const selectCallsAndPuts = (state: OptionsState, expiry: string) => {
  const calls: OptionData[] = [];
  const puts: OptionData[] = [];

  state.options.forEach((option) => {
    if (option.canonicalId.expiry !== expiry) return;

    if (option.canonicalId.right === 'C') {
      calls.push(option);
    } else {
      puts.push(option);
    }
  });

  calls.sort((a, b) => a.canonicalId.strike - b.canonicalId.strike);
  puts.sort((a, b) => a.canonicalId.strike - b.canonicalId.strike);

  return { calls, puts };
};
