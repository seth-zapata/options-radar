/**
 * Zustand store for options data state management.
 */

import { create } from 'zustand';
import type { OptionData, UnderlyingData, AbstainData, ConnectionStatus, GateResult } from '../types';
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

  // Actions
  setConnectionStatus: (status: ConnectionStatus) => void;
  updateOption: (option: OptionData) => void;
  updateUnderlying: (underlying: UnderlyingData) => void;
  setAbstain: (abstain: AbstainData | null) => void;
  setGateResults: (results: GateResult[], evaluatedOption?: EvaluatedOption) => void;
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

  clearAll: () => set({
    options: new Map(),
    underlying: null,
    abstain: null,
    gateResults: [],
    evaluatedOption: null,
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
