/**
 * Zustand store for options data state management.
 */

import { create } from 'zustand';
import type { OptionData, UnderlyingData, AbstainData, ConnectionStatus, GateResult, Recommendation, SessionStatus, TrackedPosition, ExitSignal, HotPicks, RegimeStatus, RegimeSignal, ScalpSignal, ScalpPosition, ScalpExitResult } from '../types';
import { optionKey } from '../types';

export interface EvaluatedOption {
  strike: number;
  right: string;
  expiry: string;
  premium: number | null;
}

// Default symbols for new users (can be removed by user)
// TSLA is first since it's the only regime-enabled symbol
export const DEFAULT_SYMBOLS = ['TSLA', 'QQQ', 'NVDA'];

// Load watchlist from localStorage or use defaults
const getInitialWatchlist = (): string[] => {
  try {
    const saved = localStorage.getItem('options-radar-watchlist');
    if (saved) {
      const parsed = JSON.parse(saved);
      // Return saved watchlist as-is (no forced symbols)
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    }
  } catch (e) {
    console.error('Failed to load watchlist from localStorage:', e);
  }
  return [...DEFAULT_SYMBOLS];
};

// Save watchlist to localStorage
const saveWatchlist = (symbols: string[]) => {
  try {
    localStorage.setItem('options-radar-watchlist', JSON.stringify(symbols));
  } catch (e) {
    console.error('Failed to save watchlist to localStorage:', e);
  }
};

// Track dismissed recommendation IDs in localStorage
const getDismissedRecIds = (): Set<string> => {
  try {
    const saved = localStorage.getItem('options-radar-dismissed-recs');
    if (saved) {
      return new Set(JSON.parse(saved));
    }
  } catch (e) {
    console.error('Failed to load dismissed recommendations:', e);
  }
  return new Set();
};

const saveDismissedRecIds = (ids: Set<string>) => {
  try {
    // Keep only last 100 dismissed IDs to prevent unbounded growth
    const idsArray = Array.from(ids).slice(-100);
    localStorage.setItem('options-radar-dismissed-recs', JSON.stringify(idsArray));
  } catch (e) {
    console.error('Failed to save dismissed recommendations:', e);
  }
};

// Initialize dismissed IDs
let dismissedRecIds = getDismissedRecIds();

interface OptionsState {
  // Connection
  connectionStatus: ConnectionStatus;
  lastMessageTime: number | null;

  // Symbol Selection
  activeSymbol: string;
  watchlist: string[];
  symbolNames: Record<string, string>; // symbol -> company name

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

  // Scanner
  scannerData: HotPicks | null;
  scannerLoading: boolean;
  scannerError: string | null;
  scannerLastUpdate: string | null;

  // Regime Strategy
  regimeStatus: RegimeStatus | null;
  regimeSignals: RegimeSignal[];
  regimeLoading: boolean;

  // Scalping (0DTE/1DTE intraday momentum)
  scalpEnabled: boolean;
  scalpSignals: ScalpSignal[];
  scalpPositions: ScalpPosition[];
  scalpClosedTrades: ScalpExitResult[];  // Recent closed trades for P&L tracking

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
  setActiveSymbol: (symbol: string) => void;
  addToWatchlist: (symbol: string) => void;
  removeFromWatchlist: (symbol: string) => void;
  dismissRecommendation: (recId: string) => void;
  clearExpiredRecommendations: () => void;
  setSymbolName: (symbol: string, name: string) => void;
  setScannerData: (data: HotPicks) => void;
  setScannerLoading: (loading: boolean) => void;
  setScannerError: (error: string | null) => void;
  setRegimeStatus: (status: RegimeStatus) => void;
  addRegimeSignal: (signal: RegimeSignal) => void;
  setRegimeLoading: (loading: boolean) => void;
  // Scalping actions
  setScalpEnabled: (enabled: boolean) => void;
  addScalpSignal: (signal: ScalpSignal) => void;
  addScalpPosition: (position: ScalpPosition) => void;
  updateScalpPosition: (position: ScalpPosition) => void;
  removeScalpPosition: (signalId: string) => void;
  addScalpClosedTrade: (trade: ScalpExitResult) => void;
  clearScalpSignals: () => void;
  clearAll: () => void;
}

const initialWatchlist = getInitialWatchlist();

export const useOptionsStore = create<OptionsState>((set) => ({
  // Initial state
  connectionStatus: 'disconnected',
  lastMessageTime: null,
  activeSymbol: initialWatchlist[0], // Default to first symbol
  watchlist: initialWatchlist,
  symbolNames: {
    QQQ: 'Invesco QQQ Trust',
    NVDA: 'NVIDIA Corporation',
    TSLA: 'Tesla, Inc.',
  },
  options: new Map(),
  underlying: null,
  abstain: null,
  gateResults: [],
  evaluatedOption: null,
  recommendations: [],
  sessionStatus: null,
  positions: [],
  exitSignals: [],
  scannerData: null,
  scannerLoading: false,
  scannerError: null,
  scannerLastUpdate: null,

  // Regime Strategy
  regimeStatus: null,
  regimeSignals: [],
  regimeLoading: false,

  // Scalping
  scalpEnabled: false,
  scalpSignals: [],
  scalpPositions: [],
  scalpClosedTrades: [],

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
    // Skip if already dismissed
    if (dismissedRecIds.has(rec.id)) return state;

    // Skip if already expired
    if (new Date(rec.validUntil) < new Date()) return state;

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

  setActiveSymbol: (symbol) => set({
    activeSymbol: symbol,
    // Clear options data when switching symbols
    options: new Map(),
    underlying: null,
    abstain: null,
    gateResults: [],
    evaluatedOption: null,
  }),

  addToWatchlist: (symbol) => set((state) => {
    const upperSymbol = symbol.toUpperCase();
    if (state.watchlist.includes(upperSymbol)) return state;
    const newWatchlist = [...state.watchlist, upperSymbol];
    saveWatchlist(newWatchlist);
    return { watchlist: newWatchlist };
  }),

  removeFromWatchlist: (symbol) => set((state) => {
    const upperSymbol = symbol.toUpperCase();
    // Don't allow removing the last symbol
    if (state.watchlist.length <= 1) return state;
    const newWatchlist = state.watchlist.filter(s => s !== upperSymbol);
    saveWatchlist(newWatchlist);
    // If we're removing the active symbol, switch to first in list
    const newActiveSymbol = state.activeSymbol === upperSymbol
      ? newWatchlist[0]
      : state.activeSymbol;
    return {
      watchlist: newWatchlist,
      activeSymbol: newActiveSymbol,
      // Clear data if we switched symbols
      ...(state.activeSymbol === upperSymbol ? {
        options: new Map(),
        underlying: null,
        abstain: null,
        gateResults: [],
        evaluatedOption: null,
      } : {}),
    };
  }),

  dismissRecommendation: (recId) => set((state) => {
    // Add to dismissed set and persist
    dismissedRecIds.add(recId);
    saveDismissedRecIds(dismissedRecIds);
    return {
      recommendations: state.recommendations.filter(r => r.id !== recId),
    };
  }),

  clearExpiredRecommendations: () => set((state) => {
    const now = new Date();
    const expired = state.recommendations.filter(r => new Date(r.validUntil) <= now);
    // Add all expired to dismissed set
    expired.forEach(r => dismissedRecIds.add(r.id));
    saveDismissedRecIds(dismissedRecIds);
    return {
      recommendations: state.recommendations.filter(r => {
        // Filter out expired recommendations (validUntil has passed)
        return new Date(r.validUntil) > now;
      }),
    };
  }),

  setSymbolName: (symbol, name) => set((state) => ({
    symbolNames: { ...state.symbolNames, [symbol]: name },
  })),

  setScannerData: (data) => set({
    scannerData: data,
    scannerLoading: false,
    scannerError: null,
    scannerLastUpdate: new Date().toLocaleTimeString(),
  }),

  setScannerLoading: (loading) => set({ scannerLoading: loading }),

  setScannerError: (error) => set({
    scannerError: error,
    scannerLoading: false,
  }),

  setRegimeStatus: (status) => set({
    regimeStatus: status,
    regimeLoading: false,
  }),

  addRegimeSignal: (signal) => set((state) => {
    // Keep last 20 signals
    const newSignals = [signal, ...state.regimeSignals].slice(0, 20);
    return { regimeSignals: newSignals };
  }),

  setRegimeLoading: (loading) => set({ regimeLoading: loading }),

  // Scalping actions
  setScalpEnabled: (enabled) => set({ scalpEnabled: enabled }),

  addScalpSignal: (signal) => set((state) => {
    // Check if signal already exists
    const exists = state.scalpSignals.some((s) => s.id === signal.id);
    if (exists) return state;
    // Keep last 20 signals
    const newSignals = [signal, ...state.scalpSignals].slice(0, 20);
    return { scalpSignals: newSignals };
  }),

  addScalpPosition: (position) => set((state) => {
    // Check if position already exists
    const exists = state.scalpPositions.some((p) => p.signal_id === position.signal_id);
    if (exists) return state;
    // Remove the corresponding signal (it's now a position)
    const newSignals = state.scalpSignals.filter((s) => s.id !== position.signal_id);
    return {
      scalpPositions: [position, ...state.scalpPositions],
      scalpSignals: newSignals,
    };
  }),

  updateScalpPosition: (position) => set((state) => ({
    scalpPositions: state.scalpPositions.map((p) =>
      p.signal_id === position.signal_id ? position : p
    ),
  })),

  removeScalpPosition: (signalId) => set((state) => ({
    scalpPositions: state.scalpPositions.filter((p) => p.signal_id !== signalId),
  })),

  addScalpClosedTrade: (trade) => set((state) => {
    // Remove the position
    const newPositions = state.scalpPositions.filter((p) => p.signal_id !== trade.signal_id);
    // Keep last 50 closed trades
    const newClosedTrades = [trade, ...state.scalpClosedTrades].slice(0, 50);
    return {
      scalpPositions: newPositions,
      scalpClosedTrades: newClosedTrades,
    };
  }),

  clearScalpSignals: () => set({ scalpSignals: [] }),

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
    scalpSignals: [],
    scalpPositions: [],
    scalpClosedTrades: [],
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
