/**
 * TSLA Trading Dashboard - Unified view for entry and exit.
 * Combines regime signals, active positions, and collapsible options chain.
 */

import { useEffect, useState } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { RegimeStatus, RegimeSignal, RegimeType, TrackedPosition, ExitSignal } from '../types';
import { ChainView } from './ChainView';
import { TradingControlPanel } from './TradingControlPanel';

const API_BASE = 'http://localhost:8000';

// Regime badge colors
const regimeBadgeColors: Record<RegimeType | 'neutral', { bg: string; text: string; border: string }> = {
  strong_bullish: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-500' },
  moderate_bullish: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-400' },
  moderate_bearish: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-400' },
  strong_bearish: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-500' },
  neutral: { bg: 'bg-slate-100', text: 'text-slate-600', border: 'border-slate-300' },
};

const regimeLabels: Record<RegimeType | 'neutral', string> = {
  strong_bullish: 'STRONG BULLISH',
  moderate_bullish: 'BULLISH',
  moderate_bearish: 'BEARISH',
  strong_bearish: 'STRONG BEARISH',
  neutral: 'NEUTRAL',
};

function formatTimeAgo(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function formatAction(action: string): string {
  switch (action) {
    case 'BUY_CALL': return 'Long Call';
    case 'BUY_PUT': return 'Long Put';
    case 'SELL_CALL': return 'Short Call';
    case 'SELL_PUT': return 'Short Put';
    default: return action;
  }
}

function getUrgencyColors(urgency: ExitSignal['urgency']) {
  switch (urgency) {
    case 'high':
      return 'bg-red-600 text-white border-red-700';
    case 'medium':
      return 'bg-amber-500 text-white border-amber-600';
    case 'low':
      return 'bg-yellow-400 text-yellow-900 border-yellow-500';
  }
}

// Compact Position Card for trading dashboard
function CompactPositionCard({ position, exitSignal }: { position: TrackedPosition; exitSignal?: ExitSignal }) {
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [closePrice, setClosePrice] = useState('');
  const [closing, setClosing] = useState(false);
  const [dismissing, setDismissing] = useState(false);
  const clearExitSignal = useOptionsStore((state) => state.clearExitSignal);

  const pnlColor = position.pnl >= 0 ? 'text-green-600' : 'text-red-600';
  const hasExitSignal = position.status === 'exit_signal' && exitSignal;

  const handleDismissSignal = async () => {
    setDismissing(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/${position.id}/dismiss-exit-signal`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to dismiss');
      clearExitSignal(position.id);
    } catch (error) {
      console.error('Error dismissing signal:', error);
    } finally {
      setDismissing(false);
    }
  };

  const handleClose = async () => {
    const price = parseFloat(closePrice);
    if (isNaN(price) || price <= 0) return;

    setClosing(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/${position.id}/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ close_price: price }),
      });
      if (!response.ok) throw new Error('Failed to close');
      setShowCloseModal(false);
    } catch (error) {
      console.error('Error closing:', error);
    } finally {
      setClosing(false);
    }
  };

  return (
    <div className={`p-3 rounded-lg border-2 ${
      hasExitSignal ? 'border-red-400 bg-red-50' : 'border-slate-200 bg-white'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
            position.action.includes('CALL') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            {formatAction(position.action)}
          </span>
          <span className="font-mono text-sm font-medium">
            ${position.strike} {position.right === 'C' ? 'Call' : 'Put'}
          </span>
          <span className="text-xs text-slate-500">{position.expiry}</span>
        </div>
        <div className={`text-lg font-bold ${pnlColor}`}>
          {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(0)}
          <span className="text-sm ml-1">({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(0)}%)</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-sm text-slate-600">
        <div className="flex gap-4">
          <span>Entry: ${position.fillPrice.toFixed(2)}</span>
          <span>Current: ${position.currentPrice?.toFixed(2) || '-'}</span>
          <span>{position.contracts} contracts</span>
          {position.dte !== null && <span>{position.dte} DTE</span>}
        </div>
        <span className="text-xs text-slate-400">{formatTimeAgo(position.openedAt)}</span>
      </div>

      {/* Exit Signal Banner */}
      {hasExitSignal && exitSignal && (
        <div className={`mt-2 p-2 rounded border ${getUrgencyColors(exitSignal.urgency)}`}>
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm">EXIT: {exitSignal.reason}</span>
            <span className="text-xs uppercase">{exitSignal.urgency}</span>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2 mt-2">
        <button
          onClick={() => {
            if (position.currentPrice) setClosePrice(position.currentPrice.toFixed(2));
            setShowCloseModal(true);
          }}
          className={`flex-1 py-1.5 rounded text-sm font-medium ${
            hasExitSignal
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
          }`}
        >
          {hasExitSignal ? 'Close Now' : 'Close Position'}
        </button>
        {hasExitSignal && (
          <button
            onClick={handleDismissSignal}
            disabled={dismissing}
            className="px-3 py-1.5 bg-slate-200 hover:bg-slate-300 rounded text-sm font-medium text-slate-700 disabled:opacity-50"
          >
            {dismissing ? '...' : 'Dismiss'}
          </button>
        )}
      </div>

      {/* Close Modal */}
      {showCloseModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-sm w-full mx-4 p-4">
            <h3 className="font-bold text-lg mb-3">Close Position</h3>
            <p className="text-sm text-slate-600 mb-3">
              {position.underlying} ${position.strike} {position.right === 'C' ? 'Call' : 'Put'}
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-1">Close Price</label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={closePrice}
                  onChange={(e) => setClosePrice(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500"
                  autoFocus
                />
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setShowCloseModal(false)}
                className="flex-1 py-2 border border-slate-300 rounded-md hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleClose}
                disabled={closing}
                className="flex-1 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
              >
                {closing ? 'Closing...' : 'Confirm Close'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Compact Signal Row for trading dashboard
function CompactSignalRow({ signal, isConfirmed, onTakeTrade }: {
  signal: RegimeSignal;
  isConfirmed: boolean;
  onTakeTrade: () => void;
}) {
  const isBuy = signal.signal_type === 'BUY_CALL';
  const time = new Date(signal.generated_at).toLocaleTimeString();
  const hasOption = !!signal.option;

  return (
    <div className={`p-3 rounded-lg border ${
      isConfirmed ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-white'
    }`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
            isBuy ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'
          }`}>
            {signal.signal_type}
          </span>
          {hasOption && (
            <span className="font-mono text-sm">
              ${signal.option!.strike} @ ${signal.option!.mid.toFixed(2)}
            </span>
          )}
          {hasOption && (
            <span className="text-xs text-slate-500">
              {signal.option!.expiry} ({signal.option!.dte} DTE)
            </span>
          )}
        </div>
        <span className="text-xs text-slate-400">{time}</span>
      </div>

      <div className="mt-1 text-xs text-slate-500">{signal.trigger_reason}</div>

      {hasOption && !isConfirmed && (
        <button
          onClick={onTakeTrade}
          className={`w-full mt-2 py-1.5 text-sm font-medium rounded ${
            isBuy ? 'bg-green-600 text-white hover:bg-green-700' : 'bg-red-600 text-white hover:bg-red-700'
          }`}
        >
          Take Trade ({signal.option!.suggested_contracts} @ ${signal.option!.total_cost.toFixed(0)})
        </button>
      )}

      {isConfirmed && (
        <div className="mt-2 py-1 text-xs text-center bg-emerald-100 text-emerald-700 rounded font-medium">
          Trade Confirmed
        </div>
      )}
    </div>
  );
}

// Trade History Component
function TradeHistory({ positions }: { positions: TrackedPosition[] }) {
  // Get closed TSLA positions, sorted by most recent first
  const closedPositions = positions
    .filter(p => p.underlying === 'TSLA' && p.status === 'closed')
    .sort((a, b) => {
      if (!a.closedAt || !b.closedAt) return 0;
      return new Date(b.closedAt).getTime() - new Date(a.closedAt).getTime();
    });

  // Calculate stats
  const wins = closedPositions.filter(p => p.pnl > 0).length;
  const losses = closedPositions.filter(p => p.pnl <= 0).length;
  const totalPnl = closedPositions.reduce((sum, p) => sum + p.pnl, 0);
  const winRate = closedPositions.length > 0 ? (wins / closedPositions.length) * 100 : null;

  if (closedPositions.length === 0) {
    return null; // Don't show section if no closed trades
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-indigo-600 text-white flex items-center justify-between">
        <div>
          <h3 className="font-bold">Trade History</h3>
          <p className="text-sm text-indigo-200">{closedPositions.length} closed trade{closedPositions.length !== 1 ? 's' : ''}</p>
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold ${totalPnl >= 0 ? 'text-green-300' : 'text-red-300'}`}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}
          </div>
          <div className="text-xs text-indigo-200">
            {wins}W / {losses}L {winRate !== null && `(${winRate.toFixed(0)}%)`}
          </div>
        </div>
      </div>
      <div className="divide-y divide-slate-100">
        {closedPositions.slice(0, 5).map((position) => (
          <div key={position.id} className="px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className={`w-2 h-2 rounded-full ${position.pnl > 0 ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    position.action.includes('CALL') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {formatAction(position.action)}
                  </span>
                  <span className="font-mono text-sm">${position.strike} {position.right === 'C' ? 'Call' : 'Put'}</span>
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  Entry: ${position.fillPrice.toFixed(2)} → Exit: ${position.closePrice?.toFixed(2) || '-'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className={`font-bold ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(0)}
              </div>
              <div className={`text-xs ${position.pnlPercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
        {closedPositions.length > 5 && (
          <div className="px-4 py-2 text-center text-xs text-slate-400">
            + {closedPositions.length - 5} more trades (see Stats tab for full history)
          </div>
        )}
      </div>
    </div>
  );
}

// Trade Modal
function TradeModal({ signal, onClose, onConfirm }: {
  signal: RegimeSignal;
  onClose: () => void;
  onConfirm: (fillPrice: number, contracts: number) => void;
}) {
  const option = signal.option;
  const [fillPrice, setFillPrice] = useState(option?.mid.toFixed(2) || '');
  const [contracts, setContracts] = useState(option?.suggested_contracts.toString() || '1');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const price = parseFloat(fillPrice);
    const qty = parseInt(contracts, 10);

    if (isNaN(price) || price <= 0) {
      setError('Please enter a valid fill price');
      return;
    }
    if (isNaN(qty) || qty <= 0) {
      setError('Please enter a valid number of contracts');
      return;
    }

    onConfirm(price, qty);
  };

  if (!option) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-4">
          <p className="text-slate-600">No option data available.</p>
          <button onClick={onClose} className="mt-4 w-full px-4 py-2 border rounded-md">Close</button>
        </div>
      </div>
    );
  }

  const isBuyCall = signal.signal_type === 'BUY_CALL';

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className={`px-4 py-3 border-b ${isBuyCall ? 'bg-green-600' : 'bg-red-600'} text-white rounded-t-lg`}>
          <h3 className="font-bold text-lg">Confirm Trade</h3>
          <p className="text-sm opacity-90">
            {signal.signal_type} TSLA ${option.strike} {isBuyCall ? 'Call' : 'Put'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-4">
          <div className="space-y-4">
            <div className="bg-slate-50 rounded p-3 text-sm grid grid-cols-2 gap-2">
              <div><span className="text-slate-500">Expiry:</span> {option.expiry}</div>
              <div><span className="text-slate-500">DTE:</span> {option.dte}</div>
              <div><span className="text-slate-500">Bid/Ask:</span> ${option.bid.toFixed(2)}/${option.ask.toFixed(2)}</div>
              <div><span className="text-slate-500">Mid:</span> <span className="font-bold text-green-600">${option.mid.toFixed(2)}</span></div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Fill Price</label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={fillPrice}
                  onChange={(e) => setFillPrice(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 border rounded-md focus:ring-2 focus:ring-emerald-500"
                  autoFocus
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Contracts</label>
              <input
                type="number"
                min="1"
                value={contracts}
                onChange={(e) => setContracts(e.target.value)}
                className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-emerald-500"
              />
              <p className="text-xs text-slate-500 mt-1">Suggested: {option.suggested_contracts}</p>
            </div>

            <div className="bg-emerald-50 rounded p-3 flex justify-between">
              <span className="text-slate-600">Total Cost</span>
              <span className="font-bold text-lg">${((parseFloat(fillPrice) || 0) * (parseInt(contracts, 10) || 0) * 100).toFixed(0)}</span>
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>

          <div className="flex gap-3 mt-6">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border rounded-md hover:bg-slate-50">
              Cancel
            </button>
            <button
              type="submit"
              className={`flex-1 px-4 py-2 text-white rounded-md ${isBuyCall ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}`}
            >
              Confirm Trade
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function TradingDashboard() {
  const regimeStatus = useOptionsStore((state) => state.regimeStatus);
  const regimeSignals = useOptionsStore((state) => state.regimeSignals);
  const setRegimeStatus = useOptionsStore((state) => state.setRegimeStatus);
  const underlying = useOptionsStore((state) => state.underlying);
  const positions = useOptionsStore((state) => state.positions);
  const exitSignals = useOptionsStore((state) => state.exitSignals);

  const [selectedSignal, setSelectedSignal] = useState<RegimeSignal | null>(null);
  const [confirmedSignals, setConfirmedSignals] = useState<Set<string>>(new Set());
  const [showOptionsChain, setShowOptionsChain] = useState(false);

  // Filter TSLA positions
  const tslaPositions = positions.filter(p => p.underlying === 'TSLA' && (p.status === 'open' || p.status === 'exit_signal'));
  const exitSignalMap = new Map(exitSignals.map(s => [s.positionId, s]));

  // Confirmed signal IDs
  const confirmedSignalIds = new Set([
    ...confirmedSignals,
    ...positions.filter(p => p.recommendationId.startsWith('TSLA-')).map(p => p.recommendationId),
  ]);

  const handleConfirmTrade = async (fillPrice: number, contracts: number) => {
    if (!selectedSignal || !selectedSignal.option) return;

    try {
      const response = await fetch(`${API_BASE}/api/positions/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          underlying: selectedSignal.symbol,
          expiry: selectedSignal.option.expiry,
          strike: selectedSignal.option.strike,
          right: selectedSignal.signal_type === 'BUY_CALL' ? 'C' : 'P',
          action: selectedSignal.signal_type,
          fill_price: fillPrice,
          contracts: contracts,
        }),
      });

      if (!response.ok) throw new Error('Failed to confirm trade');
      setConfirmedSignals(prev => new Set([...prev, selectedSignal.id]));
      setSelectedSignal(null);
    } catch (error) {
      console.error('Error confirming trade:', error);
      alert('Failed to confirm trade');
    }
  };

  // Fetch regime status periodically
  useEffect(() => {
    const fetchRegimeStatus = async () => {
      try {
        const response = await fetch('/api/regime/status?symbol=TSLA');
        if (!response.ok) throw new Error('Failed to fetch');
        const data: RegimeStatus = await response.json();
        setRegimeStatus(data);
      } catch (e) {
        console.error('Failed to fetch regime status:', e);
      }
    };

    fetchRegimeStatus();
    const interval = setInterval(fetchRegimeStatus, 10000);
    return () => clearInterval(interval);
  }, [setRegimeStatus]);

  const activeRegime = regimeStatus?.active_regime;
  const regimeType: RegimeType | 'neutral' = activeRegime?.type || 'neutral';
  const colors = regimeBadgeColors[regimeType];

  // Calculate totals
  const totalPnl = tslaPositions.reduce((sum, p) => sum + p.pnl, 0);
  const totalExposure = tslaPositions.reduce((sum, p) => sum + p.entryCost, 0);

  return (
    <div className="space-y-4">
      {/* Header Row - Price and Regime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* TSLA Price Card */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-800">TSLA</h2>
              <div className="text-sm text-slate-500">Tesla, Inc.</div>
            </div>
            {underlying && (
              <div className="text-right">
                <div className="text-3xl font-bold text-slate-800">${underlying.price.toFixed(2)}</div>
                <div className="text-sm text-slate-500">IV Rank: {underlying.ivRank.toFixed(0)}%</div>
              </div>
            )}
          </div>
        </div>

        {/* Regime Status Card */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div className={`inline-flex items-center px-4 py-2 rounded-full border-2 ${colors.bg} ${colors.text} ${colors.border}`}>
              <span className={`w-2 h-2 rounded-full mr-2 ${activeRegime?.is_active ? 'bg-current animate-pulse' : 'bg-slate-400'}`}></span>
              <span className="font-bold">{regimeLabels[regimeType]}</span>
            </div>
            {activeRegime && (
              <div className="text-right">
                <div className={`text-lg font-semibold ${activeRegime.sentiment_value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {activeRegime.sentiment_value >= 0 ? '+' : ''}{(activeRegime.sentiment_value * 100).toFixed(1)}% sentiment
                </div>
                <div className="text-sm text-slate-500">
                  {activeRegime.days_remaining} day{activeRegime.days_remaining !== 1 ? 's' : ''} remaining
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Auto-Execution Control Panel */}
      <TradingControlPanel />

      {/* Main Content - Two Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Column: Entry Signals */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 bg-emerald-600 text-white rounded-t-lg">
            <h3 className="font-bold">Entry Signals</h3>
            <p className="text-sm text-emerald-200">
              {regimeSignals.length > 0 ? `${regimeSignals.length} signal${regimeSignals.length !== 1 ? 's' : ''}` : 'Waiting for pullback/bounce...'}
            </p>
          </div>
          <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
            {regimeSignals.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                <div className="text-lg mb-1">No signals yet</div>
                <div className="text-sm">Signals appear when regime + pullback/bounce conditions are met</div>
              </div>
            ) : (
              regimeSignals.slice(0, 10).map((signal, idx) => (
                <CompactSignalRow
                  key={signal.id || `${signal.generated_at}-${idx}`}
                  signal={signal}
                  isConfirmed={confirmedSignalIds.has(signal.id)}
                  onTakeTrade={() => setSelectedSignal(signal)}
                />
              ))
            )}
          </div>
        </div>

        {/* Right Column: Open Positions */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 bg-slate-700 text-white rounded-t-lg flex items-center justify-between">
            <div>
              <h3 className="font-bold">Open Positions</h3>
              <p className="text-sm text-slate-300">
                {tslaPositions.length > 0
                  ? `${tslaPositions.length} position${tslaPositions.length !== 1 ? 's' : ''}`
                  : 'No open positions'}
              </p>
            </div>
            {tslaPositions.length > 0 && (
              <div className="text-right">
                <div className={`text-lg font-bold ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}
                </div>
                <div className="text-xs text-slate-400">${totalExposure.toFixed(0)} exposure</div>
              </div>
            )}
          </div>
          <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
            {tslaPositions.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                <div className="text-lg mb-1">No open TSLA positions</div>
                <div className="text-sm">Take a trade from Entry Signals</div>
              </div>
            ) : (
              tslaPositions.map((position) => (
                <CompactPositionCard
                  key={position.id}
                  position={position}
                  exitSignal={exitSignalMap.get(position.id)}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Exit Rules Reference */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <div className="flex gap-6 text-sm">
            <span className="text-green-600 font-medium">Take Profit: +40%</span>
            <span className="text-red-600 font-medium">Stop Loss: -20%</span>
            <span className="text-slate-600">Min DTE Exit: 1</span>
          </div>
          <div className="text-xs text-slate-500">
            Backtested: 71 trades, 43.7% win rate, +1238% return (Jan 2024 - Jan 2025)
          </div>
        </div>
      </div>

      {/* Trade History Section */}
      <TradeHistory positions={positions} />

      {/* Collapsible Options Chain */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <button
          onClick={() => setShowOptionsChain(!showOptionsChain)}
          className="w-full px-4 py-3 flex items-center justify-between bg-slate-50 hover:bg-slate-100 transition-colors"
        >
          <div className="flex items-center gap-2">
            <span className={`transform transition-transform ${showOptionsChain ? 'rotate-90' : ''}`}>
              ▶
            </span>
            <span className="font-medium text-slate-700">Options Chain</span>
            <span className="text-sm text-slate-500">(for manual trades)</span>
          </div>
          <span className="text-sm text-slate-500">
            {showOptionsChain ? 'Click to collapse' : 'Click to expand'}
          </span>
        </button>
        {showOptionsChain && (
          <div className="border-t">
            <ChainView />
          </div>
        )}
      </div>

      {/* Trade Modal */}
      {selectedSignal && (
        <TradeModal
          signal={selectedSignal}
          onClose={() => setSelectedSignal(null)}
          onConfirm={handleConfirmTrade}
        />
      )}
    </div>
  );
}
