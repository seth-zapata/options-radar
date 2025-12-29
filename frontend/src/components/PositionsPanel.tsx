/**
 * Positions panel showing open and closed tracked positions.
 */

import { useState } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { TrackedPosition, ExitSignal } from '../types';

const API_BASE = 'http://localhost:8000';

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

function formatAction(action: string): string {
  switch (action) {
    case 'BUY_CALL': return 'Long Call';
    case 'BUY_PUT': return 'Long Put';
    case 'SELL_CALL': return 'Short Call';
    case 'SELL_PUT': return 'Short Put';
    default: return action;
  }
}

function formatTimeAgo(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function PositionCard({ position, exitSignal }: { position: TrackedPosition; exitSignal?: ExitSignal }) {
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [closePrice, setClosePrice] = useState('');
  const [closing, setClosing] = useState(false);
  const [dismissing, setDismissing] = useState(false);
  const clearExitSignal = useOptionsStore((state) => state.clearExitSignal);

  const pnlColor = position.pnl >= 0 ? 'text-green-600' : 'text-red-600';
  const pnlBgColor = position.pnl >= 0 ? 'bg-green-50' : 'bg-red-50';
  const hasExitSignal = position.status === 'exit_signal' && exitSignal;

  const handleDismissSignal = async () => {
    setDismissing(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/${position.id}/dismiss-exit-signal`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to dismiss signal');
      }

      clearExitSignal(position.id);
    } catch (error) {
      console.error('Error dismissing signal:', error);
      alert(error instanceof Error ? error.message : 'Failed to dismiss signal');
    } finally {
      setDismissing(false);
    }
  };

  const handleClose = async () => {
    const price = parseFloat(closePrice);
    if (isNaN(price) || price <= 0) {
      alert('Please enter a valid close price');
      return;
    }

    setClosing(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/${position.id}/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ close_price: price }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to close position');
      }

      setShowCloseModal(false);
    } catch (error) {
      console.error('Error closing position:', error);
      alert(error instanceof Error ? error.message : 'Failed to close position');
    } finally {
      setClosing(false);
    }
  };

  return (
    <div className={`rounded-lg border-2 overflow-hidden ${
      hasExitSignal ? 'border-red-400 ring-2 ring-red-200' :
      position.status === 'open' ? 'border-slate-200' : 'border-slate-100 opacity-60'
    }`}>
      {/* Header */}
      <div className={`px-3 py-2 flex items-center justify-between ${
        hasExitSignal ? 'bg-red-100' :
        position.status === 'open' ? 'bg-slate-100' : 'bg-slate-50'
      }`}>
        <div className="flex items-center gap-2">
          <span className="font-bold">{position.underlying}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${
            position.action.includes('CALL') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            {formatAction(position.action)}
          </span>
        </div>
        <span className="text-xs text-slate-500">
          {formatTimeAgo(position.openedAt)}
        </span>
      </div>

      {/* Details */}
      <div className="p-3">
        <div className="font-mono text-sm mb-2">
          ${position.strike} {position.right === 'C' ? 'Call' : 'Put'} • {position.expiry}
        </div>

        <div className="grid grid-cols-3 gap-2 text-sm mb-2">
          <div className="text-center">
            <div className="text-xs text-slate-500">Entry</div>
            <div className="font-medium">${position.fillPrice.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500">Current</div>
            <div className="font-medium">
              {position.currentPrice ? `$${position.currentPrice.toFixed(2)}` : '-'}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500">Contracts</div>
            <div className="font-medium">{position.contracts}</div>
          </div>
        </div>

        {/* P/L Display */}
        <div className={`rounded p-2 text-center ${pnlBgColor}`}>
          <div className="text-xs text-slate-500">P/L</div>
          <div className={`text-lg font-bold ${pnlColor}`}>
            {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(0)}
            <span className="text-sm ml-1">
              ({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(1)}%)
            </span>
          </div>
        </div>

        {/* Exit Signal Banner */}
        {hasExitSignal && exitSignal && (
          <div className={`mt-2 p-2 rounded border ${getUrgencyColors(exitSignal.urgency)}`}>
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm">EXIT SIGNAL</span>
              <span className="text-xs uppercase">{exitSignal.urgency} urgency</span>
            </div>
            <div className="text-sm mt-1">{exitSignal.reason}</div>
          </div>
        )}

        {/* DTE and Delta */}
        {(position.status === 'open' || hasExitSignal) && (
          <div className="flex justify-between text-xs text-slate-500 mt-2">
            <span>{position.dte !== null ? `${position.dte} DTE` : ''}</span>
            <span>{position.delta !== null ? `Delta: ${position.delta.toFixed(2)}` : ''}</span>
          </div>
        )}

        {/* Close Button - for open positions */}
        {position.status === 'open' && (
          <button
            onClick={() => {
              if (position.currentPrice) {
                setClosePrice(position.currentPrice.toFixed(2));
              }
              setShowCloseModal(true);
            }}
            className="w-full mt-2 py-1.5 bg-slate-200 hover:bg-slate-300 rounded text-sm font-medium text-slate-700"
          >
            Close Position
          </button>
        )}

        {/* Exit Signal Actions */}
        {hasExitSignal && (
          <div className="flex gap-2 mt-2">
            <button
              onClick={() => {
                if (position.currentPrice) {
                  setClosePrice(position.currentPrice.toFixed(2));
                }
                setShowCloseModal(true);
              }}
              className="flex-1 py-1.5 bg-red-600 hover:bg-red-700 rounded text-sm font-bold text-white"
            >
              Close Now
            </button>
            <button
              onClick={handleDismissSignal}
              disabled={dismissing}
              className="flex-1 py-1.5 bg-slate-200 hover:bg-slate-300 rounded text-sm font-medium text-slate-700 disabled:opacity-50"
            >
              {dismissing ? 'Dismissing...' : 'Dismiss'}
            </button>
          </div>
        )}

        {/* Closed Status */}
        {position.status === 'closed' && (
          <div className="mt-2 text-center text-xs text-slate-500">
            Closed at ${position.closePrice?.toFixed(2)}
            {position.exitReason && ` • ${position.exitReason}`}
          </div>
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
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Close Price
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={closePrice}
                  onChange={(e) => setClosePrice(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500"
                  placeholder={position.currentPrice?.toFixed(2) || '0.00'}
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
                className="flex-1 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
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

export function PositionsPanel() {
  const positions = useOptionsStore((state) => state.positions);
  const exitSignals = useOptionsStore((state) => state.exitSignals);
  const [showClosed, setShowClosed] = useState(false);

  // Positions with exit signals + open positions (exit_signal positions are still active)
  const activePositions = positions.filter(p => p.status === 'open' || p.status === 'exit_signal');
  const closedPositions = positions.filter(p => p.status === 'closed');

  // Create a map of positionId -> ExitSignal for quick lookup
  const exitSignalMap = new Map(exitSignals.map(s => [s.positionId, s]));

  // Calculate totals
  const totalPnl = positions.reduce((sum, p) => sum + p.pnl, 0);
  const totalExposure = activePositions.reduce((sum, p) => sum + p.entryCost, 0);

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-green-600 text-white">
        <h2 className="font-bold text-lg">Positions</h2>
        <p className="text-sm text-green-200">
          {activePositions.length} active • {closedPositions.length} closed
        </p>
      </div>

      <div className="p-4 space-y-3">
        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-50 rounded-lg p-3 text-center">
            <div className="text-xs text-slate-500">Total P/L</div>
            <div className={`text-lg font-bold ${totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}
            </div>
          </div>
          <div className="bg-slate-50 rounded-lg p-3 text-center">
            <div className="text-xs text-slate-500">Exposure</div>
            <div className="text-lg font-bold text-slate-700">
              ${totalExposure.toFixed(0)}
            </div>
          </div>
        </div>

        {/* Active Positions (open + exit_signal) */}
        {activePositions.length > 0 ? (
          <div className="space-y-2">
            {activePositions.map((pos) => (
              <PositionCard
                key={pos.id}
                position={pos}
                exitSignal={exitSignalMap.get(pos.id)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-6 text-slate-500">
            <p className="text-lg mb-1">No open positions</p>
            <p className="text-sm">Take a trade from the Signals tab</p>
          </div>
        )}

        {/* Closed Positions Toggle */}
        {closedPositions.length > 0 && (
          <>
            <button
              onClick={() => setShowClosed(!showClosed)}
              className="w-full text-left text-sm text-slate-600 hover:text-slate-800 flex items-center gap-1 py-2 border-t"
            >
              <span className={`transform transition-transform ${showClosed ? 'rotate-90' : ''}`}>
                ▶
              </span>
              {showClosed ? 'Hide' : 'Show'} closed positions ({closedPositions.length})
            </button>

            {showClosed && (
              <div className="space-y-2">
                {closedPositions.map((pos) => (
                  <PositionCard key={pos.id} position={pos} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
