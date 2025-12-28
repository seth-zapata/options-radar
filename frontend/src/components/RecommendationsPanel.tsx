/**
 * Panel displaying trade recommendations with collapsible gate details.
 * Primary display in the right panel.
 */

import { useState } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { Recommendation, RecommendationGateResult } from '../types';

const API_BASE = 'http://localhost:8000';

function formatAction(action: string): string {
  switch (action) {
    case 'BUY_CALL':
      return 'Buy Call';
    case 'BUY_PUT':
      return 'Buy Put';
    case 'SELL_CALL':
      return 'Sell Call';
    case 'SELL_PUT':
      return 'Sell Put';
    default:
      return action;
  }
}

function formatTimeAgo(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const seconds = Math.max(0, Math.floor((now.getTime() - date.getTime()) / 1000));

  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function isExpired(validUntil: string): boolean {
  return new Date(validUntil) < new Date();
}

function GateResultRow({ gate }: { gate: RecommendationGateResult }) {
  return (
    <div className="flex items-center justify-between py-1 text-xs">
      <div className="flex items-center gap-2">
        <span className={gate.passed ? 'text-green-600' : 'text-red-600'}>
          {gate.passed ? '✓' : '✗'}
        </span>
        <span className="text-slate-600">{gate.name.replace(/_/g, ' ')}</span>
      </div>
      <span className="text-slate-500 text-right max-w-[50%] truncate">
        {gate.message}
      </span>
    </div>
  );
}

interface FillPriceModalProps {
  rec: Recommendation;
  onClose: () => void;
  onConfirm: (fillPrice: number, contracts: number) => void;
}

function FillPriceModal({ rec, onClose, onConfirm }: FillPriceModalProps) {
  const [fillPrice, setFillPrice] = useState(rec.premium.toFixed(2));
  const [contracts, setContracts] = useState(rec.contracts.toString());
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

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="px-4 py-3 border-b">
          <h3 className="font-bold text-lg">Confirm Trade</h3>
          <p className="text-sm text-slate-500">
            {formatAction(rec.action)} {rec.underlying} ${rec.strike} {rec.right === 'C' ? 'Call' : 'Put'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-4">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Fill Price (per contract)
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={fillPrice}
                  onChange={(e) => setFillPrice(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder={rec.premium.toFixed(2)}
                  autoFocus
                />
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Displayed: ${rec.premium.toFixed(2)} (adjust to actual fill)
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Contracts
              </label>
              <input
                type="number"
                min="1"
                value={contracts}
                onChange={(e) => setContracts(e.target.value)}
                className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>

            <div className="bg-slate-50 rounded p-3">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Total Cost</span>
                <span className="font-bold">
                  ${((parseFloat(fillPrice) || 0) * (parseInt(contracts, 10) || 0) * 100).toFixed(0)}
                </span>
              </div>
            </div>

            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
          </div>

          <div className="flex gap-3 mt-6">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-slate-300 rounded-md text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
            >
              Confirm Trade
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function RecommendationCard({ rec, isLatest, isConfirmed, onDismiss }: { rec: Recommendation; isLatest: boolean; isConfirmed: boolean; onDismiss: () => void }) {
  const [showGates, setShowGates] = useState(false);
  const [showFillModal, setShowFillModal] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const expired = isExpired(rec.validUntil);

  const handleConfirmTrade = async (fillPrice: number, contracts: number) => {
    setConfirming(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recommendation_id: rec.id,
          fill_price: fillPrice,
          contracts: contracts,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to confirm trade');
      }

      setShowFillModal(false);
    } catch (error) {
      console.error('Error confirming trade:', error);
      alert(error instanceof Error ? error.message : 'Failed to confirm trade');
    } finally {
      setConfirming(false);
    }
  };

  const actionColors: Record<string, string> = {
    BUY_CALL: 'bg-green-500 text-white',
    BUY_PUT: 'bg-red-500 text-white',
    SELL_CALL: 'bg-orange-500 text-white',
    SELL_PUT: 'bg-blue-500 text-white',
  };

  const confidenceColor =
    rec.confidence >= 80
      ? 'text-green-600'
      : rec.confidence >= 60
        ? 'text-yellow-600'
        : 'text-red-600';

  // Determine card styling based on state
  const cardStyles = isConfirmed
    ? 'bg-green-50 ring-2 ring-green-500 border-green-300'  // Confirmed: green highlight
    : expired
      ? 'opacity-50 bg-slate-50'  // Expired: dimmed
      : isLatest
        ? 'bg-white ring-2 ring-indigo-300'  // Latest: indigo ring
        : 'bg-white';  // Default

  return (
    <div className={`border rounded-lg overflow-hidden ${cardStyles}`}>
      {/* Header */}
      <div className={`px-3 py-2 ${actionColors[rec.action] || 'bg-slate-500 text-white'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-bold">{formatAction(rec.action)}</span>
            {isConfirmed && (
              <span className="px-1.5 py-0.5 bg-white/20 rounded text-xs font-medium">
                CONFIRMED
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm opacity-90">
              {formatTimeAgo(rec.generatedAt)}
              {expired && ' (expired)'}
            </span>
            {(expired || isConfirmed) && (
              <button
                onClick={onDismiss}
                className="text-white/70 hover:text-white text-sm"
                title="Dismiss"
              >
                ×
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-3">
        {/* Option details */}
        <div className="font-mono text-lg mb-2">
          <span className="font-bold">{rec.underlying}</span>{' '}
          <span className="text-slate-600">{rec.expiry}</span>{' '}
          <span className="font-bold">${rec.strike}</span>{' '}
          <span className="text-slate-600">{rec.right === 'C' ? 'Call' : 'Put'}</span>
        </div>

        {/* Trade details grid */}
        <div className="grid grid-cols-3 gap-2 text-sm mb-3">
          <div className="bg-slate-50 rounded p-2 text-center">
            <div className="text-slate-500 text-xs">Premium</div>
            <div className="font-bold">${rec.premium.toFixed(2)}</div>
          </div>
          <div className="bg-slate-50 rounded p-2 text-center">
            <div className="text-slate-500 text-xs">Contracts</div>
            <div className="font-bold">{rec.contracts}</div>
          </div>
          <div className="bg-slate-50 rounded p-2 text-center">
            <div className="text-slate-500 text-xs">Total</div>
            <div className="font-bold">${rec.totalCost.toFixed(0)}</div>
          </div>
        </div>

        {/* Confidence */}
        <div className="flex items-center justify-between text-sm mb-2">
          <span className={`font-medium ${confidenceColor}`}>
            Confidence: {rec.confidence}%
          </span>
        </div>

        {/* Rationale */}
        <p className="text-sm text-slate-600 italic mb-3">{rec.rationale}</p>

        {/* Collapsible gate details */}
        <button
          onClick={() => setShowGates(!showGates)}
          className="w-full text-left text-sm text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
        >
          <span className={`transform transition-transform ${showGates ? 'rotate-90' : ''}`}>
            ▶
          </span>
          {showGates ? 'Hide' : 'Show'} gate details ({rec.gateResults?.length || 0} gates)
        </button>

        {showGates && rec.gateResults && (
          <div className="mt-2 pt-2 border-t space-y-1">
            {rec.gateResults.map((gate) => (
              <GateResultRow key={gate.name} gate={gate} />
            ))}
          </div>
        )}

        {/* Take Trade Button */}
        {!expired && !isConfirmed && (
          <button
            onClick={() => setShowFillModal(true)}
            disabled={confirming}
            className="w-full mt-3 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 font-medium text-sm"
          >
            {confirming ? 'Confirming...' : 'I took this trade'}
          </button>
        )}

        {isConfirmed && (
          <div className="mt-3 py-2 bg-green-100 text-green-800 rounded-md text-center text-sm font-medium">
            Trade confirmed
          </div>
        )}
      </div>

      {/* Fill Price Modal */}
      {showFillModal && (
        <FillPriceModal
          rec={rec}
          onClose={() => setShowFillModal(false)}
          onConfirm={handleConfirmTrade}
        />
      )}
    </div>
  );
}

function SessionStatusBar() {
  const sessionStatus = useOptionsStore((state) => state.sessionStatus);

  if (!sessionStatus) {
    return (
      <div className="bg-slate-100 rounded-lg p-3 text-slate-500 text-sm">
        Loading session data...
      </div>
    );
  }

  const exposureBarWidth = Math.min(100, sessionStatus.exposurePercent);

  return (
    <div className="bg-slate-100 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">Session Exposure</span>
        <span className="text-xs text-slate-500">
          {sessionStatus.recommendationCount} signal{sessionStatus.recommendationCount !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Exposure bar */}
      <div className="h-2 bg-slate-200 rounded-full overflow-hidden mb-1">
        <div
          className={`h-full transition-all ${
            sessionStatus.isAtLimit
              ? 'bg-red-500'
              : sessionStatus.isWarning
                ? 'bg-yellow-500'
                : 'bg-green-500'
          }`}
          style={{ width: `${exposureBarWidth}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-slate-600">
        <span>
          ${sessionStatus.totalExposure.toFixed(0)} / $
          {(sessionStatus.totalExposure + sessionStatus.exposureRemaining).toFixed(0)}
        </span>
        <span
          className={
            sessionStatus.isAtLimit
              ? 'text-red-600 font-medium'
              : sessionStatus.isWarning
                ? 'text-yellow-600'
                : ''
          }
        >
          {sessionStatus.exposurePercent.toFixed(0)}%
        </span>
      </div>

      {sessionStatus.isAtLimit && (
        <p className="text-xs text-red-600 mt-1">
          Session limit reached. No more signals.
        </p>
      )}
    </div>
  );
}

export function RecommendationsPanel() {
  const recommendations = useOptionsStore((state) => state.recommendations);
  const abstain = useOptionsStore((state) => state.abstain);
  const positions = useOptionsStore((state) => state.positions);
  const dismissRecommendation = useOptionsStore((state) => state.dismissRecommendation);
  const clearExpiredRecommendations = useOptionsStore((state) => state.clearExpiredRecommendations);

  // Get set of confirmed recommendation IDs
  const confirmedRecIds = new Set(positions.map((p) => p.recommendationId));

  // Count expired recommendations
  const expiredCount = recommendations.filter(r => isExpired(r.validUntil)).length;

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-indigo-600 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-bold text-lg">Trade Signals</h2>
            <p className="text-sm text-indigo-200">
              {recommendations.length > 0
                ? `${recommendations.length} signal${recommendations.length !== 1 ? 's' : ''} generated`
                : abstain
                  ? 'Waiting for gates to pass'
                  : 'Monitoring for opportunities'}
            </p>
          </div>
          {expiredCount > 0 && (
            <button
              onClick={clearExpiredRecommendations}
              className="px-2 py-1 text-xs bg-indigo-500 hover:bg-indigo-400 rounded"
              title="Clear all expired signals"
            >
              Clear {expiredCount} expired
            </button>
          )}
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Session Status - compact at top */}
        <SessionStatusBar />

        {/* Recommendations List */}
        {recommendations.length === 0 ? (
          <div className="text-center py-6 text-slate-500">
            <p className="text-lg mb-1">No signals yet</p>
            <p className="text-sm">
              Signals appear when all gates pass
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {recommendations.map((rec, index) => (
              <RecommendationCard
                key={rec.id}
                rec={rec}
                isLatest={index === 0}
                isConfirmed={confirmedRecIds.has(rec.id)}
                onDismiss={() => dismissRecommendation(rec.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
