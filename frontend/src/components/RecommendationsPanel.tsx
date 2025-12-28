/**
 * Panel displaying trade recommendations with collapsible gate details.
 * Primary display in the right panel.
 */

import { useState } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { Recommendation, RecommendationGateResult } from '../types';

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
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

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

function RecommendationCard({ rec, isLatest }: { rec: Recommendation; isLatest: boolean }) {
  const [showGates, setShowGates] = useState(false);
  const expired = isExpired(rec.validUntil);

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

  return (
    <div
      className={`border rounded-lg overflow-hidden ${
        expired ? 'opacity-50 bg-slate-50' : isLatest ? 'bg-white ring-2 ring-indigo-300' : 'bg-white'
      }`}
    >
      {/* Header */}
      <div className={`px-3 py-2 ${actionColors[rec.action] || 'bg-slate-500 text-white'}`}>
        <div className="flex items-center justify-between">
          <span className="font-bold">{formatAction(rec.action)}</span>
          <span className="text-sm opacity-90">
            {formatTimeAgo(rec.generatedAt)}
            {expired && ' (expired)'}
          </span>
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
      </div>
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

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-indigo-600 text-white">
        <h2 className="font-bold text-lg">Trade Signals</h2>
        <p className="text-sm text-indigo-200">
          {recommendations.length > 0
            ? `${recommendations.length} signal${recommendations.length !== 1 ? 's' : ''} generated`
            : abstain
              ? 'Waiting for gates to pass'
              : 'Monitoring for opportunities'}
        </p>
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
              <RecommendationCard key={rec.id} rec={rec} isLatest={index === 0} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
