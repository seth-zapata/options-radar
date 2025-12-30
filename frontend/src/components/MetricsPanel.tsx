/**
 * Metrics Dashboard Panel
 * Displays signal accuracy, confidence calibration, symbol breakdown, and P/L tracking.
 */

import { useState, useEffect } from 'react';

interface SymbolStats {
  symbol: string;
  totalSignals: number;
  withOutcomes: number;
  wins: number;
  losses: number;
  accuracy: number | null;
  totalPnl: number;
}

interface CalibrationBucket {
  bucket: string;
  signals: number;
  wins: number;
  expectedRate: number;
  actualRate: number | null;
}

interface AlignmentStat {
  type: string;
  signals: number;
  withOutcomes: number;
  wins: number;
  losses: number;
  accuracy: number | null;
  pnl: number;
}

interface PositionSummary {
  openPositions: number;
  closedPositions: number;
  unrealizedPnl: number;
  realizedPnl: number;
  totalPnl: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number | null;
}

interface DashboardData {
  generatedAt: string;
  overall: {
    volume: { totalDecisions: number; recommendations: number; abstains: number };
    rates: { recommendationsPerHour: number; abstentionRate: number };
    outcomes: { recorded: number; wins: number; losses: number; winRate: number };
    profitability: { totalTheoreticalPnl: number; profitFactor: number };
    confidenceCalibration: { avgConfidence: number; highConfidenceWinRate: number; lowConfidenceWinRate: number };
  };
  bySymbol: SymbolStats[];
  confidenceCalibration: CalibrationBucket[];
  alignmentAnalysis: AlignmentStat[];
  positionSummary: PositionSummary;
  config: { signalsEnabled: string[]; signalsDisabled: string[] };
}

export function MetricsPanel() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboard();
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboard, 30000);
    return () => clearInterval(interval);
  }, []);

  async function fetchDashboard() {
    try {
      const response = await fetch('/api/metrics/dashboard');
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse flex space-x-4">
          <div className="flex-1 space-y-4 py-1">
            <div className="h-4 bg-slate-200 rounded w-3/4"></div>
            <div className="space-y-2">
              <div className="h-4 bg-slate-200 rounded"></div>
              <div className="h-4 bg-slate-200 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-red-500">Error: {error}</div>
        <button
          onClick={fetchDashboard}
          className="mt-2 text-sm text-indigo-600 hover:text-indigo-800"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const { overall, bySymbol, confidenceCalibration, alignmentAnalysis, positionSummary, config } = data;

  return (
    <div className="space-y-4">
      {/* Note about what this tracks */}
      <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-600 border border-slate-200">
        <span className="font-medium">Tracking:</span> All confirmed trades from both signal systems. Position P/L updates in real-time.
      </div>

      {/* Overall Stats */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Overall Performance</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500 text-xs">Total Signals</div>
            <div className="text-lg font-semibold">{overall.volume.recommendations}</div>
          </div>
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500 text-xs">Win Rate</div>
            <div className={`text-lg font-semibold ${overall.outcomes.winRate > 50 ? 'text-green-600' : 'text-red-600'}`}>
              {overall.outcomes.winRate}%
            </div>
          </div>
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500 text-xs">Avg Confidence</div>
            <div className="text-lg font-semibold">{overall.confidenceCalibration.avgConfidence}%</div>
          </div>
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500 text-xs">Profit Factor</div>
            <div className={`text-lg font-semibold ${overall.profitability.profitFactor > 1 ? 'text-green-600' : 'text-red-600'}`}>
              {overall.profitability.profitFactor}x
            </div>
          </div>
        </div>
      </div>

      {/* P/L Summary */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Position P/L</h3>
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="text-center">
            <div className="text-slate-500 text-xs">Open</div>
            <div className={`font-medium ${positionSummary.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${positionSummary.unrealizedPnl.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-slate-500 text-xs">Realized</div>
            <div className={`font-medium ${positionSummary.realizedPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${positionSummary.realizedPnl.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-slate-500 text-xs">Total</div>
            <div className={`font-semibold ${positionSummary.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${positionSummary.totalPnl.toFixed(2)}
            </div>
          </div>
        </div>
        <div className="mt-2 text-xs text-slate-500 text-center">
          {positionSummary.closedPositions} trades | {positionSummary.winRate !== null ? `${positionSummary.winRate}% win rate` : 'No closed trades'}
        </div>
      </div>

      {/* Symbol Breakdown */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">By Symbol</h3>
        {bySymbol.length === 0 ? (
          <div className="text-slate-400 text-sm text-center py-2">No signal data yet</div>
        ) : (
          <div className="space-y-1">
            {bySymbol.slice(0, 8).map((s) => (
              <div key={s.symbol} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    config.signalsDisabled.includes(s.symbol) ? 'bg-slate-300' : 'bg-green-500'
                  }`}></span>
                  <span className="font-medium">{s.symbol}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-slate-500">{s.totalSignals} signals</span>
                  {s.accuracy !== null ? (
                    <span className={`font-medium ${s.accuracy > 50 ? 'text-green-600' : 'text-red-600'}`}>
                      {s.accuracy}%
                    </span>
                  ) : (
                    <span className="text-slate-400">-</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Confidence Calibration */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Confidence Calibration</h3>
        <div className="space-y-1">
          {confidenceCalibration.map((bucket) => (
            <div key={bucket.bucket} className="flex items-center justify-between text-sm">
              <span className="text-slate-600 w-16">{bucket.bucket}%</span>
              <div className="flex-1 mx-2 h-2 bg-slate-100 rounded overflow-hidden">
                {bucket.actualRate !== null && (
                  <div
                    className={`h-full ${bucket.actualRate >= bucket.expectedRate ? 'bg-green-500' : 'bg-amber-500'}`}
                    style={{ width: `${Math.min(bucket.actualRate, 100)}%` }}
                  ></div>
                )}
              </div>
              <span className="w-20 text-right">
                {bucket.actualRate !== null ? (
                  <span className={bucket.actualRate >= bucket.expectedRate ? 'text-green-600' : 'text-amber-600'}>
                    {bucket.actualRate}% ({bucket.signals})
                  </span>
                ) : (
                  <span className="text-slate-400">- ({bucket.signals})</span>
                )}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-2 text-xs text-slate-500">
          Expected: confidence % = win rate %
        </div>
      </div>

      {/* Alignment Analysis */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">News Alignment Impact</h3>
        {alignmentAnalysis.every(a => a.signals === 0) ? (
          <div className="text-slate-400 text-sm text-center py-2">No alignment data yet</div>
        ) : (
          <div className="space-y-2">
            {alignmentAnalysis.map((a) => (
              <div key={a.type} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    a.type === 'aligned' ? 'bg-green-500' :
                    a.type === 'notAligned' ? 'bg-red-500' : 'bg-slate-400'
                  }`}></span>
                  <span className="capitalize">
                    {a.type === 'aligned' ? 'Aligned (News + WSB)' :
                     a.type === 'notAligned' ? 'Conflicting' : 'WSB Only'}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-slate-500">{a.signals}</span>
                  {a.accuracy !== null ? (
                    <span className={`font-medium ${a.accuracy > 50 ? 'text-green-600' : 'text-red-600'}`}>
                      {a.accuracy}%
                    </span>
                  ) : (
                    <span className="text-slate-400">-</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        <div className="mt-2 text-xs text-slate-500">
          Comparing aligned (News + WSB agree) vs conflicting sentiment signals
        </div>
      </div>

      {/* Config */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Signal Configuration</h3>
        <div className="text-xs space-y-1">
          <div>
            <span className="text-green-600 font-medium">Enabled:</span>{' '}
            <span className="text-slate-600">{config.signalsEnabled.join(', ')}</span>
          </div>
          <div>
            <span className="text-slate-400 font-medium">Disabled:</span>{' '}
            <span className="text-slate-500">{config.signalsDisabled.join(', ')}</span>
          </div>
        </div>
      </div>

      {/* Refresh indicator */}
      <div className="text-center text-xs text-slate-400">
        Last updated: {new Date(data.generatedAt).toLocaleTimeString()}
      </div>
    </div>
  );
}
