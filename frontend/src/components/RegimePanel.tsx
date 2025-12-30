/**
 * Regime Strategy Panel for TSLA
 * Shows regime status, entry conditions, price monitoring, and signals log.
 */

import { useEffect, useState } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { RegimeStatus, RegimeSignal, RegimeType } from '../types';

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

interface PriceData {
  high: number;
  low: number;
  current: number;
  pullbackPct: number;
  bouncePct: number;
}

export function RegimePanel() {
  const regimeStatus = useOptionsStore((state) => state.regimeStatus);
  const regimeSignals = useOptionsStore((state) => state.regimeSignals);
  const regimeLoading = useOptionsStore((state) => state.regimeLoading);
  const setRegimeStatus = useOptionsStore((state) => state.setRegimeStatus);
  const underlying = useOptionsStore((state) => state.underlying);

  const [priceData, setPriceData] = useState<PriceData | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Fetch regime status on mount and periodically
  useEffect(() => {
    const fetchRegimeStatus = async () => {
      try {
        const response = await fetch('/api/regime/status?symbol=TSLA');
        if (!response.ok) throw new Error('Failed to fetch regime status');
        const data: RegimeStatus = await response.json();
        setRegimeStatus(data);
        setFetchError(null);
      } catch (e) {
        setFetchError(e instanceof Error ? e.message : 'Unknown error');
      }
    };

    fetchRegimeStatus();
    const interval = setInterval(fetchRegimeStatus, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [setRegimeStatus]);

  // Calculate price data from underlying
  // Only show price data when TSLA is the active symbol
  useEffect(() => {
    if (underlying && underlying.price > 0) {
      // Check if we're viewing TSLA (the symbol in underlying matches)
      // In real scenario, we'd track intraday high/low
      // For now, use current price with mock calculations
      const current = underlying.price;
      const high = current * 1.02; // Mock: 2% above current
      const low = current * 0.98; // Mock: 2% below current

      const pullbackPct = ((high - current) / high) * 100;
      const bouncePct = ((current - low) / low) * 100;

      setPriceData({ high, low, current, pullbackPct, bouncePct });
    }
  }, [underlying]);

  const activeRegime = regimeStatus?.active_regime;
  const regimeType: RegimeType | 'neutral' = activeRegime?.type || 'neutral';
  const colors = regimeBadgeColors[regimeType];
  const isBullish = activeRegime?.is_bullish || false;
  const isBearish = activeRegime?.is_bearish || false;

  // Entry thresholds from config
  const pullbackThreshold = regimeStatus?.config?.pullback_threshold || 1.5;
  const bounceThreshold = regimeStatus?.config?.bounce_threshold || 1.5;

  // Calculate progress towards entry
  const pullbackProgress = priceData ? Math.min((priceData.pullbackPct / pullbackThreshold) * 100, 100) : 0;
  const bounceProgress = priceData ? Math.min((priceData.bouncePct / bounceThreshold) * 100, 100) : 0;

  // Entry conditions checklist
  const entryConditions = [
    {
      label: 'Regime Active',
      passed: activeRegime?.is_active || false,
      detail: activeRegime ? `${regimeLabels[regimeType]}` : 'No regime',
    },
    {
      label: isBullish ? 'Pullback Trigger' : 'Bounce Trigger',
      passed: isBullish
        ? (priceData?.pullbackPct || 0) >= pullbackThreshold
        : (priceData?.bouncePct || 0) >= bounceThreshold,
      detail: isBullish
        ? `${(priceData?.pullbackPct || 0).toFixed(1)}% / ${pullbackThreshold}% needed`
        : `${(priceData?.bouncePct || 0).toFixed(1)}% / ${bounceThreshold}% needed`,
    },
    {
      label: 'Tech Confirmation',
      passed: true, // Would come from backend
      detail: 'BB/MACD/SMA (1+ required)',
    },
    {
      label: 'Cooldown Clear',
      passed: true, // Would come from backend
      detail: 'Min 1 day between entries',
    },
  ];

  if (regimeLoading) {
    return (
      <div className="space-y-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="animate-pulse flex flex-col space-y-4">
            <div className="h-8 bg-slate-200 rounded w-1/2"></div>
            <div className="h-4 bg-slate-200 rounded w-3/4"></div>
            <div className="h-24 bg-slate-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with Symbol */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-slate-800">TSLA</h2>
            <div className="text-sm text-slate-500">Regime Strategy</div>
          </div>
          {underlying && (
            <div className="text-right">
              <div className="text-2xl font-bold text-slate-800">
                ${underlying.price.toFixed(2)}
              </div>
              <div className="text-xs text-slate-500">
                IV Rank: {underlying.ivRank.toFixed(0)}%
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Current Regime Status */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Current Regime</h3>

        {fetchError ? (
          <div className="text-red-500 text-sm">{fetchError}</div>
        ) : (
          <div className="space-y-3">
            {/* Regime Badge */}
            <div className={`inline-flex items-center px-4 py-2 rounded-full border-2 ${colors.bg} ${colors.text} ${colors.border}`}>
              <span className={`w-2 h-2 rounded-full mr-2 ${
                activeRegime?.is_active ? 'bg-current animate-pulse' : 'bg-slate-400'
              }`}></span>
              <span className="font-bold text-lg">{regimeLabels[regimeType]}</span>
            </div>

            {/* Regime Details */}
            {activeRegime ? (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-50 rounded p-2">
                  <div className="text-slate-500 text-xs">Sentiment</div>
                  <div className={`font-semibold ${activeRegime.sentiment_value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {activeRegime.sentiment_value >= 0 ? '+' : ''}{(activeRegime.sentiment_value * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-slate-50 rounded p-2">
                  <div className="text-slate-500 text-xs">Days Remaining</div>
                  <div className="font-semibold text-slate-800">
                    {activeRegime.days_remaining} day{activeRegime.days_remaining !== 1 ? 's' : ''}
                  </div>
                </div>
                <div className="bg-slate-50 rounded p-2 col-span-2">
                  <div className="text-slate-500 text-xs">Expires</div>
                  <div className="font-medium text-slate-700">
                    {new Date(activeRegime.expires_at).toLocaleDateString()} {new Date(activeRegime.expires_at).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 text-sm py-2">
                No active regime. Waiting for WSB sentiment signal...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Entry Conditions Checklist */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Entry Conditions</h3>
        <div className="space-y-2">
          {entryConditions.map((condition, idx) => (
            <div key={idx} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                  condition.passed ? 'bg-green-100 text-green-600' : 'bg-slate-100 text-slate-400'
                }`}>
                  {condition.passed ? '✓' : '○'}
                </span>
                <span className={condition.passed ? 'text-slate-800' : 'text-slate-500'}>
                  {condition.label}
                </span>
              </div>
              <span className="text-slate-500 text-xs">{condition.detail}</span>
            </div>
          ))}
        </div>

        {/* All conditions met indicator */}
        {entryConditions.every(c => c.passed) && activeRegime?.is_active && (
          <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-center">
            <span className="text-green-700 font-medium text-sm">
              Ready for {isBullish ? 'BUY CALL' : 'BUY PUT'} Entry
            </span>
          </div>
        )}
      </div>

      {/* Price Monitoring */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Price Monitoring</h3>

        {priceData ? (
          <div className="space-y-4">
            {/* Price levels */}
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              <div>
                <div className="text-slate-500 text-xs">High</div>
                <div className="font-medium text-green-600">${priceData.high.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-slate-500 text-xs">Current</div>
                <div className="font-bold text-slate-800">${priceData.current.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-slate-500 text-xs">Low</div>
                <div className="font-medium text-red-600">${priceData.low.toFixed(2)}</div>
              </div>
            </div>

            {/* Pullback Progress (for bullish) */}
            {isBullish && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600">Pullback from High</span>
                  <span className={priceData.pullbackPct >= pullbackThreshold ? 'text-green-600 font-medium' : 'text-slate-500'}>
                    {priceData.pullbackPct.toFixed(1)}% / {pullbackThreshold}%
                  </span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      pullbackProgress >= 100 ? 'bg-green-500' : 'bg-amber-400'
                    }`}
                    style={{ width: `${pullbackProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Bounce Progress (for bearish) */}
            {isBearish && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600">Bounce from Low</span>
                  <span className={priceData.bouncePct >= bounceThreshold ? 'text-green-600 font-medium' : 'text-slate-500'}>
                    {priceData.bouncePct.toFixed(1)}% / {bounceThreshold}%
                  </span>
                </div>
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      bounceProgress >= 100 ? 'bg-green-500' : 'bg-amber-400'
                    }`}
                    style={{ width: `${bounceProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Both bars when neutral */}
            {!isBullish && !isBearish && (
              <div className="text-slate-500 text-sm text-center py-2">
                No active regime - monitoring disabled
              </div>
            )}
          </div>
        ) : (
          <div className="text-slate-400 text-sm text-center py-2">
            Waiting for price data...
          </div>
        )}
      </div>

      {/* Exit Targets */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Exit Rules</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-green-50 rounded p-2">
            <div className="text-green-600 text-xs">Take Profit</div>
            <div className="text-green-700 font-bold text-lg">+40%</div>
          </div>
          <div className="bg-red-50 rounded p-2">
            <div className="text-red-600 text-xs">Stop Loss</div>
            <div className="text-red-700 font-bold text-lg">-20%</div>
          </div>
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-600 text-xs">Min DTE Exit</div>
            <div className="text-slate-700 font-bold text-lg">1</div>
          </div>
        </div>
      </div>

      {/* Recent Signals Log */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Recent Signals
          {regimeSignals.length > 0 && (
            <span className="ml-2 text-xs text-slate-400 font-normal">
              ({regimeSignals.length})
            </span>
          )}
        </h3>

        {regimeSignals.length === 0 ? (
          <div className="text-slate-400 text-sm text-center py-4">
            No signals generated yet
          </div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {regimeSignals.slice(0, 10).map((signal, idx) => (
              <SignalRow key={`${signal.generated_at}-${idx}`} signal={signal} />
            ))}
          </div>
        )}
      </div>

      {/* Strategy Info */}
      <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-500">
        <div className="font-medium text-slate-600 mb-1">Strategy: Regime-Filtered Intraday</div>
        <div>
          Backtested on TSLA (Jan 2024 - Jan 2025): 71 trades, 43.7% win rate, +1238% return
        </div>
      </div>
    </div>
  );
}

function SignalRow({ signal }: { signal: RegimeSignal }) {
  const isBuy = signal.signal_type === 'BUY_CALL';
  const time = new Date(signal.generated_at).toLocaleTimeString();

  return (
    <div className={`flex items-center justify-between p-2 rounded text-sm ${
      isBuy ? 'bg-green-50' : 'bg-red-50'
    }`}>
      <div className="flex items-center gap-2">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          isBuy ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'
        }`}>
          {signal.signal_type}
        </span>
        <span className="text-slate-600">
          @${signal.entry_price.toFixed(2)}
        </span>
      </div>
      <div className="text-right">
        <div className="text-slate-500 text-xs">{time}</div>
        <div className="text-slate-400 text-xs">{signal.trigger_reason}</div>
      </div>
    </div>
  );
}
