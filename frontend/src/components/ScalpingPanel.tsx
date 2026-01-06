/**
 * Scalping Panel for 0DTE/1DTE intraday momentum trading.
 * Shows scalp signals, active positions, and closed trades.
 */

import { useOptionsStore } from '../store/optionsStore';
import type { ScalpSignal, ScalpPosition, ScalpExitResult, ScalpSignalType } from '../types';

// Signal type badge colors
const signalBadgeColors: Record<ScalpSignalType, { bg: string; text: string }> = {
  SCALP_CALL: { bg: 'bg-green-100', text: 'text-green-800' },
  SCALP_PUT: { bg: 'bg-red-100', text: 'text-red-800' },
};

// Trigger labels
const triggerLabels: Record<string, string> = {
  momentum_burst: 'Momentum',
  vwap_bounce: 'VWAP Bounce',
  vwap_rejection: 'VWAP Reject',
  breakout: 'Breakout',
};

// Format time as HH:MM:SS
function formatTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour12: false });
  } catch {
    return '--:--:--';
  }
}

// Format hold time as M:SS
function formatHoldTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

interface SignalCardProps {
  signal: ScalpSignal;
}

function SignalCard({ signal }: SignalCardProps) {
  const colors = signalBadgeColors[signal.signal_type];
  const triggerLabel = triggerLabels[signal.trigger] || signal.trigger;

  return (
    <div className="border border-slate-200 rounded-lg p-3 bg-white shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors.bg} ${colors.text}`}>
          {signal.signal_type.replace('SCALP_', '')}
        </span>
        <span className="text-xs text-slate-500">{formatTime(signal.timestamp)}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-slate-500">Strike:</span>
          <span className="ml-1 font-medium">${signal.strike}</span>
        </div>
        <div>
          <span className="text-slate-500">DTE:</span>
          <span className="ml-1 font-medium">{signal.dte}</span>
        </div>
        <div>
          <span className="text-slate-500">Entry:</span>
          <span className="ml-1 font-medium">${signal.entry_price.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-slate-500">Spread:</span>
          <span className="ml-1 font-medium">{signal.spread_pct.toFixed(1)}%</span>
        </div>
      </div>

      <div className="mt-2 pt-2 border-t border-slate-100">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-500">
            {triggerLabel} | Vel: {signal.velocity_pct.toFixed(2)}%
          </span>
          <span className="font-medium text-blue-600">
            {signal.confidence}% conf
          </span>
        </div>
      </div>

      <div className="mt-2 flex gap-1">
        <span className="text-xs px-1.5 py-0.5 bg-green-50 text-green-700 rounded">
          TP: +{signal.take_profit_pct}%
        </span>
        <span className="text-xs px-1.5 py-0.5 bg-red-50 text-red-700 rounded">
          SL: -{signal.stop_loss_pct}%
        </span>
        <span className="text-xs px-1.5 py-0.5 bg-slate-50 text-slate-600 rounded">
          Max: {signal.max_hold_minutes}m
        </span>
      </div>
    </div>
  );
}

interface PositionCardProps {
  position: ScalpPosition;
}

function PositionCard({ position }: PositionCardProps) {
  const colors = signalBadgeColors[position.signal_type];
  const pnlColor = position.current_pnl_pct >= 0 ? 'text-green-600' : 'text-red-600';

  // Calculate progress towards TP/SL
  const tpProgress = Math.min((position.current_pnl_pct / position.take_profit_pct) * 100, 100);
  const slProgress = Math.min((Math.abs(position.current_pnl_pct) / position.stop_loss_pct) * 100, 100);
  const timeProgress = (position.hold_seconds / position.max_hold_seconds) * 100;

  return (
    <div className="border-2 border-blue-200 rounded-lg p-3 bg-blue-50/30 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors.bg} ${colors.text}`}>
          {position.signal_type.replace('SCALP_', '')} OPEN
        </span>
        <span className={`text-lg font-bold ${pnlColor}`}>
          {position.current_pnl_pct >= 0 ? '+' : ''}{position.current_pnl_pct.toFixed(1)}%
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-slate-500">Strike:</span>
          <span className="ml-1 font-medium">${position.strike}</span>
        </div>
        <div>
          <span className="text-slate-500">Entry:</span>
          <span className="ml-1 font-medium">${position.entry_price.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-slate-500">Current:</span>
          <span className={`ml-1 font-medium ${pnlColor}`}>
            ${position.current_price?.toFixed(2) || '--'}
          </span>
        </div>
        <div>
          <span className="text-slate-500">Hold:</span>
          <span className="ml-1 font-medium">{formatHoldTime(position.hold_seconds)}</span>
        </div>
      </div>

      {/* Progress bars */}
      <div className="mt-3 space-y-2">
        {/* Take Profit progress (only show when positive) */}
        {position.current_pnl_pct > 0 && (
          <div>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-green-600">To TP (+{position.take_profit_pct}%)</span>
              <span className="text-green-600">{Math.round(tpProgress)}%</span>
            </div>
            <div className="h-1.5 bg-green-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 rounded-full transition-all"
                style={{ width: `${tpProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Stop Loss progress (only show when negative) */}
        {position.current_pnl_pct < 0 && (
          <div>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-red-600">To SL (-{position.stop_loss_pct}%)</span>
              <span className="text-red-600">{Math.round(slProgress)}%</span>
            </div>
            <div className="h-1.5 bg-red-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500 rounded-full transition-all"
                style={{ width: `${slProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Time progress */}
        <div>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-slate-500">Time Limit</span>
            <span className="text-slate-500">{Math.round(timeProgress)}%</span>
          </div>
          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${timeProgress > 80 ? 'bg-orange-500' : 'bg-slate-400'}`}
              style={{ width: `${timeProgress}%` }}
            />
          </div>
        </div>
      </div>

      <div className="mt-2 pt-2 border-t border-blue-100 text-xs text-slate-500">
        Max gain: +{position.max_gain_pct.toFixed(1)}% | Max DD: {position.max_drawdown_pct.toFixed(1)}%
      </div>
    </div>
  );
}

interface ClosedTradeRowProps {
  trade: ScalpExitResult;
}

function ClosedTradeRow({ trade }: ClosedTradeRowProps) {
  const pnlColor = trade.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600';
  const pnlBg = trade.pnl_pct >= 0 ? 'bg-green-50' : 'bg-red-50';

  const exitLabels: Record<string, { label: string; color: string }> = {
    take_profit: { label: 'TP', color: 'text-green-600' },
    stop_loss: { label: 'SL', color: 'text-red-600' },
    time_exit: { label: 'TIME', color: 'text-orange-600' },
  };

  const exit = exitLabels[trade.exit_reason] || { label: trade.exit_reason, color: 'text-slate-600' };

  return (
    <div className={`flex items-center justify-between px-2 py-1.5 ${pnlBg} rounded text-sm`}>
      <div className="flex items-center gap-2">
        <span className="font-medium">${trade.entry_price.toFixed(2)}</span>
        <span className="text-slate-400">→</span>
        <span className={`font-medium ${pnlColor}`}>${trade.exit_price.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className={`font-medium ${pnlColor}`}>
          {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(1)}%
        </span>
        <span className={`text-xs font-medium ${exit.color}`}>{exit.label}</span>
        <span className="text-xs text-slate-400">{formatHoldTime(trade.hold_seconds)}</span>
      </div>
    </div>
  );
}

export function ScalpingPanel() {
  const scalpEnabled = useOptionsStore((state) => state.scalpEnabled);
  const scalpSignals = useOptionsStore((state) => state.scalpSignals);
  const scalpPositions = useOptionsStore((state) => state.scalpPositions);
  const scalpClosedTrades = useOptionsStore((state) => state.scalpClosedTrades);

  // Calculate session stats
  const totalTrades = scalpClosedTrades.length;
  const winningTrades = scalpClosedTrades.filter((t) => t.pnl_pct > 0).length;
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  const totalPnlPct = scalpClosedTrades.reduce((sum, t) => sum + t.pnl_pct, 0);
  const avgPnl = totalTrades > 0 ? totalPnlPct / totalTrades : 0;

  if (!scalpEnabled) {
    return (
      <div className="p-4 text-center text-slate-500">
        <p className="text-sm">Scalping module is not enabled.</p>
        <p className="text-xs mt-1">Set SCALPING_ENABLED=true in backend config to enable.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header with stats */}
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-slate-800">0DTE/1DTE Scalping</h3>
        {totalTrades > 0 && (
          <div className="flex items-center gap-4 text-sm">
            <span className="text-slate-500">
              {totalTrades} trades | {winRate.toFixed(0)}% win
            </span>
            <span className={totalPnlPct >= 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
              {totalPnlPct >= 0 ? '+' : ''}{totalPnlPct.toFixed(1)}% total
            </span>
          </div>
        )}
      </div>

      {/* Active Positions */}
      {scalpPositions.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">
            Active Positions ({scalpPositions.length})
          </h4>
          <div className="space-y-2">
            {scalpPositions.map((position) => (
              <PositionCard key={position.signal_id} position={position} />
            ))}
          </div>
        </div>
      )}

      {/* Pending Signals */}
      {scalpSignals.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">
            Signals ({scalpSignals.length})
          </h4>
          <div className="space-y-2">
            {scalpSignals.map((signal) => (
              <SignalCard key={signal.id} signal={signal} />
            ))}
          </div>
        </div>
      )}

      {/* Closed Trades */}
      {scalpClosedTrades.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-slate-500 uppercase mb-2">
            Recent Trades
          </h4>
          <div className="space-y-1">
            {scalpClosedTrades.slice(0, 10).map((trade) => (
              <ClosedTradeRow key={trade.signal_id} trade={trade} />
            ))}
          </div>
          {totalTrades > 0 && (
            <div className="mt-2 pt-2 border-t border-slate-200 text-xs text-slate-500 text-center">
              Avg P&L: {avgPnl >= 0 ? '+' : ''}{avgPnl.toFixed(1)}%
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {scalpPositions.length === 0 && scalpSignals.length === 0 && scalpClosedTrades.length === 0 && (
        <div className="text-center text-slate-400 py-8">
          <p className="text-sm">Waiting for scalp signals...</p>
          <p className="text-xs mt-1">Scalping targets rapid moves in 0DTE/1DTE options</p>
        </div>
      )}
    </div>
  );
}
