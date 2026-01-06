/**
 * Trading Control Panel - Auto-execution status and controls.
 * Shows Alpaca account info and allows enabling/disabling auto-execution.
 */

import { useEffect, useState, useCallback } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { TradingStatus, AlpacaAccount, AlpacaPosition, SimulationStatus } from '../types';

const API_BASE = 'http://localhost:8000';

export function TradingControlPanel() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [account, setAccount] = useState<AlpacaAccount | null>(null);
  const [positions, setPositions] = useState<AlpacaPosition[]>([]);
  const [simStatus, setSimStatus] = useState<SimulationStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Sync scalping status to store
  const setScalpEnabled = useOptionsStore((state) => state.setScalpEnabled);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/trading/status`);
      if (!response.ok) throw new Error('Failed to fetch status');
      const data = await response.json();
      setStatus(data);
      // Sync scalping enabled status to the store
      if (data.scalping_enabled !== undefined) {
        setScalpEnabled(data.scalping_enabled);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch status');
    }
  }, [setScalpEnabled]);

  const fetchAccount = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/trading/account`);
      if (!response.ok) throw new Error('Failed to fetch account');
      const data = await response.json();
      // Handle error response or extract nested account object
      if (data.error) {
        console.log('Account fetch returned error (may be in mock mode):', data.error);
        return;
      }
      // Backend returns {account: {...}, autoExecuteEnabled: ...} - extract the account
      if (data.account) {
        setAccount(data.account);
      } else {
        // Fallback for direct account object
        setAccount(data);
      }
    } catch (e) {
      // Account fetch may fail in mock mode - that's okay
      console.log('Account fetch failed (may be in mock mode)');
    }
  }, []);

  const fetchPositions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/trading/positions`);
      if (!response.ok) throw new Error('Failed to fetch positions');
      const data = await response.json();
      // Backend returns {positions: [...], count: N} - extract positions array
      if (data.positions) {
        setPositions(data.positions);
      } else if (Array.isArray(data)) {
        setPositions(data);
      }
    } catch (e) {
      console.log('Positions fetch failed (may be in mock mode)');
    }
  }, []);

  const fetchSimulationStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/simulation/status`);
      if (!response.ok) throw new Error('Failed to fetch simulation status');
      const data = await response.json();
      if (data.active) {
        setSimStatus(data);
      } else {
        setSimStatus(null);
      }
    } catch (e) {
      console.log('Simulation status fetch failed');
    }
  }, []);

  useEffect(() => {
    const fetchAll = async (isInitial = false) => {
      if (isInitial) setLoading(true);
      await Promise.all([fetchStatus(), fetchAccount(), fetchPositions(), fetchSimulationStatus()]);
      if (isInitial) setLoading(false);
    };

    fetchAll(true);
    // Refresh more frequently in simulation mode
    const interval = setInterval(() => fetchAll(false), simStatus?.active ? 2000 : 10000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchAccount, fetchPositions, fetchSimulationStatus, simStatus?.active]);

  const handleEnable = async () => {
    setActionLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/trading/enable`, { method: 'POST' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to enable');
      }
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to enable');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDisable = async () => {
    setActionLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/trading/disable`, { method: 'POST' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to disable');
      }
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to disable');
    } finally {
      setActionLoading(false);
    }
  };

  const handleSync = async () => {
    setActionLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/trading/sync`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to sync');
      const data = await response.json();
      alert(`Synced ${data.synced_count} positions from Alpaca`);
      await Promise.all([fetchStatus(), fetchPositions()]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to sync');
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="animate-pulse flex items-center gap-2">
          <div className="w-4 h-4 bg-slate-200 rounded-full"></div>
          <div className="h-4 bg-slate-200 rounded w-32"></div>
        </div>
      </div>
    );
  }

  // Not configured state
  if (!status?.configured) {
    return (
      <div className="bg-slate-100 rounded-lg border-2 border-dashed border-slate-300 p-4">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 bg-slate-400 rounded-full"></div>
          <div>
            <span className="font-medium text-slate-600">Auto-Execution Not Configured</span>
            <p className="text-xs text-slate-500 mt-0.5">
              Set AUTO_EXECUTE=true in .env and restart with valid Alpaca credentials
            </p>
          </div>
        </div>
      </div>
    );
  }

  const isEnabled = status.enabled && status.auto_execution;
  const isSimulation = status.simulation_mode && simStatus?.active;

  // Determine styling based on mode
  const getHeaderStyle = () => {
    if (isSimulation) return 'bg-purple-600 text-white';
    if (isEnabled) return 'bg-emerald-600 text-white';
    return 'bg-slate-600 text-white';
  };

  const getPanelStyle = () => {
    if (isSimulation) return 'bg-purple-50 border-2 border-purple-400';
    if (isEnabled) return 'bg-emerald-50 border-2 border-emerald-400';
    return 'bg-white';
  };

  return (
    <div className={`rounded-lg shadow overflow-hidden ${getPanelStyle()}`}>
      {/* Header */}
      <div className={`px-4 py-3 flex items-center justify-between ${getHeaderStyle()}`}>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${
            isSimulation ? 'bg-purple-300 animate-pulse' :
            isEnabled ? 'bg-green-300 animate-pulse' : 'bg-slate-400'
          }`}></div>
          <span className="font-bold">
            {isSimulation ? 'SIMULATION MODE' :
             isEnabled ? 'AUTO-EXECUTION ACTIVE' : 'Auto-Execution Off'}
          </span>
          <span className="text-xs opacity-75 ml-1">
            {isSimulation ? `${simStatus?.speed}x Speed` : 'Paper Trading'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {isEnabled ? (
            <button
              onClick={handleDisable}
              disabled={actionLoading}
              className="px-3 py-1 bg-red-500 hover:bg-red-600 rounded text-sm font-medium disabled:opacity-50"
            >
              {actionLoading ? '...' : 'Disable'}
            </button>
          ) : (
            <button
              onClick={handleEnable}
              disabled={actionLoading}
              className="px-3 py-1 bg-emerald-500 hover:bg-emerald-600 rounded text-sm font-medium disabled:opacity-50"
            >
              {actionLoading ? '...' : 'Enable'}
            </button>
          )}
          <button
            onClick={handleSync}
            disabled={actionLoading}
            className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm disabled:opacity-50"
            title="Sync positions from Alpaca"
          >
            Sync
          </button>
        </div>
      </div>

      {/* Status Details */}
      <div className="p-4">
        {error && (
          <div className="mb-3 p-2 bg-red-100 text-red-700 rounded text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Positions */}
          <div className="bg-slate-50 rounded-lg p-3 text-center">
            <div className="text-xs text-slate-500 uppercase">Positions</div>
            <div className="text-xl font-bold text-slate-800">
              {status.open_positions} / {status.max_positions}
            </div>
          </div>

          {/* Position Size */}
          <div className="bg-slate-50 rounded-lg p-3 text-center">
            <div className="text-xs text-slate-500 uppercase">Position Size</div>
            <div className="text-xl font-bold text-slate-800">
              {status.position_size_pct}%
            </div>
          </div>

          {/* Exit Monitor */}
          <div className="bg-slate-50 rounded-lg p-3 text-center">
            <div className="text-xs text-slate-500 uppercase">Exit Monitor</div>
            <div className={`text-sm font-bold ${
              status.exit_monitor_running ? 'text-emerald-600' : 'text-slate-400'
            }`}>
              {status.exit_monitor_running ? 'Running' : 'Stopped'}
            </div>
          </div>

          {/* Buying Power */}
          {account && (
            <div className="bg-slate-50 rounded-lg p-3 text-center">
              <div className="text-xs text-slate-500 uppercase">Buying Power</div>
              <div className="text-xl font-bold text-emerald-600">
                ${account.buying_power?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}
              </div>
            </div>
          )}
        </div>

        {/* Account Summary */}
        {account && (
          <div className="mt-4 flex flex-wrap gap-4 text-sm text-slate-600">
            <span>Equity: <strong>${account.equity?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}</strong></span>
            <span>Cash: <strong>${account.cash?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}</strong></span>
            <span>Alpaca Positions: <strong>{account.positions_count ?? '—'}</strong></span>
            {account.pattern_day_trader && (
              <span className="text-amber-600">PDT Flagged</span>
            )}
          </div>
        )}

        {/* Alpaca Positions */}
        {positions.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-slate-700 mb-2">Alpaca Positions</h4>
            <div className="space-y-1">
              {positions.slice(0, 5).map((pos) => (
                <div key={pos.symbol} className="flex items-center justify-between text-sm bg-slate-50 rounded px-3 py-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-medium">{pos.symbol}</span>
                    <span className="text-slate-500">{pos.qty} @ ${pos.avg_entry_price.toFixed(2)}</span>
                  </div>
                  <div className={`font-medium ${pos.unrealized_pl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {pos.unrealized_pl >= 0 ? '+' : ''}${pos.unrealized_pl.toFixed(0)}
                    <span className="text-xs ml-1">
                      ({pos.unrealized_plpc >= 0 ? '+' : ''}{(pos.unrealized_plpc * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
              ))}
              {positions.length > 5 && (
                <div className="text-xs text-slate-400 text-center py-1">
                  + {positions.length - 5} more positions
                </div>
              )}
            </div>
          </div>
        )}

        {/* Simulation Info */}
        {isSimulation && simStatus && (
          <div className="mt-4 space-y-3">
            {/* Current Regime */}
            <div className="flex items-center gap-3 p-3 bg-purple-100 rounded-lg">
              <div className="text-purple-600 font-medium">Regime:</div>
              <div className={`px-2 py-1 rounded text-sm font-bold ${
                simStatus.currentRegime.includes('bullish') ? 'bg-green-200 text-green-800' :
                simStatus.currentRegime.includes('bearish') ? 'bg-red-200 text-red-800' :
                'bg-slate-200 text-slate-700'
              }`}>
                {simStatus.currentRegime.toUpperCase()}
              </div>
              <div className="text-sm text-purple-600">
                Sentiment: {simStatus.sentiment >= 0 ? '+' : ''}{simStatus.sentiment.toFixed(3)}
              </div>
            </div>

            {/* Simulated Prices */}
            {Object.entries(simStatus.prices).length > 0 && (
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                {Object.entries(simStatus.prices).map(([symbol, priceData]) => (
                  <div key={symbol} className="bg-purple-100 rounded p-2 text-center">
                    <div className="text-xs text-purple-500 font-medium">{symbol}</div>
                    <div className="text-lg font-bold text-purple-800">
                      ${priceData.price.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Mock Portfolio */}
            {simStatus.mockPortfolio && (
              <div className="bg-purple-100 rounded-lg p-3">
                <div className="text-xs text-purple-500 uppercase mb-2">Mock Portfolio</div>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <span className="text-purple-600">Cash:</span>{' '}
                    <strong>${simStatus.mockPortfolio.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
                  </div>
                  <div>
                    <span className="text-purple-600">Positions:</span>{' '}
                    <strong>{simStatus.mockPortfolio.openPositions}</strong>
                  </div>
                  <div className={simStatus.mockPortfolio.totalPl >= 0 ? 'text-green-600' : 'text-red-600'}>
                    <span>P&L:</span>{' '}
                    <strong>
                      {simStatus.mockPortfolio.totalPl >= 0 ? '+' : ''}
                      ${simStatus.mockPortfolio.totalPl.toFixed(0)}
                    </strong>
                  </div>
                </div>

                {/* Mock Positions */}
                {simStatus.mockPortfolio.positions.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {simStatus.mockPortfolio.positions.map((pos) => (
                      <div key={pos.symbol} className="flex items-center justify-between text-xs bg-white/50 rounded px-2 py-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium">{pos.symbol}</span>
                          <span className="text-purple-500">{pos.qty} @ ${pos.entryPrice.toFixed(2)}</span>
                        </div>
                        <div className={pos.pl >= 0 ? 'text-green-600' : 'text-red-600'}>
                          {pos.pl >= 0 ? '+' : ''}${pos.pl.toFixed(0)}
                          <span className="ml-1">({pos.plPercent >= 0 ? '+' : ''}{pos.plPercent.toFixed(1)}%)</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Warning Banner */}
        {isSimulation && (
          <div className="mt-4 p-2 bg-purple-100 border border-purple-300 rounded text-sm text-purple-800">
            SIMULATION MODE - Using mock data and accelerated time. No real trades.
          </div>
        )}
        {isEnabled && !isSimulation && (
          <div className="mt-4 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-800">
            Signals will auto-execute trades. Monitor your Alpaca paper account.
          </div>
        )}
      </div>
    </div>
  );
}
