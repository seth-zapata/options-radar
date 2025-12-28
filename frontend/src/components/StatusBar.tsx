/**
 * Status bar showing connection status, symbol tabs, and data freshness.
 */

import { useEffect, useState, useCallback } from 'react';
import { useOptionsStore, CORE_SYMBOLS } from '../store/optionsStore';

interface MarketInfo {
  isOpen: boolean;
  currentTime: string;
  nextOpen: string;
  nextClose: string;
  secondsUntilOpen: number | null;
  secondsUntilClose: number | null;
}

function formatDuration(totalSeconds: number): string {
  if (totalSeconds < 0) return '0s';

  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0 || days > 0) parts.push(`${hours}h`);
  if (minutes > 0 || hours > 0 || days > 0) parts.push(`${minutes}m`);
  parts.push(`${seconds}s`);

  return parts.join(' ');
}

export function StatusBar() {
  const {
    connectionStatus,
    underlying,
    lastMessageTime,
    options,
    activeSymbol,
    setActiveSymbol,
    watchlist,
    addToWatchlist,
    removeFromWatchlist,
  } = useOptionsStore();
  const [marketInfo, setMarketInfo] = useState<MarketInfo | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<number>(0);
  const [showAddSymbol, setShowAddSymbol] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');

  const handleAddSymbol = () => {
    if (newSymbol.trim()) {
      addToWatchlist(newSymbol.trim().toUpperCase());
      setNewSymbol('');
      setShowAddSymbol(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddSymbol();
    } else if (e.key === 'Escape') {
      setShowAddSymbol(false);
      setNewSymbol('');
    }
  };

  // Fetch market status from API
  const fetchMarketStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/market');
      const data = await response.json();
      setMarketInfo({
        isOpen: data.isOpen,
        currentTime: data.currentTime,
        nextOpen: data.nextOpen,
        nextClose: data.nextClose,
        secondsUntilOpen: data.secondsUntilOpen,
        secondsUntilClose: data.secondsUntilClose,
      });
      // Set initial countdown based on market status
      if (data.isOpen && data.secondsUntilClose != null) {
        setCountdown(data.secondsUntilClose);
      } else if (!data.isOpen && data.secondsUntilOpen != null) {
        setCountdown(data.secondsUntilOpen);
      }
      setLastFetchTime(Date.now());
    } catch (error) {
      console.error('Failed to fetch market status:', error);
    }
  }, []);

  // Fetch on mount and every 5 minutes (API data refresh)
  useEffect(() => {
    fetchMarketStatus();
    const interval = setInterval(fetchMarketStatus, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, [fetchMarketStatus]);

  // Client-side countdown every second
  useEffect(() => {
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 0) return prev;
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [lastFetchTime]);

  const optionsCount = options.size;

  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'bg-green-500';
      case 'connecting':
        return 'bg-yellow-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getTimeSinceUpdate = () => {
    if (!lastMessageTime) return 'No data';
    const seconds = Math.floor((Date.now() - lastMessageTime) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    return `${Math.floor(seconds / 60)}m ago`;
  };

  return (
    <div className="bg-slate-800 text-white">
      {/* Main Status Bar */}
      <div className="px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold">OptionsRadar</h1>

          {/* Symbol and Price */}
          <span className="text-slate-400">|</span>
          <span className="text-lg font-semibold">{underlying?.symbol || 'NVDA'}</span>
          {underlying && (
            <span className="text-lg font-mono">${underlying.price.toFixed(2)}</span>
          )}
        </div>

        <div className="flex items-center gap-6">
          {/* Market Status */}
          <div className="text-sm flex items-center gap-2">
            {marketInfo ? (
              <>
                <span
                  className={`inline-block w-2 h-2 rounded-full ${marketInfo.isOpen ? 'bg-green-400' : 'bg-yellow-400'}`}
                />
                <span
                  className={marketInfo.isOpen ? 'text-green-400' : 'text-yellow-400'}
                  title={marketInfo.isOpen ? `Closes: ${marketInfo.nextClose}` : `Opens: ${marketInfo.nextOpen}`}
                >
                  {marketInfo.isOpen ? 'Open' : 'Closed'}
                </span>
                <span className="text-slate-400">|</span>
                <span className="font-mono">
                  {marketInfo.isOpen ? (
                    <span title={`Closes: ${marketInfo.nextClose}`}>
                      Closes in {countdown !== null ? formatDuration(countdown) : '--'}
                    </span>
                  ) : (
                    <span title={`Opens: ${marketInfo.nextOpen}`}>
                      Opens in {countdown !== null ? formatDuration(countdown) : '--'}
                    </span>
                  )}
                </span>
              </>
            ) : (
              <span className="text-slate-400">Loading...</span>
            )}
          </div>

          {/* Options Count */}
          <div className="text-sm">
            <span className="text-slate-400">Options:</span>{' '}
            <span>{optionsCount}</span>
          </div>

          {/* IV Rank */}
          {underlying && (
            <div className="text-sm">
              <span className="text-slate-400">IV Rank:</span>{' '}
              <span className={underlying.ivRank > 50 ? 'text-orange-400' : 'text-blue-400'}>
                {underlying.ivRank.toFixed(1)}
              </span>
            </div>
          )}

          {/* Last Update */}
          <div className="text-sm">
            <span className="text-slate-400">Updated:</span>{' '}
            <span>{getTimeSinceUpdate()}</span>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${getConnectionColor()}`} />
            <span className="text-sm capitalize">{connectionStatus}</span>
          </div>
        </div>
      </div>

      {/* Symbol Tabs */}
      <div className="bg-slate-700 px-4 py-1 flex items-center gap-1 overflow-x-auto">
        {watchlist.map((symbol) => {
          const isActive = symbol === activeSymbol;
          const isCore = CORE_SYMBOLS.includes(symbol);

          return (
            <div key={symbol} className="flex items-center">
              <button
                onClick={() => setActiveSymbol(symbol)}
                className={`px-3 py-1 rounded-l text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-600 text-slate-200 hover:bg-slate-500'
                } ${isCore ? 'rounded-r' : ''}`}
              >
                {symbol}
              </button>
              {!isCore && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFromWatchlist(symbol);
                  }}
                  className={`px-1.5 py-1 rounded-r text-xs transition-colors ${
                    isActive
                      ? 'bg-indigo-700 text-indigo-200 hover:bg-red-600 hover:text-white'
                      : 'bg-slate-500 text-slate-300 hover:bg-red-600 hover:text-white'
                  }`}
                  title={`Remove ${symbol}`}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}

        {/* Add Symbol Button/Input */}
        {showAddSymbol ? (
          <div className="flex items-center">
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              onKeyDown={handleKeyPress}
              placeholder="SYMBOL"
              className="w-20 px-2 py-1 text-sm bg-slate-600 text-white rounded-l border-0 focus:ring-1 focus:ring-indigo-500 outline-none placeholder-slate-400"
              autoFocus
              maxLength={5}
            />
            <button
              onClick={handleAddSymbol}
              className="px-2 py-1 bg-green-600 hover:bg-green-500 text-white text-sm rounded-r"
            >
              +
            </button>
            <button
              onClick={() => {
                setShowAddSymbol(false);
                setNewSymbol('');
              }}
              className="ml-1 px-2 py-1 bg-slate-500 hover:bg-slate-400 text-white text-sm rounded"
            >
              ×
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowAddSymbol(true)}
            className="px-3 py-1 rounded text-sm font-medium bg-slate-600 text-slate-300 hover:bg-green-600 hover:text-white transition-colors"
            title="Add symbol to watchlist"
          >
            +
          </button>
        )}
      </div>
    </div>
  );
}
