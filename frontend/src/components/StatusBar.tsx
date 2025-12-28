/**
 * Status bar showing connection status, symbol tabs, and data freshness.
 */

import { useEffect, useState, useCallback } from 'react';
import { useOptionsStore } from '../store/optionsStore';

// Watchlist - for now only NVDA is active, others are for display
const WATCHLIST = ['NVDA', 'AAPL', 'TSLA', 'AMD', 'QQQ', 'SPY'];

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
  const { connectionStatus, underlying, lastMessageTime, options } = useOptionsStore();
  const [marketInfo, setMarketInfo] = useState<MarketInfo | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<number>(0);
  const [activeSymbol, setActiveSymbol] = useState('NVDA');

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
        {WATCHLIST.map((symbol) => {
          const isActive = symbol === activeSymbol;
          const isAvailable = symbol === 'NVDA'; // Only NVDA is active for now

          return (
            <button
              key={symbol}
              onClick={() => isAvailable && setActiveSymbol(symbol)}
              disabled={!isAvailable}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : isAvailable
                    ? 'bg-slate-600 text-slate-200 hover:bg-slate-500'
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed'
              }`}
              title={isAvailable ? undefined : 'Coming soon'}
            >
              {symbol}
              {!isAvailable && <span className="ml-1 text-xs opacity-50">*</span>}
            </button>
          );
        })}
        <span className="text-xs text-slate-500 ml-2">* Coming soon</span>
      </div>
    </div>
  );
}
