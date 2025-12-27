/**
 * Status bar showing connection status and data freshness.
 */

import { useOptionsStore } from '../store/optionsStore';

export function StatusBar() {
  const { connectionStatus, underlying, lastMessageTime } = useOptionsStore();

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
    <div className="bg-slate-800 text-white px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold">OptionsRadar</h1>
        <span className="text-slate-400">|</span>
        <span className="text-lg">{underlying?.symbol || 'NVDA'}</span>
        {underlying && (
          <span className="text-lg font-mono">${underlying.price.toFixed(2)}</span>
        )}
      </div>

      <div className="flex items-center gap-6">
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
  );
}
