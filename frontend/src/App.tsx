/**
 * Main application component for OptionsRadar.
 * Full-width tabbed layout focused on actionable trading.
 */

import { useState } from 'react';
import { useOptionsStream } from './hooks/useOptionsStream';
import { useScannerData } from './hooks/useScannerData';
import { useOptionsStore } from './store/optionsStore';
import { StatusBar } from './components/StatusBar';
import { RecommendationsPanel } from './components/RecommendationsPanel';
import { ScannerPanel } from './components/ScannerPanel';
import { MetricsPanel } from './components/MetricsPanel';
import { TradingDashboard } from './components/TradingDashboard';

type MainTab = 'tsla' | 'signals' | 'scanner' | 'stats';

function App() {
  // Connect to WebSocket
  useOptionsStream();

  // Fetch scanner data on app start
  useScannerData();

  // Get state for tab badges
  const recommendations = useOptionsStore((state) => state.recommendations);
  const positions = useOptionsStore((state) => state.positions);
  const regimeStatus = useOptionsStore((state) => state.regimeStatus);

  // Main tab state - default to TSLA trading dashboard
  const [activeTab, setActiveTab] = useState<MainTab>('tsla');

  const openPositionCount = positions.filter(p => p.status === 'open' || p.status === 'exit_signal').length;
  const hasActiveRegime = regimeStatus?.active_regime?.is_active;

  return (
    <div className="min-h-screen bg-slate-100">
      <StatusBar />

      <main className="container mx-auto py-4 px-4">
        {/* Tab Navigation - Full Width */}
        <div className="bg-white rounded-lg shadow mb-4 overflow-hidden">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('tsla')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
                activeTab === 'tsla'
                  ? 'bg-emerald-50 text-emerald-700 border-b-2 border-emerald-600'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span className="font-bold">TSLA</span>
              <span className="ml-1 text-xs opacity-75">Trading</span>
              {hasActiveRegime && (
                <span className="absolute top-2 right-2 w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              )}
              {openPositionCount > 0 && (
                <span className="ml-2 px-1.5 py-0.5 text-xs bg-emerald-600 text-white rounded-full">
                  {openPositionCount}
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('signals')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
                activeTab === 'signals'
                  ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-600'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span className="font-bold">Signals</span>
              <span className="ml-1 text-xs opacity-75">11-Gate</span>
              {recommendations.length > 0 && (
                <span className="ml-2 px-1.5 py-0.5 text-xs bg-indigo-600 text-white rounded-full">
                  {recommendations.length}
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('scanner')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'scanner'
                  ? 'bg-purple-50 text-purple-700 border-b-2 border-purple-600'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span className="font-bold">Scanner</span>
              <span className="ml-1 text-xs opacity-75">WSB/News</span>
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'stats'
                  ? 'bg-amber-50 text-amber-700 border-b-2 border-amber-600'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span className="font-bold">Stats</span>
              <span className="ml-1 text-xs opacity-75">Performance</span>
            </button>
          </div>
        </div>

        {/* Tab Content - Full Width */}
        {activeTab === 'tsla' && <TradingDashboard />}

        {activeTab === 'signals' && <RecommendationsPanel />}

        {activeTab === 'scanner' && <ScannerPanel />}

        {activeTab === 'stats' && <MetricsPanel />}
      </main>
    </div>
  );
}

export default App;
