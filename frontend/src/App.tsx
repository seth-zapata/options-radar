/**
 * Main application component for OptionsRadar.
 * Features tabbed panel with Regime, Signals, Scanner, Positions, and Metrics.
 * Now focused on TSLA regime strategy.
 */

import { useState } from 'react';
import { useOptionsStream } from './hooks/useOptionsStream';
import { useScannerData } from './hooks/useScannerData';
import { useOptionsStore } from './store/optionsStore';
import { StatusBar } from './components/StatusBar';
import { ChainView } from './components/ChainView';
import { AbstainPanel } from './components/AbstainPanel';
import { RecommendationsPanel } from './components/RecommendationsPanel';
import { ScannerPanel } from './components/ScannerPanel';
import { PositionsPanel } from './components/PositionsPanel';
import { MetricsPanel } from './components/MetricsPanel';
import { RegimePanel } from './components/RegimePanel';

type RightPanelTab = 'regime' | 'signals' | 'scanner' | 'positions' | 'metrics';

function App() {
  // Connect to WebSocket
  useOptionsStream();

  // Fetch scanner data on app start
  useScannerData();

  // Get state for tab badges
  const recommendations = useOptionsStore((state) => state.recommendations);
  const positions = useOptionsStore((state) => state.positions);
  const abstain = useOptionsStore((state) => state.abstain);
  const regimeStatus = useOptionsStore((state) => state.regimeStatus);

  // Right panel tab state - default to regime
  const [activeTab, setActiveTab] = useState<RightPanelTab>('regime');

  // Show AbstainPanel only when abstaining AND on signals tab with no recommendations
  const showAbstainPanel = activeTab === 'signals' && abstain && recommendations.length === 0;

  const openPositionCount = positions.filter(p => p.status === 'open').length;
  const hasActiveRegime = regimeStatus?.active_regime?.is_active;

  return (
    <div className="min-h-screen bg-slate-100">
      <StatusBar />

      <main className="container mx-auto py-4 px-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Options Chain - 2 columns */}
          <div className="lg:col-span-2">
            <ChainView />
          </div>

          {/* Right side panel with tabs - 1 column */}
          <div className="space-y-4">
            {/* Tab Navigation */}
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <div className="flex border-b">
                <button
                  onClick={() => setActiveTab('regime')}
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors relative ${
                    activeTab === 'regime'
                      ? 'bg-emerald-50 text-emerald-700 border-b-2 border-emerald-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Regime
                  {hasActiveRegime && (
                    <span className="ml-1 w-2 h-2 inline-block bg-emerald-500 rounded-full animate-pulse"></span>
                  )}
                </button>
                <button
                  onClick={() => setActiveTab('signals')}
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors relative ${
                    activeTab === 'signals'
                      ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Signals
                  {recommendations.length > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs bg-indigo-600 text-white rounded-full">
                      {recommendations.length}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setActiveTab('scanner')}
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'scanner'
                      ? 'bg-purple-50 text-purple-700 border-b-2 border-purple-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Scanner
                </button>
                <button
                  onClick={() => setActiveTab('positions')}
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'positions'
                      ? 'bg-green-50 text-green-700 border-b-2 border-green-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Pos
                  {openPositionCount > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs bg-green-600 text-white rounded-full">
                      {openPositionCount}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setActiveTab('metrics')}
                  className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'metrics'
                      ? 'bg-amber-50 text-amber-700 border-b-2 border-amber-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Stats
                </button>
              </div>
            </div>

            {/* Tab Content */}
            {activeTab === 'regime' && <RegimePanel />}

            {activeTab === 'signals' && (
              <>
                <RecommendationsPanel />
                {showAbstainPanel && <AbstainPanel />}
              </>
            )}

            {activeTab === 'scanner' && <ScannerPanel />}

            {activeTab === 'positions' && <PositionsPanel />}

            {activeTab === 'metrics' && <MetricsPanel />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
