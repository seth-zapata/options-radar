/**
 * Main application component for OptionsRadar.
 * Features tabbed right panel with Signals, Scanner, and Positions.
 */

import { useState } from 'react';
import { useOptionsStream } from './hooks/useOptionsStream';
import { useOptionsStore } from './store/optionsStore';
import { StatusBar } from './components/StatusBar';
import { ChainView } from './components/ChainView';
import { AbstainPanel } from './components/AbstainPanel';
import { RecommendationsPanel } from './components/RecommendationsPanel';
import { ScannerPanel } from './components/ScannerPanel';
import { PositionsPanel } from './components/PositionsPanel';

type RightPanelTab = 'signals' | 'scanner' | 'positions';

function App() {
  // Connect to WebSocket
  useOptionsStream();

  // Get state for tab badges
  const recommendations = useOptionsStore((state) => state.recommendations);
  const positions = useOptionsStore((state) => state.positions);
  const abstain = useOptionsStore((state) => state.abstain);

  // Right panel tab state
  const [activeTab, setActiveTab] = useState<RightPanelTab>('signals');

  // Show AbstainPanel only when abstaining AND on signals tab with no recommendations
  const showAbstainPanel = activeTab === 'signals' && abstain && recommendations.length === 0;

  const openPositionCount = positions.filter(p => p.status === 'open').length;

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
                  onClick={() => setActiveTab('signals')}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
                    activeTab === 'signals'
                      ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Signals
                  {recommendations.length > 0 && (
                    <span className="ml-1.5 px-1.5 py-0.5 text-xs bg-indigo-600 text-white rounded-full">
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
                  Scanner
                </button>
                <button
                  onClick={() => setActiveTab('positions')}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'positions'
                      ? 'bg-green-50 text-green-700 border-b-2 border-green-600'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Positions
                  {openPositionCount > 0 && (
                    <span className="ml-1.5 px-1.5 py-0.5 text-xs bg-green-600 text-white rounded-full">
                      {openPositionCount}
                    </span>
                  )}
                </button>
              </div>
            </div>

            {/* Tab Content */}
            {activeTab === 'signals' && (
              <>
                <RecommendationsPanel />
                {showAbstainPanel && <AbstainPanel />}
              </>
            )}

            {activeTab === 'scanner' && <ScannerPanel />}

            {activeTab === 'positions' && <PositionsPanel />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
