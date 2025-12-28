/**
 * Main application component for OptionsRadar.
 */

import { useOptionsStream } from './hooks/useOptionsStream';
import { useOptionsStore } from './store/optionsStore';
import { StatusBar } from './components/StatusBar';
import { ChainView } from './components/ChainView';
import { AbstainPanel } from './components/AbstainPanel';
import { RecommendationsPanel } from './components/RecommendationsPanel';

function App() {
  // Connect to WebSocket
  useOptionsStream();

  // Get abstain state to determine which panel to show
  const abstain = useOptionsStore((state) => state.abstain);
  const recommendations = useOptionsStore((state) => state.recommendations);

  // Show AbstainPanel only when abstaining AND we have no recommendations yet
  const showAbstainPanel = abstain && recommendations.length === 0;

  return (
    <div className="min-h-screen bg-slate-100">
      <StatusBar />

      <main className="container mx-auto py-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Options Chain - 2 columns */}
          <div className="lg:col-span-2">
            <ChainView />
          </div>

          {/* Right side panels - 1 column */}
          <div className="space-y-4">
            {/* Primary: Trade Signals */}
            <RecommendationsPanel />

            {/* Secondary: Only show when abstaining with no signals */}
            {showAbstainPanel && <AbstainPanel />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
