/**
 * Main application component for OptionsRadar.
 */

import { useOptionsStream } from './hooks/useOptionsStream';
import { StatusBar } from './components/StatusBar';
import { ChainView } from './components/ChainView';
import { AbstainPanel } from './components/AbstainPanel';

function App() {
  // Connect to WebSocket
  useOptionsStream();

  return (
    <div className="min-h-screen bg-slate-100">
      <StatusBar />

      <main className="container mx-auto py-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Options Chain - 2 columns */}
          <div className="lg:col-span-2">
            <ChainView />
          </div>

          {/* Abstain Panel - 1 column */}
          <div>
            <AbstainPanel />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
