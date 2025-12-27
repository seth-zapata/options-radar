/**
 * Panel showing why the system is abstaining from recommendations.
 */

import { useOptionsStore } from '../store/optionsStore';
import type { GateResult } from '../types';

function GateResultRow({ gate }: { gate: GateResult }) {
  const statusClass = gate.passed
    ? 'bg-green-100 text-green-800'
    : 'bg-red-100 text-red-800';

  const statusText = gate.passed ? 'PASS' : 'FAIL';

  return (
    <div className="flex items-center justify-between py-2 px-3 border-b last:border-b-0">
      <div className="flex items-center gap-3">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusClass}`}>
          {statusText}
        </span>
        <span className="font-medium text-sm">{gate.name}</span>
      </div>
      <span className="text-sm text-slate-600">{gate.message}</span>
    </div>
  );
}

export function AbstainPanel() {
  const { abstain, gateResults } = useOptionsStore();

  const failedGates = gateResults.filter((g) => !g.passed);
  const passedGates = gateResults.filter((g) => g.passed);

  // Determine overall status
  const hasHardFailure = failedGates.some((g) =>
    ['underlying_price_fresh', 'quote_fresh', 'greeks_fresh', 'spread_acceptable',
     'open_interest_sufficient', 'delta_in_range', 'cash_available', 'position_size_limit']
      .includes(g.name)
  );

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className={`px-4 py-3 ${hasHardFailure || abstain ? 'bg-amber-100' : 'bg-green-100'}`}>
        <div className="flex items-center justify-between">
          <h2 className="font-bold text-lg">
            {hasHardFailure || abstain ? 'ABSTAINING' : 'MONITORING'}
          </h2>
          {abstain && (
            <span className="px-3 py-1 bg-amber-200 rounded-full text-sm font-medium">
              {abstain.reason.replace(/_/g, ' ')}
            </span>
          )}
        </div>
        {abstain && (
          <p className="text-sm mt-1 text-amber-800">
            {abstain.resumeCondition}
          </p>
        )}
      </div>

      <div className="divide-y">
        {/* Failed gates first */}
        {failedGates.length > 0 && (
          <div>
            <div className="px-4 py-2 bg-red-50">
              <h3 className="text-sm font-semibold text-red-800">
                Failed Gates ({failedGates.length})
              </h3>
            </div>
            {failedGates.map((gate) => (
              <GateResultRow key={gate.name} gate={gate} />
            ))}
          </div>
        )}

        {/* Passed gates */}
        {passedGates.length > 0 && (
          <div>
            <div className="px-4 py-2 bg-green-50">
              <h3 className="text-sm font-semibold text-green-800">
                Passed Gates ({passedGates.length})
              </h3>
            </div>
            {passedGates.map((gate) => (
              <GateResultRow key={gate.name} gate={gate} />
            ))}
          </div>
        )}

        {/* No gates yet */}
        {gateResults.length === 0 && !abstain && (
          <div className="px-4 py-8 text-center text-slate-500">
            <p>No gate evaluations yet.</p>
            <p className="text-sm mt-1">Gates are evaluated when options data is received.</p>
          </div>
        )}
      </div>
    </div>
  );
}
