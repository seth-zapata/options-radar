/**
 * Options chain view with expiration dropdown selector.
 * Shows calls and puts for the selected expiration date.
 */

import { useState, useMemo } from 'react';
import { useOptionsStore, selectOptionsByExpiry } from '../store/optionsStore';
import type { OptionData } from '../types';

interface OptionRowProps {
  call: OptionData | null;
  put: OptionData | null;
  strike: number;
  atmStrike: number | null;
}

function OptionCell({ option, type }: { option: OptionData | null; type: 'call' | 'put' }) {
  if (!option) {
    return <td colSpan={4} className="px-2 py-1 text-center text-slate-400">-</td>;
  }

  const isStale = (option.quoteAge ?? 0) > 5;
  const staleClass = isStale ? 'opacity-50' : '';

  const formatPrice = (value: number | null) => {
    if (value === null) return '-';
    return value.toFixed(2);
  };

  const formatGreek = (value: number | null) => {
    if (value === null) return '-';
    return value.toFixed(2);
  };

  const formatIV = (iv: number | null) => {
    if (iv === null) return '-';
    return `${(iv * 100).toFixed(0)}%`;
  };

  return (
    <>
      <td className={`px-2 py-1 text-right font-mono text-sm ${staleClass}`}>
        {formatPrice(option.bid)}
      </td>
      <td className={`px-2 py-1 text-right font-mono text-sm ${staleClass}`}>
        {formatPrice(option.ask)}
      </td>
      <td className={`px-2 py-1 text-right font-mono text-sm ${staleClass} ${type === 'call' ? 'text-green-600' : 'text-red-600'}`}>
        {formatGreek(option.delta)}
      </td>
      <td className={`px-2 py-1 text-right font-mono text-sm ${staleClass}`}>
        {formatIV(option.iv)}
      </td>
    </>
  );
}

function OptionRow({ call, put, strike, atmStrike }: OptionRowProps) {
  const isATM = atmStrike !== null && Math.abs(strike - atmStrike) < 2.5;

  return (
    <tr className={`border-b border-slate-200 hover:bg-slate-50 ${isATM ? 'bg-yellow-50' : ''}`}>
      <OptionCell option={call} type="call" />
      <td className={`px-3 py-1 text-center font-bold ${isATM ? 'text-yellow-700' : ''}`}>
        {strike.toFixed(0)}
      </td>
      <OptionCell option={put} type="put" />
    </tr>
  );
}

function formatExpiry(exp: string): string {
  const date = new Date(exp);
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function getDTE(exp: string): number {
  const now = new Date();
  const expDate = new Date(exp);
  const diffTime = expDate.getTime() - now.getTime();
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
}

export function ChainView() {
  const optionsByExpiry = useOptionsStore(selectOptionsByExpiry);
  const underlying = useOptionsStore((state) => state.underlying);
  const atmStrike = underlying?.price ?? null;

  const expiries = useMemo(() =>
    Array.from(optionsByExpiry.keys()).sort(),
    [optionsByExpiry]
  );

  // Default to first (nearest) expiration
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null);

  // Use first expiry if none selected or selection no longer valid
  const activeExpiry = selectedExpiry && expiries.includes(selectedExpiry)
    ? selectedExpiry
    : expiries[0] || null;

  const options = activeExpiry ? optionsByExpiry.get(activeExpiry) || [] : [];

  // Group by strike
  const byStrike = useMemo(() => {
    const map = new Map<number, { call?: OptionData; put?: OptionData }>();

    options.forEach((option) => {
      const strike = option.canonicalId.strike;
      if (!map.has(strike)) {
        map.set(strike, {});
      }
      const entry = map.get(strike)!;
      if (option.canonicalId.right === 'C') {
        entry.call = option;
      } else {
        entry.put = option;
      }
    });

    return map;
  }, [options]);

  const strikes = Array.from(byStrike.keys()).sort((a, b) => a - b);

  if (expiries.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow">
        <div className="p-8 text-center text-slate-500">
          <p className="text-lg font-medium">Waiting for options data...</p>
          <p className="text-sm mt-2">If the market is closed, no quotes will be received.</p>
          <p className="text-sm mt-1">US Options Market: Mon-Fri, 9:30 AM - 4:00 PM ET</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      {/* Expiration Selector */}
      <div className="bg-slate-50 px-4 py-3 border-b flex items-center justify-between">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-slate-600">Expiration:</label>
          <select
            value={activeExpiry || ''}
            onChange={(e) => setSelectedExpiry(e.target.value)}
            className="px-3 py-1.5 bg-white border border-slate-300 rounded-md text-sm font-medium focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {expiries.map((exp) => {
              const dte = getDTE(exp);
              return (
                <option key={exp} value={exp}>
                  {formatExpiry(exp)} ({dte} DTE)
                </option>
              );
            })}
          </select>
        </div>
        <div className="text-sm text-slate-500">
          {strikes.length} strikes
        </div>
      </div>

      {/* Options Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 sticky top-0">
            <tr>
              <th colSpan={4} className="px-2 py-2 text-center text-green-700 border-b font-semibold">
                CALLS
              </th>
              <th className="px-3 py-2 text-center border-b font-semibold">Strike</th>
              <th colSpan={4} className="px-2 py-2 text-center text-red-700 border-b font-semibold">
                PUTS
              </th>
            </tr>
            <tr className="text-xs text-slate-500 bg-slate-50">
              <th className="px-2 py-1 text-right font-medium">Bid</th>
              <th className="px-2 py-1 text-right font-medium">Ask</th>
              <th className="px-2 py-1 text-right font-medium">Delta</th>
              <th className="px-2 py-1 text-right font-medium">IV</th>
              <th className="px-3 py-1"></th>
              <th className="px-2 py-1 text-right font-medium">Bid</th>
              <th className="px-2 py-1 text-right font-medium">Ask</th>
              <th className="px-2 py-1 text-right font-medium">Delta</th>
              <th className="px-2 py-1 text-right font-medium">IV</th>
            </tr>
          </thead>
          <tbody>
            {strikes.map((strike) => {
              const entry = byStrike.get(strike)!;
              return (
                <OptionRow
                  key={strike}
                  strike={strike}
                  call={entry.call ?? null}
                  put={entry.put ?? null}
                  atmStrike={atmStrike}
                />
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
