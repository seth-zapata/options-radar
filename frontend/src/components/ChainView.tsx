/**
 * Options chain view showing calls and puts by expiration.
 */

import { useMemo } from 'react';
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

  const formatGreek = (value: number | null, decimals: number = 2) => {
    if (value === null) return '-';
    return value.toFixed(decimals);
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

interface ChainTableProps {
  expiry: string;
  options: OptionData[];
}

function ChainTable({ expiry, options }: ChainTableProps) {
  const underlying = useOptionsStore((state) => state.underlying);
  const atmStrike = underlying?.price ?? null;

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

  // Format expiry for display
  const formatExpiry = (exp: string) => {
    const date = new Date(exp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="bg-slate-100 px-4 py-2 border-b">
        <h3 className="font-semibold">{formatExpiry(expiry)}</h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-slate-50">
          <tr>
            <th colSpan={4} className="px-2 py-2 text-center text-green-700 border-b">
              CALLS
            </th>
            <th className="px-3 py-2 text-center border-b">Strike</th>
            <th colSpan={4} className="px-2 py-2 text-center text-red-700 border-b">
              PUTS
            </th>
          </tr>
          <tr className="text-xs text-slate-500">
            <th className="px-2 py-1 text-right">Bid</th>
            <th className="px-2 py-1 text-right">Ask</th>
            <th className="px-2 py-1 text-right">Delta</th>
            <th className="px-2 py-1 text-right">IV</th>
            <th className="px-3 py-1"></th>
            <th className="px-2 py-1 text-right">Bid</th>
            <th className="px-2 py-1 text-right">Ask</th>
            <th className="px-2 py-1 text-right">Delta</th>
            <th className="px-2 py-1 text-right">IV</th>
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
  );
}

export function ChainView() {
  const optionsByExpiry = useOptionsStore(selectOptionsByExpiry);

  const expiries = Array.from(optionsByExpiry.keys()).sort();

  if (expiries.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500">
        <p className="text-lg">Waiting for options data...</p>
        <p className="text-sm mt-2">Make sure the backend server is running.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      {expiries.map((expiry) => (
        <ChainTable
          key={expiry}
          expiry={expiry}
          options={optionsByExpiry.get(expiry)!}
        />
      ))}
    </div>
  );
}
