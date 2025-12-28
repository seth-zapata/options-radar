/**
 * Options chain view with expiration dropdown selector.
 * Shows calls and puts for the selected expiration date.
 */

import { useState, useMemo } from 'react';
import { useOptionsStore, selectOptionsByExpiry } from '../store/optionsStore';
import type { OptionData } from '../types';

const API_BASE = 'http://localhost:8000';

interface ManualTradeModalProps {
  option: OptionData;
  action: 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT';
  onClose: () => void;
  onConfirm: () => void;
}

function ManualTradeModal({ option, action, onClose, onConfirm }: ManualTradeModalProps) {
  const midPrice = option.bid && option.ask ? (option.bid + option.ask) / 2 : option.bid || option.ask || 0;
  const [fillPrice, setFillPrice] = useState(midPrice.toFixed(2));
  const [contracts, setContracts] = useState('1');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const price = parseFloat(fillPrice);
    const qty = parseInt(contracts, 10);

    if (isNaN(price) || price <= 0) {
      setError('Please enter a valid fill price');
      return;
    }
    if (isNaN(qty) || qty <= 0) {
      setError('Please enter a valid number of contracts');
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch(`${API_BASE}/api/positions/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          underlying: option.canonicalId.underlying,
          expiry: option.canonicalId.expiry,
          strike: option.canonicalId.strike,
          right: option.canonicalId.right,
          action: action,
          contracts: qty,
          fill_price: price,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to open position');
      }

      onConfirm();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open position');
    } finally {
      setSubmitting(false);
    }
  };

  const actionLabel = action.replace('_', ' ');
  const totalCost = (parseFloat(fillPrice) || 0) * (parseInt(contracts, 10) || 0) * 100;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="px-4 py-3 border-b">
          <h3 className="font-bold text-lg">{actionLabel}</h3>
          <p className="text-sm text-slate-500">
            {option.canonicalId.underlying} ${option.canonicalId.strike} {option.canonicalId.right === 'C' ? 'Call' : 'Put'}
            <span className="ml-2 text-xs">({option.canonicalId.expiry})</span>
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-4">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Fill Price (per contract)
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={fillPrice}
                  onChange={(e) => setFillPrice(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  autoFocus
                />
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Bid: ${option.bid?.toFixed(2) || '-'} / Ask: ${option.ask?.toFixed(2) || '-'}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Contracts
              </label>
              <input
                type="number"
                min="1"
                value={contracts}
                onChange={(e) => setContracts(e.target.value)}
                className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>

            <div className="bg-slate-50 rounded p-3">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Total Cost</span>
                <span className="font-bold">${totalCost.toFixed(0)}</span>
              </div>
            </div>

            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
          </div>

          <div className="flex gap-3 mt-6">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-slate-300 rounded-md text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
            >
              {submitting ? 'Opening...' : 'Open Position'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

interface OptionRowProps {
  call: OptionData | null;
  put: OptionData | null;
  strike: number;
  atmStrike: number | null;
  onTrade: (option: OptionData, action: 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT') => void;
}

function OptionCell({ option, type, onTrade }: { option: OptionData | null; type: 'call' | 'put'; onTrade: (option: OptionData, action: 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT') => void }) {
  if (!option) {
    return <td colSpan={5} className="px-2 py-1 text-center text-slate-400">-</td>;
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

  const buyAction = type === 'call' ? 'BUY_CALL' : 'BUY_PUT';
  const sellAction = type === 'call' ? 'SELL_CALL' : 'SELL_PUT';

  return (
    <>
      <td className={`px-1 py-1 ${staleClass}`}>
        <div className="flex gap-0.5">
          <button
            onClick={() => onTrade(option, buyAction)}
            className="px-1.5 py-0.5 text-xs bg-green-100 hover:bg-green-200 text-green-700 rounded font-medium"
            title={`Buy ${type}`}
          >
            B
          </button>
          <button
            onClick={() => onTrade(option, sellAction)}
            className="px-1.5 py-0.5 text-xs bg-red-100 hover:bg-red-200 text-red-700 rounded font-medium"
            title={`Sell ${type}`}
          >
            S
          </button>
        </div>
      </td>
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

function OptionRow({ call, put, strike, atmStrike, onTrade }: OptionRowProps) {
  const isATM = atmStrike !== null && Math.abs(strike - atmStrike) < 2.5;

  return (
    <tr className={`border-b border-slate-200 hover:bg-slate-50 ${isATM ? 'bg-yellow-50' : ''}`}>
      <OptionCell option={call} type="call" onTrade={onTrade} />
      <td className={`px-3 py-1 text-center font-bold ${isATM ? 'text-yellow-700' : ''}`}>
        {strike.toFixed(0)}
      </td>
      <OptionCell option={put} type="put" onTrade={onTrade} />
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

  // Filter out expired options (DTE <= 0) and sort
  const expiries = useMemo(() =>
    Array.from(optionsByExpiry.keys())
      .filter((exp) => getDTE(exp) > 0)
      .sort(),
    [optionsByExpiry]
  );

  // Default to first (nearest) expiration
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null);

  // Trade modal state
  const [tradeModal, setTradeModal] = useState<{
    option: OptionData;
    action: 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT';
  } | null>(null);

  const handleTrade = (option: OptionData, action: 'BUY_CALL' | 'BUY_PUT' | 'SELL_CALL' | 'SELL_PUT') => {
    setTradeModal({ option, action });
  };

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
              <th colSpan={5} className="px-2 py-2 text-center text-green-700 border-b font-semibold">
                CALLS
              </th>
              <th className="px-3 py-2 text-center border-b font-semibold">Strike</th>
              <th colSpan={5} className="px-2 py-2 text-center text-red-700 border-b font-semibold">
                PUTS
              </th>
            </tr>
            <tr className="text-xs text-slate-500 bg-slate-50">
              <th className="px-1 py-1 text-center font-medium">Trade</th>
              <th className="px-2 py-1 text-right font-medium">Bid</th>
              <th className="px-2 py-1 text-right font-medium">Ask</th>
              <th className="px-2 py-1 text-right font-medium">Delta</th>
              <th className="px-2 py-1 text-right font-medium">IV</th>
              <th className="px-3 py-1"></th>
              <th className="px-1 py-1 text-center font-medium">Trade</th>
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
                  onTrade={handleTrade}
                />
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Trade Modal */}
      {tradeModal && (
        <ManualTradeModal
          option={tradeModal.option}
          action={tradeModal.action}
          onClose={() => setTradeModal(null)}
          onConfirm={() => setTradeModal(null)}
        />
      )}
    </div>
  );
}
