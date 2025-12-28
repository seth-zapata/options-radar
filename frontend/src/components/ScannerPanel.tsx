/**
 * Scanner panel showing WSB trending, opportunities, and sentiment data.
 * Fetches data from the backend scanner API endpoints.
 */

import { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

interface WSBTrending {
  symbol: string;
  mentions24h: number;
  sentiment: number;
  sentimentScore: number;
  rank: number;
  buzzLevel: string;
  isBullish: boolean;
}

interface ScanResult {
  symbol: string;
  score: number;
  direction: string;
  isOpportunity: boolean;
  isStrongOpportunity: boolean;
  signals: string[];
  sentiment: {
    scores: {
      news: number;
      wsb: number;
      combined: number;
    };
    signal: string;
    strength: string;
    flags: {
      newsBuzzing: boolean;
      wsbTrending: boolean;
      wsbBullish: boolean;
      sourcesAligned: boolean;
    };
  } | null;
}

interface HotPicks {
  wsbTrending: WSBTrending[];
  topOpportunities: ScanResult[];
}

function SentimentBadge({ score, label }: { score: number; label: string }) {
  const getBadgeColor = () => {
    if (score >= 25) return 'bg-green-100 text-green-800';
    if (score <= -25) return 'bg-red-100 text-red-800';
    return 'bg-slate-100 text-slate-600';
  };

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getBadgeColor()}`}>
      {label}: {score >= 0 ? '+' : ''}{score.toFixed(0)}
    </span>
  );
}

function WSBCard({ item }: { item: WSBTrending }) {
  return (
    <div className="flex items-center justify-between py-2 px-3 bg-slate-50 rounded-lg">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-slate-500">#{item.rank}</span>
        <span className="font-bold">{item.symbol}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${item.isBullish ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
          {item.isBullish ? 'Bullish' : 'Bearish'}
        </span>
      </div>
      <div className="flex items-center gap-3 text-sm">
        <span className="text-slate-500">{item.mentions24h.toLocaleString()} mentions</span>
        <span className={`font-medium ${item.sentimentScore >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {item.sentimentScore >= 0 ? '+' : ''}{item.sentimentScore.toFixed(0)}
        </span>
      </div>
    </div>
  );
}

function OpportunityCard({ item }: { item: ScanResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`p-3 rounded-lg border ${item.isStrongOpportunity ? 'border-green-300 bg-green-50' : 'border-slate-200 bg-white'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold text-lg">{item.symbol}</span>
          <span className={`text-xs px-2 py-0.5 rounded font-medium ${
            item.direction === 'bullish' ? 'bg-green-100 text-green-700' :
            item.direction === 'bearish' ? 'bg-red-100 text-red-700' :
            'bg-slate-100 text-slate-600'
          }`}>
            {item.direction}
          </span>
          {item.isStrongOpportunity && (
            <span className="text-xs px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded font-medium">
              Strong
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-lg font-bold ${item.score >= 75 ? 'text-green-600' : item.score >= 50 ? 'text-yellow-600' : 'text-slate-600'}`}>
            {item.score.toFixed(0)}
          </span>
          <span className="text-xs text-slate-500">score</span>
        </div>
      </div>

      {/* Sentiment badges */}
      {item.sentiment && (
        <div className="flex flex-wrap gap-1 mt-2">
          <SentimentBadge score={item.sentiment.scores.news} label="News" />
          <SentimentBadge score={item.sentiment.scores.wsb} label="WSB" />
          {item.sentiment.flags.wsbTrending && (
            <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded">Trending</span>
          )}
          {item.sentiment.flags.sourcesAligned && (
            <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">Aligned</span>
          )}
        </div>
      )}

      {/* Signals (collapsible) */}
      {item.signals.length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-left text-xs text-slate-500 mt-2 hover:text-slate-700"
        >
          {expanded ? '- Hide signals' : `+ ${item.signals.length} signals`}
        </button>
      )}

      {expanded && item.signals.length > 0 && (
        <ul className="mt-1 text-xs text-slate-600 space-y-0.5">
          {item.signals.map((signal, i) => (
            <li key={i} className="flex items-center gap-1">
              <span className="text-green-500">+</span>
              {signal}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ScannerPanel() {
  const [hotPicks, setHotPicks] = useState<HotPicks | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  const fetchHotPicks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE}/api/scanner/hot-picks`);
      if (!response.ok) throw new Error('Failed to fetch');

      const data = await response.json();
      setHotPicks(data);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on mount and every 5 minutes
  useEffect(() => {
    fetchHotPicks();
    const interval = setInterval(fetchHotPicks, 300000);
    return () => clearInterval(interval);
  }, [fetchHotPicks]);

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-purple-600 text-white flex items-center justify-between">
        <div>
          <h2 className="font-bold text-lg">Daily Scanner</h2>
          <p className="text-sm text-purple-200">
            Sentiment-driven opportunities
          </p>
        </div>
        <button
          onClick={fetchHotPicks}
          disabled={loading}
          className="px-3 py-1 bg-purple-500 hover:bg-purple-400 rounded text-sm disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      <div className="p-4 space-y-4">
        {error && (
          <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* WSB Trending Section */}
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
            <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
            WSB Trending
          </h3>
          {loading && !hotPicks ? (
            <div className="text-center py-4 text-slate-500 text-sm">Loading...</div>
          ) : hotPicks?.wsbTrending && hotPicks.wsbTrending.length > 0 ? (
            <div className="space-y-2">
              {hotPicks.wsbTrending.slice(0, 5).map((item) => (
                <WSBCard key={item.symbol} item={item} />
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-slate-500 text-sm">
              No WSB data available
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="border-t border-slate-200"></div>

        {/* Top Opportunities Section */}
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            Top Opportunities
          </h3>
          {loading && !hotPicks ? (
            <div className="text-center py-4 text-slate-500 text-sm">Loading...</div>
          ) : hotPicks?.topOpportunities && hotPicks.topOpportunities.length > 0 ? (
            <div className="space-y-2">
              {hotPicks.topOpportunities.map((item) => (
                <OpportunityCard key={item.symbol} item={item} />
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-slate-500 text-sm">
              No opportunities found
            </div>
          )}
        </div>

        {/* Last update */}
        {lastUpdate && (
          <div className="text-xs text-slate-400 text-center pt-2 border-t">
            Last updated: {lastUpdate}
          </div>
        )}
      </div>
    </div>
  );
}
