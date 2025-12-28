/**
 * Symbol search component with autocomplete.
 * Searches against the backend API for valid tradable symbols.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

interface SearchResult {
  symbol: string;
  name: string;
}

interface SymbolSearchProps {
  onSelect: (symbol: string) => void;
  onCancel: () => void;
}

export function SymbolSearch({ onSelect, onCancel }: SymbolSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle clicks outside the search container
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Search for symbols with debouncing
  const searchSymbols = useCallback(async (searchQuery: string) => {
    if (!searchQuery || searchQuery.length < 1) {
      setResults([]);
      setShowResults(false);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/symbols/search?q=${encodeURIComponent(searchQuery)}&limit=10`
      );
      const data = await response.json();
      setResults(data.results || []);
      setShowResults(data.results?.length > 0);
      setSelectedIndex(0);
    } catch (error) {
      console.error('Error searching symbols:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounce search
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      searchSymbols(query);
    }, 150);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [query, searchSymbols]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (results.length > 0 && showResults) {
        onSelect(results[selectedIndex].symbol);
      } else if (query.trim()) {
        onSelect(query.trim().toUpperCase());
      }
    } else if (e.key === 'Escape') {
      onCancel();
    } else if (e.key === 'Tab' && results.length > 0 && showResults) {
      // Tab to autocomplete with first result
      e.preventDefault();
      setQuery(results[0].symbol);
      setShowResults(false);
    }
  };

  const handleSelect = (symbol: string) => {
    onSelect(symbol);
  };

  return (
    <div ref={containerRef} className="relative flex items-center">
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value.toUpperCase())}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (results.length > 0) setShowResults(true);
          }}
          placeholder="Search..."
          className="w-32 px-2 py-1 text-sm bg-slate-600 text-white rounded-l border-0 focus:ring-1 focus:ring-indigo-500 outline-none placeholder-slate-400"
          maxLength={10}
        />
        {loading && (
          <span className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 text-xs">
            ...
          </span>
        )}

        {/* Search Results Dropdown */}
        {showResults && results.length > 0 && (
          <div className="absolute top-full left-0 mt-1 w-72 bg-slate-700 rounded-md shadow-xl overflow-hidden z-[100] border border-slate-600">
            {results.map((result, index) => (
              <button
                key={result.symbol}
                onMouseDown={(e) => {
                  e.preventDefault(); // Prevent input blur
                  handleSelect(result.symbol);
                }}
                className={`w-full px-3 py-2 text-left text-sm flex justify-between items-center ${
                  index === selectedIndex
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-200 hover:bg-slate-600'
                }`}
              >
                <span className="font-bold">{result.symbol}</span>
                <span className={`text-xs truncate ml-2 max-w-[180px] ${
                  index === selectedIndex ? 'text-indigo-200' : 'text-slate-400'
                }`}>
                  {result.name}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      <button
        onMouseDown={(e) => {
          e.preventDefault();
          if (query.trim()) {
            onSelect(query.trim().toUpperCase());
          }
        }}
        className="px-2 py-1 bg-green-600 hover:bg-green-500 text-white text-sm"
      >
        +
      </button>
      <button
        onMouseDown={(e) => {
          e.preventDefault();
          onCancel();
        }}
        className="px-2 py-1 bg-slate-500 hover:bg-slate-400 text-white text-sm rounded-r"
      >
        ×
      </button>
    </div>
  );
}
