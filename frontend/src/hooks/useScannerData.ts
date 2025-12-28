/**
 * Hook to fetch and manage scanner data.
 * Fetches on mount and refreshes every 5 minutes.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useOptionsStore } from '../store/optionsStore';

const API_BASE = 'http://localhost:8000';
const REFRESH_INTERVAL = 300000; // 5 minutes
const FETCH_TIMEOUT = 30000; // 30 seconds timeout (increased for slow APIs)
const MAX_RETRIES = 2;

export function useScannerData() {
  const setScannerData = useOptionsStore((state) => state.setScannerData);
  const setScannerLoading = useOptionsStore((state) => state.setScannerLoading);
  const setScannerError = useOptionsStore((state) => state.setScannerError);
  const retryCountRef = useRef(0);
  const isFetchingRef = useRef(false);

  const fetchHotPicks = useCallback(async (isRetry = false) => {
    // Prevent concurrent fetches
    if (isFetchingRef.current && !isRetry) return;
    isFetchingRef.current = true;

    try {
      setScannerLoading(true);
      if (!isRetry) {
        setScannerError(null);
        retryCountRef.current = 0;
      }

      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

      const response = await fetch(`${API_BASE}/api/scanner/hot-picks`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) throw new Error('Failed to fetch');

      const data = await response.json();

      // Check if there was an error in the response
      if (data.error) {
        throw new Error(data.error);
      }

      setScannerData(data);
      retryCountRef.current = 0;
    } catch (err) {
      const errorMessage = err instanceof Error
        ? (err.name === 'AbortError' ? 'Request timed out' : err.message)
        : 'Failed to load';

      // Auto-retry on timeout or network error
      const currentRetry = retryCountRef.current;
      if (currentRetry < MAX_RETRIES && (err instanceof Error && (err.name === 'AbortError' || err.message.includes('fetch')))) {
        retryCountRef.current = currentRetry + 1;
        setScannerError(`${errorMessage} - Retrying... (${currentRetry + 1}/${MAX_RETRIES})`);
        // Exponential backoff: 3s, 6s
        setTimeout(() => fetchHotPicks(true), 3000 * Math.pow(2, currentRetry));
        return;
      }

      setScannerError(errorMessage);
    } finally {
      isFetchingRef.current = false;
    }
  }, [setScannerData, setScannerLoading, setScannerError]);

  // Fetch on mount and every 5 minutes
  useEffect(() => {
    fetchHotPicks(false);
    const interval = setInterval(() => fetchHotPicks(false), REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchHotPicks]);

  return { refetch: () => fetchHotPicks(false) };
}
