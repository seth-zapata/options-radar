/**
 * WebSocket hook for real-time options data streaming.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useOptionsStore, EvaluatedOption } from '../store/optionsStore';
import type { OptionData, UnderlyingData, WebSocketMessage, GateResult, AbstainData, Recommendation, SessionStatus, TrackedPosition, ExitSignal, RegimeStatus, RegimeSignal } from '../types';

const WS_URL = import.meta.env.PROD
  ? `wss://${window.location.host}/ws`
  : 'ws://localhost:8000/ws';

const RECONNECT_DELAY = 5000; // 5 seconds
const PING_INTERVAL = 25000; // 25 seconds
const DISCONNECT_GRACE_PERIOD = 3000; // Only show disconnected after 3 seconds

export function useOptionsStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const disconnectGraceRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const currentSymbolRef = useRef<string | null>(null);

  // Get store methods and state
  const setConnectionStatus = useOptionsStore((state) => state.setConnectionStatus);
  const updateOption = useOptionsStore((state) => state.updateOption);
  const updateUnderlying = useOptionsStore((state) => state.updateUnderlying);
  const setAbstain = useOptionsStore((state) => state.setAbstain);
  const setGateResults = useOptionsStore((state) => state.setGateResults);
  const addRecommendation = useOptionsStore((state) => state.addRecommendation);
  const setSessionStatus = useOptionsStore((state) => state.setSessionStatus);
  const addPosition = useOptionsStore((state) => state.addPosition);
  const updatePosition = useOptionsStore((state) => state.updatePosition);
  const addExitSignal = useOptionsStore((state) => state.addExitSignal);
  const clearExitSignal = useOptionsStore((state) => state.clearExitSignal);
  const setRegimeStatus = useOptionsStore((state) => state.setRegimeStatus);
  const addRegimeSignal = useOptionsStore((state) => state.addRegimeSignal);
  const activeSymbol = useOptionsStore((state) => state.activeSymbol);

  // Send subscribe message to switch symbols
  const sendSubscribe = useCallback((symbol: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        symbol: symbol
      }));
      currentSymbolRef.current = symbol;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const cleanup = () => {
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (disconnectGraceRef.current) {
        clearTimeout(disconnectGraceRef.current);
        disconnectGraceRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };

    const connect = (isInitial = false) => {
      if (!mountedRef.current) return;

      // Clean up existing connection but keep grace timer running
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      // Only show "connecting" on initial connection
      if (isInitial) {
        setConnectionStatus('connecting');
      }

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mountedRef.current) return;

          // Cancel grace period timer - we're connected
          if (disconnectGraceRef.current) {
            clearTimeout(disconnectGraceRef.current);
            disconnectGraceRef.current = null;
          }

          console.log('WebSocket connected');
          setConnectionStatus('connected');

          // Start ping interval to keep connection alive
          pingIntervalRef.current = window.setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, PING_INTERVAL);
        };

        ws.onclose = (event) => {
          if (!mountedRef.current) return;
          console.log('WebSocket closed:', event.code, event.reason);

          // Clear ping interval
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }

          // Start grace period timer - only show disconnected after delay
          if (!disconnectGraceRef.current) {
            disconnectGraceRef.current = window.setTimeout(() => {
              if (mountedRef.current) {
                setConnectionStatus('disconnected');
              }
              disconnectGraceRef.current = null;
            }, DISCONNECT_GRACE_PERIOD);
          }

          // Attempt reconnect only if still mounted
          if (mountedRef.current) {
            reconnectTimeoutRef.current = window.setTimeout(() => connect(false), RECONNECT_DELAY);
          }
        };

        ws.onerror = (error) => {
          if (!mountedRef.current) return;
          console.error('WebSocket error:', error);
          setConnectionStatus('error');
        };

        ws.onmessage = (event) => {
          if (!mountedRef.current) return;

          try {
            const message: WebSocketMessage = JSON.parse(event.data);

            switch (message.type) {
              case 'option_update':
                updateOption(message.data as OptionData);
                break;

              case 'underlying_update':
                updateUnderlying(message.data as UnderlyingData);
                break;

              case 'gate_status': {
                const gateData = message.data as { gates: GateResult[]; evaluatedOption?: EvaluatedOption };
                setGateResults(gateData.gates, gateData.evaluatedOption);
                break;
              }

              case 'abstain':
                setAbstain(message.data as AbstainData);
                break;

              case 'recommendation':
                addRecommendation(message.data as Recommendation);
                break;

              case 'session_status':
                setSessionStatus(message.data as SessionStatus);
                break;

              case 'position_opened':
                addPosition(message.data as TrackedPosition);
                break;

              case 'position_closed': {
                // Backend sends {position: {...}, order: {...}} - extract the position
                const closedData = message.data as { position: TrackedPosition };
                updatePosition(closedData.position);
                // Clear any exit signals for this position since it's now closed
                clearExitSignal(closedData.position.id);
                break;
              }

              case 'exit_signal':
                addExitSignal(message.data as ExitSignal);
                break;

              case 'position_updated':
                updatePosition(message.data as TrackedPosition);
                break;

              case 'connection_status':
                // Server confirmed connection
                break;

              case 'ping':
                // Respond to server ping
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send('pong');
                }
                break;

              case 'error':
                console.error('Server error:', message.data);
                break;

              case 'symbol_changed':
                // Server confirmed symbol switch
                console.log('Symbol changed to:', (message.data as { symbol: string }).symbol);
                break;

              case 'regime_status':
                setRegimeStatus(message.data as RegimeStatus);
                break;

              case 'regime_signal':
                addRegimeSignal(message.data as RegimeSignal);
                break;

              default:
                // Ignore unknown message types silently
                break;
            }
          } catch {
            // Ignore parse errors for ping/pong text messages
          }
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        setConnectionStatus('error');
      }
    };

    connect(true); // Initial connection shows "connecting"

    return () => {
      mountedRef.current = false;
      cleanup();
      setConnectionStatus('disconnected');
    };
  }, [setConnectionStatus, updateOption, updateUnderlying, setAbstain, setGateResults, addRecommendation, setSessionStatus, addPosition, updatePosition, addExitSignal, clearExitSignal, setRegimeStatus, addRegimeSignal]);

  // Handle symbol changes - send subscribe message to backend
  useEffect(() => {
    // Skip if same symbol or not connected yet
    if (currentSymbolRef.current === activeSymbol) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // Store for when we connect
      currentSymbolRef.current = activeSymbol;
      return;
    }

    // Send subscribe message to switch symbols
    sendSubscribe(activeSymbol);
  }, [activeSymbol, sendSubscribe]);
}
