/**
 * WebSocket hook for real-time options data streaming.
 */

import { useEffect, useRef } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { OptionData, UnderlyingData, WebSocketMessage, GateResult, AbstainData } from '../types';

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

  // Get store methods once (they're stable)
  const setConnectionStatus = useOptionsStore((state) => state.setConnectionStatus);
  const updateOption = useOptionsStore((state) => state.updateOption);
  const updateUnderlying = useOptionsStore((state) => state.updateUnderlying);
  const setAbstain = useOptionsStore((state) => state.setAbstain);
  const setGateResults = useOptionsStore((state) => state.setGateResults);

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

              case 'gate_status':
                setGateResults((message.data as { gates: GateResult[] }).gates);
                break;

              case 'abstain':
                setAbstain(message.data as AbstainData);
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
  }, [setConnectionStatus, updateOption, updateUnderlying, setAbstain, setGateResults]);
}
