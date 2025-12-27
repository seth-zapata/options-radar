/**
 * WebSocket hook for real-time options data streaming.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useOptionsStore } from '../store/optionsStore';
import type { OptionData, UnderlyingData, WebSocketMessage, GateResult, AbstainData } from '../types';

const WS_URL = import.meta.env.PROD
  ? `wss://${window.location.host}/ws`
  : 'ws://localhost:8000/ws';

const RECONNECT_DELAY = 3000; // 3 seconds

export function useOptionsStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const {
    setConnectionStatus,
    updateOption,
    updateUnderlying,
    setAbstain,
    setGateResults,
  } = useOptionsStore();

  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionStatus('connecting');

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');

        // Attempt reconnect
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = window.setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

      ws.onmessage = (event) => {
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

            case 'error':
              console.error('Server error:', message.data);
              break;

            default:
              console.warn('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('error');
    }
  }, [setConnectionStatus, updateOption, updateUnderlying, setAbstain, setGateResults]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('disconnected');
  }, [setConnectionStatus]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return { connect, disconnect };
}
