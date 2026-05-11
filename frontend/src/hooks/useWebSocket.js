import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

export const useWebSocket = (url) => {
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    console.log('Initializing WebSocket connection to:', url);
    
    socketRef.current = io(url, {
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socketRef.current.on('connect', () => {
      console.log('Socket.IO connected successfully');
      setReadyState(true);
    });

    socketRef.current.on('disconnect', (reason) => {
      console.log('Socket.IO disconnected:', reason);
      setReadyState(false);
    });

    socketRef.current.onAny((event, payload) => {
      console.log('WebSocket event received:', event);
      setLastMessage({ event, payload });
    });

    socketRef.current.on('connect_error', (error) => {
      console.error('Socket.IO connect error:', error);
    });

    socketRef.current.on('error', (error) => {
      console.error('Socket.IO error:', error);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [url]);

  const sendMessage = (event, payload) => {
    if (socketRef.current && readyState) {
      console.log('Sending WebSocket message:', event);
      socketRef.current.emit(event, payload);
    } else {
      console.warn('WebSocket not ready. Event not sent:', event);
    }
  };

  return { lastMessage, readyState, sendMessage };
};