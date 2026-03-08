import { useEffect, useRef, useState } from 'react';

export const useWebSocket = (url) => {
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setReadyState(true);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setReadyState(false);
    };

    ws.current.onmessage = (event) => {
      setLastMessage(event);
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.current.close();
    };
  }, [url]);

  const sendMessage = (message) => {
    if (ws.current && readyState) {
      ws.current.send(message);
    }
  };

  return { lastMessage, readyState, sendMessage };
};