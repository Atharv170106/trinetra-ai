import React, { useEffect, useState } from "react";
import { api } from "../api/client";

export default function TriageModal({ onAcknowledge, onFalseAlarm }) {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // SSE Stream for watchdog alerts
    const sseUrl = `${api.baseUrl}/watchdog/stream`;
    const eventSource = new EventSource(sseUrl);

    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setAlerts((prev) => [...prev, data]);
      } catch (err) {
        console.error("Failed to parse SSE data", err);
      }
    };

    eventSource.onerror = (e) => {
      // Handle connection errors gracefully without breaking the UI
      console.error("SSE connection error", e);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  if (alerts.length === 0) return null;

  const currentAlert = alerts[0];

  const handleAcknowledge = () => {
    if (onAcknowledge) onAcknowledge(currentAlert);
    setAlerts((prev) => prev.slice(1));
  };

  const handleFalseAlarm = () => {
    if (onFalseAlarm) onFalseAlarm(currentAlert);
    setAlerts((prev) => prev.slice(1));
  };

  return (
    <div className="modal-overlay" style={{
      position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', 
      backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 9999, display: 'flex', 
      justifyContent: 'center', alignItems: 'center'
    }}>
      <div className="modal-content" style={{
        backgroundColor: 'var(--surface)', padding: '2rem', borderRadius: '8px',
        maxWidth: '500px', border: currentAlert.severity === 'critical' ? '2px solid var(--red)' : '2px solid var(--amber)'
      }}>
        <h2 style={{ color: currentAlert.severity === 'critical' ? 'var(--red)' : 'var(--amber)' }}>
          ⚠️ Watchdog Alert: {currentAlert.severity.toUpperCase()}
        </h2>
        <p><strong>Scene:</strong> {currentAlert.scene}</p>
        <p><strong>Reason:</strong> {currentAlert.reason}</p>
        
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
          <button className="btn primary" style={{ backgroundColor: 'var(--green)' }} onClick={handleAcknowledge}>
            Acknowledge
          </button>
          <button className="btn" style={{ backgroundColor: 'var(--red)', color: 'white' }} onClick={handleFalseAlarm}>
            False Alarm
          </button>
        </div>
      </div>
    </div>
  );
}
