import React, { useState, useEffect } from 'react';

export default function FraudDetection() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetch('/api/fraud')
      .then(res => res.json())
      .then(d => setLogs(d))
      .catch(e => console.error("Could not fetch fraud logs", e));
  }, []);

  return (
    <div className="animate-in">
      <header className="header">
        <div>
          <h2>Fraud Detection Engine</h2>
          <p>Real-time GNN & Isolation Forest Interception Logs</p>
        </div>
      </header>

      <div className="metrics-grid">
        <div className="panel metric-card" style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
          <div className="metric-title">GNN Syndicate Intercepts</div>
          <div className="metric-value">412 Disrupted</div>
          <div className="metric-trend trend-down">Last 24 Hours</div>
        </div>
        <div className="panel metric-card" style={{ borderColor: 'rgba(245, 158, 11, 0.3)' }}>
          <div className="metric-title">FakeGPS Blocked Returns</div>
          <div className="metric-value">84 APIs</div>
          <div className="metric-trend trend-down">Zero Trust Framework</div>
        </div>
        <div className="panel metric-card" style={{ padding: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
             ML Model Status:<br/><span style={{ color: 'var(--success)', fontWeight: 'bold' }}>ONLINE</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3 style={{ marginBottom: '1.5rem' }}>Automated Fraud Audits</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Audit ID</th>
                <th>Worker Entity</th>
                <th>CVS Output</th>
                <th>Rejection Logic</th>
                <th>Policy Action Taken</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, i) => (
                <tr key={log.id + i}>
                  <td><strong>{log.id}</strong></td>
                  <td>{log.worker}</td>
                  <td>
                    <span style={{ color: log.cvs_score > 0.75 ? 'var(--success)' : log.cvs_score > 0.3 ? 'var(--warning)' : 'var(--danger)' }}>
                      {(log.cvs_score * 100).toFixed(1)}% Confidence
                    </span>
                  </td>
                  <td><span style={{ fontStyle: log.reason === 'Verified' ? 'normal' : 'italic' }}>{log.reason}</span></td>
                  <td>
                    <span className={`badge ${
                      log.status === 'INSTANT_PAYOUT' ? 'success' : 
                      log.status === 'DENIED' ? 'danger' : 'warning'
                    }`}>
                      {log.status.replace('_', ' ')}
                    </span>
                  </td>
                </tr>
              ))}
              {logs.length === 0 && <tr><td colSpan="5">Loading real-time fraud data...</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
