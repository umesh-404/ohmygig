import React, { useState, useEffect } from 'react';

const mockFeed = [
  { id: 'CLM-0182', worker: 'ZOM-8812', zone: 'Kukatpally', cvs: 0.98, status: 'INSTANT_PAYOUT', time: 'Just now' },
  { id: 'CLM-0181', worker: 'SWG-1102', zone: 'HITEC City', cvs: 0.45, status: 'LIVENESS_CHECK', time: '2m ago' },
  { id: 'CLM-0180', worker: 'ZOM-9921', zone: 'Gachibowli', cvs: 0.12, status: 'ADMIN_REVIEW', time: '5m ago' },
  { id: 'CLM-0179', worker: 'ZOM-4412', zone: 'Secunderabad', cvs: 0.88, status: 'INSTANT_PAYOUT', time: '8m ago' }
];

export default function Dashboard() {
  const [liveFeed, setLiveFeed] = useState(mockFeed);
  const [apiResponse, setApiResponse] = useState(null);
  const [isInjecting, setIsInjecting] = useState(false);
  const [metrics, setMetrics] = useState({
      aggregate_loss_ratio: '...',
      daily_premium_pool: '...',
      fraud_ring_intercepts: '...'
  });

  useEffect(() => {
    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error("Metrics load failed", err));
  }, []);

  const triggerBackendEvaluation = async () => {
    setIsInjecting(true);
    try {
      const resp = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer 12345'
        },
        body: JSON.stringify({
          worker_id: `ZOM-${Math.floor(1000 + Math.random() * 9000)}`,
          zone_id: "hyderabad_test",
          is_mock_location_flag: false,
          gps_cell_tower_delta_km: 0.5,
          accelerometer_variance: 5.4,
          aggregator_app_inactive_hours: 0.1,
          syndicate_cluster_density: 2
        })
      });
      
      const data = await resp.json();
      setApiResponse(data);
      
      const newClaim = {
        id: `CLM-1${Math.floor(100 + Math.random() * 900)}`,
        worker: data.worker_id,
        zone: 'API Triggered',
        cvs: data.fraud_engine_response.cvs_score,
        status: data.fraud_engine_response.action_required,
        time: 'Just now'
      };
      
      setLiveFeed(prev => [newClaim, ...prev.slice(0, 4)]);
    } catch (err) {
      console.error("API Gateway unreachable.", err);
    } finally {
      setIsInjecting(false);
    }
  };

  return (
    <>
      <header className="header animate-in" style={{ animationDelay: '0.1s' }}>
        <div>
          <h2>Command Center</h2>
          <p>Live Actuarial System & Risk Monitoring <span className="live-indicator" style={{ marginLeft: '8px' }}></span></p>
        </div>
        <div>
           <button onClick={triggerBackendEvaluation} disabled={isInjecting} className="api-test-btn">
             {isInjecting ? 'Processing...' : 'Inject Test Claim (POST /api/evaluate)'}
           </button>
        </div>
      </header>

      <section className="metrics-grid animate-in" style={{ animationDelay: '0.2s' }}>
        <div className="panel metric-card">
          <div className="metric-title">Aggregate Loss Ratio</div>
          <div className="metric-value">{metrics.aggregate_loss_ratio}{typeof metrics.aggregate_loss_ratio === 'number' && '%'}</div>
          <div className="metric-trend trend-success">
             🎯 Active (Target: 55-65%)
          </div>
        </div>
        <div className="panel metric-card">
          <div className="metric-title">Daily Premium Pool</div>
          <div className="metric-value">₹{typeof metrics.daily_premium_pool === 'number' ? (metrics.daily_premium_pool / 100000).toFixed(1) + 'L' : '...'}</div>
          <div className="metric-trend trend-up">
            ↑ +2.4% vs last week
          </div>
        </div>
        <div className="panel metric-card">
          <div className="metric-title">Fraud Ring Intercepts</div>
          <div className="metric-value">{metrics.fraud_ring_intercepts}</div>
          <div className="metric-trend trend-down">
            GNN Network Defended
          </div>
        </div>
      </section>

      <div className="content-grid animate-in" style={{ animationDelay: '0.3s' }}>
        <div className="panel">
          <h3 style={{ marginBottom: '1.5rem' }}>Live Claim Validity Feed</h3>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Claim ID</th>
                  <th>Worker</th>
                  <th>Zone</th>
                  <th>CVS Score</th>
                  <th>Decision</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {liveFeed.map((claim, idx) => (
                  <tr key={claim.id + idx}>
                    <td><strong style={{ color: '#c4b5fd' }}>{claim.id}</strong></td>
                    <td>{claim.worker}</td>
                    <td>{claim.zone}</td>
                    <td>
                      <span style={{ color: claim.cvs > 0.75 ? 'var(--success)' : claim.cvs > 0.3 ? 'var(--warning)' : 'var(--danger)' }}>
                         {(claim.cvs * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${
                        claim.status === 'INSTANT_PAYOUT' ? 'success' : 
                        claim.status === 'ADMIN_REVIEW' ? 'danger' : 'warning'
                      }`}>
                        {claim.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{claim.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
           <h3 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between' }}>
             <span>API Gateway Monitor</span>
             <span className="live-indicator"></span>
           </h3>
           <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
             Intercepting live payload responses from the Fast API Back-end.
           </p>
           
           {apiResponse ? (
              <pre className="json-response">
                {JSON.stringify(apiResponse, null, 2)}
              </pre>
           ) : (
              <div style={{ marginTop: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                 No endpoints triggered yet.<br/><br/>Click "Inject Test Claim" to POST payload.
              </div>
           )}
        </div>
      </div>
    </>
  );
}
