import React, { useState, useEffect } from 'react';

export default function PayoutLog() {
  const [payouts, setPayouts] = useState([]);

  useEffect(() => {
    fetch('/api/payouts')
      .then(res => res.json())
      .then(d => setPayouts(d))
      .catch(e => console.error("Could not fetch payouts", e));
  }, []);

  return (
    <div className="animate-in">
      <header className="header">
        <div>
          <h2>Payout Financial Log</h2>
          <p>Disbursements securely executed via RazorX / UPI Gateway</p>
        </div>
      </header>

      <div className="panel animate-in" style={{ animationDelay: '0.1s' }}>
        <h3 style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between' }}>
           <span>Recent Automated Transactions</span>
           <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 'normal' }}>Escrow Balance: <strong style={{color: 'white'}}>₹18.4L</strong></span>
        </h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Transaction Reference ID</th>
                <th>Disbursed Amount</th>
                <th>Routing Gateway</th>
                <th>Time Executed</th>
                <th>Network Status</th>
              </tr>
            </thead>
            <tbody>
              {payouts.map((p, i) => (
                <tr key={p.trx_id + i}>
                  <td style={{ fontFamily: 'monospace' }}>{p.trx_id}</td>
                  <td style={{ fontWeight: 'bold' }}>₹{p.amount.toFixed(2)}</td>
                  <td>{p.gateway}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{p.time}</td>
                  <td>
                    <span className={`badge ${p.status === 'SUCCESS' ? 'success' : 'warning'}`}>
                      {p.status}
                    </span>
                  </td>
                </tr>
              ))}
              {payouts.length === 0 && <tr><td colSpan="5">Loading real-time financial ledger...</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
