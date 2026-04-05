import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, LineChart, Line } from 'recharts';

export default function RegionalMatrix() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/api/matrix')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(e => console.error("Could not fetch matrix", e));
  }, []);

  return (
    <div className="animate-in">
      <header className="header">
        <div>
          <h2>Regional Matrix</h2>
          <p>Live Actuarial Weather Trigger Mapping across Hyderabad</p>
        </div>
      </header>

      <div className="metrics-grid">
        <div className="panel metric-card" style={{ gridColumn: 'span 2' }}>
           <h3 style={{ marginBottom: '1rem' }}>Zonal Rain Levels & Multiplier Forecasting</h3>
           <div style={{ width: '100%', height: 300 }}>
             <ResponsiveContainer>
               <BarChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="zone" fill="#94a3b8" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8b5cf6" />
                  <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
                  <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: 'rgba(255,255,255,0.1)' }} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="rainLevel_mm" fill="url(#colorRain)" name="Live Rain Level (mm)" radius={[4, 4, 0, 0]} />
                  <Bar yAxisId="right" dataKey="riskMultiplier" fill="#10b981" name="Risk Multiplier (LSTM Output)" radius={[4, 4, 0, 0]} />
                  <defs>
                     <linearGradient id="colorRain" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.2}/>
                     </linearGradient>
                  </defs>
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>

      <div className="panel">
        <h3 style={{ marginBottom: '1rem' }}>Live Zonal Claims Data</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Zone Name</th>
                <th>Avg Rain Level</th>
                <th>Risk Multiplier</th>
                <th>Active Claims Processed</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d, i) => (
                <tr key={i}>
                  <td><strong>{d.zone}</strong></td>
                  <td>{d.rainLevel_mm} mm</td>
                  <td>{d.riskMultiplier.toFixed(2)}x</td>
                  <td>{d.claims} API Events</td>
                  <td>
                    <span className={`badge ${d.riskMultiplier > 2 ? 'danger' : d.riskMultiplier > 1.4 ? 'warning' : 'success'}`}>
                      {d.riskMultiplier > 2 ? 'High Risk' : d.riskMultiplier > 1.4 ? 'Elevated' : 'Nominal'}
                    </span>
                  </td>
                </tr>
              ))}
              {data.length === 0 && <tr><td colSpan="5">Loading real-time matrix data from Backend...</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
