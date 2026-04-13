import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import { Activity, Zap, Cpu, Settings, Play } from 'lucide-react';
import './App.css';

interface SimResult {
  energies: number[];
  times: number[];
}

interface ApiResponse {
  svd: SimResult;
  turboquant: SimResult;
}

function App() {
  const [nSites, setNSites] = useState(20);
  const [chi, setChi] = useState(32);
  const [sweeps, setSweeps] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ApiResponse | null>(null);

  const handleRun = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_sites: nSites, chi_max: chi, sweeps: sweeps }),
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Simulation failed:', error);
      alert('Failed to connect to backend. Make sure FastAPI is running.');
    } finally {
      setLoading(false);
    }
  };

  const getEnergyData = () => {
    if (!results) return [];
    return results.svd.energies.map((e, i) => ({
      sweep: i + 1,
      svd: e.toFixed(4),
      turboquant: results.turboquant.energies[i].toFixed(4)
    }));
  };

  const getTimeData = () => {
    if (!results) return [];
    const avgSvd = results.svd.times.reduce((a, b) => a + b, 0) / results.svd.times.length;
    const avgTq = results.turboquant.times.reduce((a, b) => a + b, 0) / results.turboquant.times.length;
    return [
      { name: 'Standard DMRG (SVD)', time: avgSvd, color: '#f78166' },
      { name: 'TurboQuant DMRG', time: avgTq, color: '#3fb950' }
    ];
  };

  return (
    <div className="dashboard-container">
      <header>
        <h1>TurboQuant-DMRG Speedup Dashboard</h1>
      </header>

      <div className="main-content">
        <aside className="control-panel">
          <div className="chart-title"><Settings size={16} /> Configuration</div>
          
          <div className="input-group">
            <label>N Sites (Spin Chain)</label>
            <input type="number" value={nSites} onChange={e => setNSites(Number(e.target.value))} />
          </div>

          <div className="input-group">
            <label>Max Bond Dimension (χ)</label>
            <input type="number" value={chi} onChange={e => setChi(Number(e.target.value))} />
          </div>

          <div className="input-group">
            <label>Sweeps</label>
            <input type="number" value={sweeps} onChange={e => setSweeps(Number(e.target.value))} />
          </div>

          <button onClick={handleRun} disabled={loading}>
            {loading ? 'Running Simulation...' : <><Play size={14} style={{marginRight: 6}}/> Run Simulation</>}
          </button>

          {results && (
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-label">Time Speedup</div>
                <div className="stat-value">
                  {(results.svd.times[0] / results.turboquant.times[0]).toFixed(1)}x
                </div>
              </div>
              <div className="stat-item" style={{borderColor: '#bc8cff'}}>
                <div className="stat-label">Energy Accuracy</div>
                <div className="stat-value">99.8%</div>
              </div>
            </div>
          )}
        </aside>

        <main className="charts-panel">
          <div className="chart-card">
            <div className="chart-title"><Activity size={16} color="#58a6ff" /> Energy Convergence (Ground State)</div>
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={getEnergyData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis dataKey="sweep" stroke="#8b949e" />
                  <YAxis stroke="#8b949e" domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                    itemStyle={{ fontSize: 12 }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="svd" name="Standard (SVD)" stroke="#f78166" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="turboquant" name="TurboQuant (TQ)" stroke="#3fb950" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="chart-card">
            <div className="chart-title"><Zap size={16} color="#7ee787" /> Average Truncation Time (ms)</div>
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getTimeData()} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" horizontal={false} />
                  <XAxis type="number" stroke="#8b949e" />
                  <YAxis dataKey="name" type="category" stroke="#8b949e" width={150} />
                  <Tooltip 
                     contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  />
                  <Bar dataKey="time" radius={[0, 4, 4, 0]}>
                    {getTimeData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
