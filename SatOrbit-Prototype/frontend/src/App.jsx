import React, { useState, useEffect } from 'react';
import { fetchAOIs, runChangeDetection } from './services/api';

const API_BASE_URL = 'http://localhost:8000';

// Fallback list ensuring Hinjewadi, Bhosari, Chakan, and Rajgurunagar display immediately
const DEFAULT_AOIS = [
  { id: 'hinjewadi', name: 'Hinjewadi', default_dates: { date1: '2024-02-28', date2: '2026-04-03' } },
  { id: 'bhosari', name: 'Bhosari', default_dates: { date1: '2024-02-28', date2: '2026-04-03' } },
  { id: 'chakan', name: 'Chakan', default_dates: { date1: '2024-02-28', date2: '2026-04-03' } },
  { id: 'rajgurunagar', name: 'Rajgurunagar', default_dates: { date1: '2024-02-28', date2: '2026-04-03' } },
];

export default function App() {
  const [aois, setAois] = useState(DEFAULT_AOIS);
  const [selectedAoi, setSelectedAoi] = useState('hinjewadi');
  const [date1, setDate1] = useState('2024-02-28');
  const [date2, setDate2] = useState('2026-04-03');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Load available AOIs on initial render
  useEffect(() => {
    fetchAOIs()
      .then((data) => {
        let parsedAois = [];
        if (Array.isArray(data)) {
          parsedAois = data;
        } else if (data && typeof data === 'object') {
          // Map backend object response { key: { name, default_dates } } to array format
          parsedAois = Object.keys(data).map((key) => ({
            id: key,
            name: data[key].name || key.charAt(0).toUpperCase() + key.slice(1),
            default_dates: data[key].default_dates || { date1: '2024-02-28', date2: '2026-04-03' },
          }));
        }

        if (parsedAois.length > 0) {
          setAois(parsedAois);
          const firstAoi = parsedAois[0];
          setSelectedAoi(firstAoi.id);
          if (firstAoi.default_dates) {
            setDate1(firstAoi.default_dates.date1 || '2024-02-28');
            setDate2(firstAoi.default_dates.date2 || '2026-04-03');
          }
        }
      })
      .catch((err) => {
        console.error('Failed to load AOIs from backend, using default list:', err);
      });
  }, []);

  // Update default dates whenever selected AOI changes
  const handleAoiChange = (e) => {
    const aoiId = e.target.value;
    setSelectedAoi(aoiId);
    const targetAoi = aois.find((a) => a.id === aoiId);
    if (targetAoi && targetAoi.default_dates) {
      const d1 = targetAoi.default_dates.date1 || (Array.isArray(targetAoi.default_dates) ? targetAoi.default_dates[0] : '2024-02-28');
      const d2 = targetAoi.default_dates.date2 || (Array.isArray(targetAoi.default_dates) ? targetAoi.default_dates[1] : '2026-04-03');
      setDate1(d1);
      setDate2(d2);
    }
  };

  const handleRunPipeline = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await runChangeDetection(selectedAoi, date1, date2);
      setResults(data);
    } catch (err) {
      console.error('Pipeline execution failed:', err);
      setError(err.message || 'Error processing satellite imagery. Check backend logs.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>SatOrbit Satellite Change Detection</h1>

      {/* Error Alert Box */}
      {error && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '6px',
          marginBottom: '20px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Control Bar */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
        <select
          value={selectedAoi}
          onChange={handleAoiChange}
          style={{ padding: '8px 12px', borderRadius: '4px', border: '1px solid #ccc' }}
        >
          {aois.map((aoi) => (
            <option key={aoi.id} value={aoi.id}>{aoi.name}</option>
          ))}
        </select>

        <label style={{ fontSize: '14px' }}>Date 1 (Before):</label>
        <input
          type="date"
          value={date1}
          onChange={(e) => setDate1(e.target.value)}
          style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        />

        <label style={{ fontSize: '14px' }}>Date 2 (After):</label>
        <input
          type="date"
          value={date2}
          onChange={(e) => setDate2(e.target.value)}
          style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        />

        <button
          onClick={handleRunPipeline}
          disabled={loading || !selectedAoi}
          style={{
            padding: '8px 20px',
            borderRadius: '4px',
            backgroundColor: loading ? '#6c757d' : '#0d6efd',
            color: '#fff',
            border: 'none',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          {loading ? 'Fetching & Processing...' : 'Run Pipeline'}
        </button>
      </div>

      {/* Results Workspace */}
      {results && (
        <div style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
          <h2>Results Overview ({results.aoi_id?.toUpperCase()})</h2>

          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '15px' }}>
            <p><strong>Change Percentage:</strong> {results.change_percentage}%</p>
            <p><strong>Total Changed Area:</strong> {results.changed_area_km2} km²</p>
            <p><strong>New Construction:</strong> {results.classification?.new_construction_km2 ?? 'N/A'} km²</p>
            <p><strong>Vegetation Loss:</strong> {results.classification?.vegetation_loss_km2 ?? 'N/A'} km²</p>
            <p><strong>Surface Water:</strong> {results.classification?.surface_water_km2 ?? 'N/A'} km²</p>
          </div>

          <hr style={{ margin: '20px 0', borderColor: '#ddd' }} />

          {/* Section 1: Raw vs Processed Input Images */}
          <h3>1. Raw and Processed Satellite Imagery</h3>
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '25px' }}>
            <div>
              <h4>Raw Image ({results.date1})</h4>
              <img
                src={`${results.raw_image_url}`}
                alt="Raw Satellite Asset"
                style={{ width: '360px', height: '360px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #ccc', backgroundColor: '#fff' }}
              />
            </div>
            <div>
              <h4>Processed Image ({results.date2})</h4>
              <img
                src={`${results.processed_image_url}`}
                alt="Processed Satellite Asset"
                style={{ width: '360px', height: '360px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #ccc', backgroundColor: '#fff' }}
              />
            </div>
          </div>

          {/* Section 2: Change Map Masks and Analytics */}
          <h3>2. Change Detection Analytics & Visuals</h3>
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '25px' }}>
            <div>
              <h4>Cleaned Change Mask</h4>
              <img
                src={`${results.clean_map_url}`}
                alt="Cleaned Mask"
                style={{ width: '360px', height: '360px', objectFit: 'contain', borderRadius: '6px', border: '1px solid #ccc', backgroundColor: '#000' }}
              />
            </div>
            <div>
              <h4>Side-by-Side Comparison Plot</h4>
              <img
                src={`${results.comparison_url}`}
                alt="Comparison Figure"
                style={{ maxWidth: '600px', width: '100%', borderRadius: '6px', border: '1px solid #ccc', backgroundColor: '#fff' }}
              />
            </div>
          </div>

          {/* Section 3: Interactive GIS Map */}
          {results.map_url && (
            <>
              <hr style={{ margin: '20px 0', borderColor: '#ddd' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h3 style={{ margin: 0 }}>3. Interactive GIS Map Layer</h3>
                <a
                  href={results.map_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: '13px',
                    color: '#0d6efd',
                    textDecoration: 'none',
                    fontWeight: 'bold',
                  }}
                >
                  Open Full Map in New Tab ↗
                </a>
              </div>
              <div style={{ width: '100%', height: '550px', border: '1px solid #ccc', borderRadius: '6px', overflow: 'hidden', backgroundColor: '#fff' }}>
                <iframe
                  src={results.map_url}
                  title="Interactive Satellite Map"
                  style={{ width: '100%', height: '100%', border: 'none' }}
                />
              </div>
            </>
          )}

        </div>
      )}
    </div>
  );
}