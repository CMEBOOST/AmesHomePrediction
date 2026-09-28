import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    GrLivArea: 1500,
    OverallQual: 6,
    YearBuilt: 2000,
    TotalBsmtSF: 1000,
    GarageCars: 2,
    FullBath: 2,
    Neighborhood: 'CollgCr'
  });
  const [price, setPrice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // In production (Vercel), point this to your Render backend URL
  // For local development, use localhost
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['Neighborhood'].includes(name) ? value : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPrice(response.data.predicted_price);
    } catch (err) {
      console.error(err);
      setError('An error occurred while predicting the price.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Ames Home Predictor</h1>
        <p>Advanced ensemble ML model to estimate your property value</p>
      </div>

      <div className="glass-panel">
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label>Living Area (sq ft)</label>
              <input 
                type="number" 
                name="GrLivArea" 
                value={formData.GrLivArea} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Overall Quality (1-10)</label>
              <input 
                type="number" 
                name="OverallQual" 
                min="1" max="10" 
                value={formData.OverallQual} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Year Built</label>
              <input 
                type="number" 
                name="YearBuilt" 
                min="1800" max="2025"
                value={formData.YearBuilt} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Total Basement (sq ft)</label>
              <input 
                type="number" 
                name="TotalBsmtSF" 
                value={formData.TotalBsmtSF} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Garage Cars</label>
              <input 
                type="number" 
                name="GarageCars" 
                min="0" max="5"
                value={formData.GarageCars} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group">
              <label>Full Bathrooms</label>
              <input 
                type="number" 
                name="FullBath" 
                min="0" max="5"
                value={formData.FullBath} 
                onChange={handleChange} 
                required 
              />
            </div>

            <div className="form-group" style={{ gridColumn: '1 / -1' }}>
              <label>Neighborhood</label>
              <select name="Neighborhood" value={formData.Neighborhood} onChange={handleChange}>
                <option value="CollgCr">College Creek</option>
                <option value="Veenker">Veenker</option>
                <option value="Crawfor">Crawford</option>
                <option value="NoRidge">Northridge</option>
                <option value="Mitchel">Mitchell</option>
                <option value="Somerst">Somerset</option>
                <option value="NWAmes">Northwest Ames</option>
                <option value="OldTown">Old Town</option>
                <option value="BrkSide">Brookside</option>
                <option value="Sawyer">Sawyer</option>
                <option value="NridgHt">Northridge Heights</option>
                <option value="NAmes">North Ames</option>
                <option value="SawyerW">Sawyer West</option>
                <option value="IDOTRR">Iowa DOT and Rail Road</option>
                <option value="MeadowV">Meadow Village</option>
                <option value="Edwards">Edwards</option>
                <option value="Timber">Timberland</option>
                <option value="Gilbert">Gilbert</option>
                <option value="StoneBr">Stone Brook</option>
                <option value="ClearCr">Clear Creek</option>
                <option value="NPkVill">Northpark Villa</option>
                <option value="Blmngtn">Bloomington Heights</option>
                <option value="BrDale">Briardale</option>
                <option value="SWISU">South & West of Iowa State Univ.</option>
                <option value="Blueste">Bluestem</option>
              </select>
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict Property Value'}
          </button>
        </form>

        {error && <p style={{color: '#ff4d4d', marginTop: '1rem', textAlign: 'center'}}>{error}</p>}

        {price !== null && !loading && (
          <div className="result-container">
            <h3>Estimated Market Value</h3>
            <div className="price-display">
              ${price.toLocaleString('en-US')}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
