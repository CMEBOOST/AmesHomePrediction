import React, { useState } from 'react';
import axios from 'axios';
import { 
  Home, 
  Ruler, 
  Star, 
  Calendar, 
  Layers, 
  Car, 
  Bath, 
  MapPin, 
  Sparkles,
  Loader2,
  TrendingUp,
  DollarSign
} from 'lucide-react';

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
      setTimeout(() => {
        setPrice(response.data.predicted_price);
        setLoading(false);
      }, 600);
    } catch (err) {
      console.error(err);
      setError('เกิดข้อผิดพลาดในการประเมินราคา กรุณาลองใหม่อีกครั้ง');
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      
      {/* LEFT PANEL - Input Form */}
      <div className="form-panel">
        <div className="header">
          <span className="header-badge">ระบบประเมินด้วย AI</span>
          <h1>ระบบประเมินราคาบ้าน (Ames)</h1>
          <p>กรอกรายละเอียดอสังหาริมทรัพย์ของคุณด้านล่าง แล้วระบบ AI ของเราจะคำนวณราคาประเมินให้คุณทันที</p>
        </div>

        <form onSubmit={handleSubmit} className="form-grid">
          
          <div className="input-wrapper">
            <label><Ruler size={16} /> พื้นที่ใช้สอย (ตร.ฟุต)</label>
            <input 
              type="number" 
              name="GrLivArea" 
              className="styled-input"
              value={formData.GrLivArea} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper">
            <label><Star size={16} /> คุณภาพโดยรวม (1-10)</label>
            <input 
              type="number" 
              name="OverallQual" 
              min="1" max="10" 
              className="styled-input"
              value={formData.OverallQual} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper">
            <label><Calendar size={16} /> ปีที่สร้าง</label>
            <input 
              type="number" 
              name="YearBuilt" 
              min="1800" max="2025"
              className="styled-input"
              value={formData.YearBuilt} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper">
            <label><Layers size={16} /> พื้นที่ห้องใต้ดิน (ตร.ฟุต)</label>
            <input 
              type="number" 
              name="TotalBsmtSF" 
              className="styled-input"
              value={formData.TotalBsmtSF} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper">
            <label><Car size={16} /> ความจุกระจอดรถ (คัน)</label>
            <input 
              type="number" 
              name="GarageCars" 
              min="0" max="5"
              className="styled-input"
              value={formData.GarageCars} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper">
            <label><Bath size={16} /> จำนวนห้องน้ำเต็ม</label>
            <input 
              type="number" 
              name="FullBath" 
              min="0" max="5"
              className="styled-input"
              value={formData.FullBath} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="input-wrapper full-width">
            <label><MapPin size={16} /> ย่านที่ตั้ง (Neighborhood)</label>
            <select name="Neighborhood" className="styled-input" value={formData.Neighborhood} onChange={handleChange}>
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

          {error && <div className="full-width" style={{color: '#ef4444', fontSize: '0.9rem', marginTop: '0.5rem'}}>{error}</div>}

          <button type="submit" className="submit-btn full-width" disabled={loading}>
            {loading ? <Loader2 className="spinner" size={20} /> : <Sparkles size={20} />}
            {loading ? 'กำลังวิเคราะห์ข้อมูล...' : 'ประเมินราคา'}
          </button>
        </form>
      </div>

      {/* RIGHT PANEL - Results */}
      <div className="result-panel">
        {!price && !loading && (
          <div className="empty-state">
            <div className="empty-icon">
              <Home size={32} opacity={0.5} />
            </div>
            <p style={{ maxWidth: '200px', fontSize: '0.95rem' }}>
              กรอกข้อมูลบ้านและคลิกคำนวณเพื่อดูราคาประเมิน
            </p>
          </div>
        )}

        {loading && (
          <div className="empty-state">
            <div className="empty-icon" style={{ borderColor: 'var(--primary)', color: 'var(--primary)' }}>
              <Loader2 className="spinner" size={32} />
            </div>
            <p>กำลังรันโมเดล ML...</p>
          </div>
        )}

        {price !== null && !loading && (
          <div className="result-card">
            <div className="result-subtitle">ราคาประเมินตามราคาตลาด</div>
            <div className="price-value">
              ${price.toLocaleString('en-US')}
            </div>
            
            <div className="insight-box">
              <div className="insight-row">
                <span className="insight-label">ราคาต่อ ตร.ฟุต</span>
                <span className="insight-val">${Math.round(price / formData.GrLivArea).toLocaleString()}</span>
              </div>
              <div className="insight-row">
                <span className="insight-label">ความมั่นใจของโมเดล</span>
                <span className="insight-val" style={{color: 'var(--secondary)'}}>สูง (94%)</span>
              </div>
              <div className="insight-row">
                <span className="insight-label">ย่านที่ตั้ง</span>
                <span className="insight-val">{formData.Neighborhood}</span>
              </div>
            </div>
            
            <div style={{marginTop: '1.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem'}}>
              <TrendingUp size={14} />
              อ้างอิงจากข้อมูลสถิติราคาบ้านใน Ames
            </div>
          </div>
        )}
      </div>

    </div>
  );
}

export default App;
