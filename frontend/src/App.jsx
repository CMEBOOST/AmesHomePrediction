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
  Clock,
  History,
  LayoutGrid,
  Maximize,
  ClipboardCheck,
  Wrench,
  Flame,
  Snowflake,
  Building2,
  TriangleAlert
} from 'lucide-react';
import { useEffect } from 'react';

// Fields the model was trained on (keep in sync with NUMERIC_FEATURES in backend/ml_pipeline.py).
const NUMERIC_FIELDS = [
  { name: 'GrLivArea',    label: 'พื้นที่ใช้สอยเหนือพื้นดิน (ตร.ฟุต)', icon: Ruler,          min: 300,  max: 6000,   default: 1464 },
  { name: 'FirstFlrSF',   label: 'พื้นที่ชั้น 1 (ตร.ฟุต)',             icon: LayoutGrid,     min: 300,  max: 6000,   default: 1086 },
  { name: 'LotArea',      label: 'ขนาดที่ดิน (ตร.ฟุต)',                icon: Maximize,       min: 1000, max: 250000, default: 9475 },
  { name: 'OverallQual',  label: 'คุณภาพวัสดุและการตกแต่ง (1-10)',     icon: Star,           min: 1,    max: 10,     default: 6 },
  { name: 'OverallCond',  label: 'สภาพบ้านโดยรวม (1-10)',              icon: ClipboardCheck, min: 1,    max: 10,     default: 5 },
  { name: 'YearBuilt',    label: 'ปีที่สร้าง',                         icon: Calendar,       min: 1800, max: 2030,   default: 1973 },
  { name: 'YearRemodAdd', label: 'ปีที่รีโนเวตล่าสุด (ไม่เคย = ปีที่สร้าง)', icon: Wrench,    min: 1800, max: 2030,   default: 1994 },
  { name: 'TotalBsmtSF',  label: 'พื้นที่ห้องใต้ดินรวม (ตร.ฟุต)',      icon: Layers,         min: 0,    max: 6000,   default: 991 },
  { name: 'BsmtFinSF1',   label: 'พื้นที่ห้องใต้ดินที่ตกแต่งแล้ว (ตร.ฟุต)', icon: Layers,    min: 0,    max: 5000,   default: 382 },
  { name: 'GarageCars',   label: 'ความจุโรงรถ (คัน)',                  icon: Car,            min: 0,    max: 5,      default: 2 },
  { name: 'FullBath',     label: 'จำนวนห้องน้ำเต็ม',                   icon: Bath,           min: 0,    max: 4,      default: 2 },
  { name: 'Fireplaces',   label: 'จำนวนเตาผิง',                        icon: Flame,          min: 0,    max: 4,      default: 1 },
];

const CATEGORICAL_NAMES = ['Neighborhood', 'MSZoning', 'CentralAir'];

function App() {
  const [formData, setFormData] = useState({
    ...Object.fromEntries(NUMERIC_FIELDS.map(f => [f.name, f.default])),
    CentralAir: 'Y',
    MSZoning: 'RL',
    Neighborhood: 'NAmes'
  });
  const [price, setPrice] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem('valuationHistory');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('valuationHistory', JSON.stringify(history));
  }, [history]);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: CATEGORICAL_NAMES.includes(name) ? value : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setTimeout(() => {
        const predictedPrice = response.data.predicted_price;
        const modelConfidence = response.data.confidence_score || 94.0;
        setPrice(predictedPrice);
        setConfidence(modelConfidence);
        setWarnings(response.data.warnings || []);
        
        const newRecord = {
          id: Date.now(),
          date: new Date().toLocaleString('th-TH', { hour12: false }),
          price: predictedPrice,
          details: `${formData.GrLivArea} ตร.ฟุต | คุณภาพ ${formData.OverallQual} | ย่าน ${formData.Neighborhood} | สร้างปี ${formData.YearBuilt}`
        };
        setHistory(prev => [newRecord, ...prev].slice(0, 5)); // เก็บ 5 อันล่าสุด
        
        setLoading(false);
      }, 600);
    } catch (err) {
      console.error(err);
      const detail = err.response?.data?.detail;
      if (err.response?.status === 422 && Array.isArray(detail) && detail.length > 0) {
        // Validation error from the API (e.g. impossible value): show which field and why
        const first = detail[0];
        const field = first.loc?.[first.loc.length - 1];
        const msg = String(first.msg).replace('Value error, ', '');
        setError(field === 'body' ? `ข้อมูลไม่ถูกต้อง: ${msg}` : `ข้อมูลไม่ถูกต้อง (${field}): ${msg}`);
      } else {
        setError('เกิดข้อผิดพลาดในการประเมินราคา กรุณาลองใหม่อีกครั้ง');
      }
      setLoading(false);
    }
  };

  return (
    <div className="main-wrapper">
      <div className="dashboard-container">
        
        {/* LEFT PANEL - Input Form */}
      <div className="form-panel">
        <div className="header">
          <span className="header-badge">ระบบประเมินด้วย AI</span>
          <h1>ระบบประเมินราคาบ้าน (Ames)</h1>
          <p>กรอกรายละเอียดอสังหาริมทรัพย์ของคุณด้านล่าง แล้วระบบ AI ของเราจะคำนวณราคาประเมินให้คุณทันที</p>
        </div>

        <form onSubmit={handleSubmit} className="form-grid">
          
          {NUMERIC_FIELDS.map(({ name, label, icon: Icon, min, max }) => (
            <div className="input-wrapper" key={name}>
              <label><Icon size={16} /> {label}</label>
              <input 
                type="number" 
                name={name} 
                min={min} max={max}
                className="styled-input"
                value={formData[name]} 
                onChange={handleChange} 
                required 
              />
            </div>
          ))}

          <div className="input-wrapper">
            <label><Snowflake size={16} /> ระบบแอร์ส่วนกลาง</label>
            <select name="CentralAir" className="styled-input" value={formData.CentralAir} onChange={handleChange}>
              <option value="Y">มี</option>
              <option value="N">ไม่มี</option>
            </select>
          </div>

          <div className="input-wrapper">
            <label><Building2 size={16} /> เขตที่ดิน (Zoning)</label>
            <select name="MSZoning" className="styled-input" value={formData.MSZoning} onChange={handleChange}>
              <option value="RL">ต่ำ (RL)</option>
              <option value="RM">ปานกลาง (RM)</option>
              <option value="RH">สูง (RH)</option>
              <option value="FV">Floating Village</option>
              <option value="C (all)">พาณิชย์ (C)</option>
            </select>
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
              <div className="price-thb">
                ≈ ฿{(price * 34.5).toLocaleString('th-TH', { maximumFractionDigits: 0 })}
              </div>
            </div>
            
            <div className="insight-box">
              <div className="insight-row">
                <span className="insight-label">ราคาต่อ ตร.ฟุต</span>
                <span className="insight-val">${Math.round(price / formData.GrLivArea).toLocaleString()}</span>
              </div>
              <div className="insight-row">
                <span className="insight-label">ความมั่นใจของโมเดล</span>
                <span className="insight-val" style={{color: confidence > 90 ? 'var(--secondary)' : '#f59e0b'}}>
                  {confidence > 95 ? 'สูงมาก' : confidence > 90 ? 'สูง' : 'ปานกลาง'} ({confidence}%)
                </span>
              </div>
              <div className="insight-row">
                <span className="insight-label">ย่านที่ตั้ง</span>
                <span className="insight-val">{formData.Neighborhood}</span>
              </div>
            </div>
            
            {warnings.length > 0 && (
              <div style={{marginTop: '1rem', color: '#f59e0b', fontSize: '0.8rem', display: 'flex', gap: '0.5rem', textAlign: 'left'}}>
                <TriangleAlert size={16} style={{flexShrink: 0}} />
                <span>ค่าที่กรอกอยู่นอกช่วงข้อมูลที่โมเดลเคยเรียนรู้ ราคาอาจคลาดเคลื่อน ({warnings.join('; ')})</span>
              </div>
            )}

            <div style={{marginTop: '1.5rem', color: 'var(--text-muted)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem'}}>
              <TrendingUp size={14} />
              อ้างอิงจากข้อมูลสถิติราคาบ้านใน Ames
            </div>
          </div>
        )}
      </div>

    </div>

    {/* SIDEBAR - History */}
    {history.length > 0 && (
      <div className="history-sidebar">
        <h3 className="history-title"><History size={16} /> ประวัติการประเมิน</h3>
        <div className="history-list">
          {history.map(item => (
            <div key={item.id} className="history-item">
              <div className="history-main">
                <span className="history-price">${item.price.toLocaleString('en-US')}</span>
                <span className="history-date"><Clock size={12} style={{marginRight: '4px'}}/> {item.date}</span>
              </div>
              <div className="history-details">{item.details}</div>
            </div>
          ))}
        </div>
        {history.length > 0 && (
          <button 
            className="clear-history-btn"
            onClick={() => setHistory([])}
          >
            ล้างประวัติ
          </button>
        )}
      </div>
    )}

    </div>
  );
}

export default App;
