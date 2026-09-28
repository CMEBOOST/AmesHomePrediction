# Ames Home Prediction Web App 🏡

โปรเจคนี้คือเว็บแอปพลิเคชันแบบ Full Stack สำหรับการทำนายราคาบ้านใน Ames, Iowa โดยใช้โมเดล Machine Learning Ensemble (ประกอบด้วย Ridge, Lasso, Gradient Boosting และ LightGBM)

## โครงสร้างโฟลเดอร์ (Folder Structure)

- `backend/`: แอปพลิเคชัน Python ที่สร้างด้วย FastAPI ทำหน้าที่เป็น API ในการรันโมเดล (`model_artifacts.joblib`) ที่เทรนจากชุดข้อมูล Ames
- `frontend/`: เว็บแอปพลิเคชัน React + Vite ที่ออกแบบด้วยสไตล์ Glassmorphism สวยงาม สำหรับเชื่อมต่อและแสดงผลจาก API
- `render.yaml`: ไฟล์การตั้งค่าสำหรับการนำ backend ไปรันบน Render ได้อย่างง่ายดาย

## 🚀 คำแนะนำการใช้งานและการนำไปติดตั้ง (Deployment Instructions)

### 1. การ Deploy Backend (บน Render)

Render เป็นบริการที่เหมาะมากสำหรับการโฮสต์ FastAPI backend ของเราได้ฟรี

1. ไปที่ [Render](https://render.com/) และล็อกอินด้วย GitHub
2. คลิก **New +** และเลือก **Blueprint**
3. เชื่อมต่อกับ GitHub repository นี้
4. Render จะอ่านไฟล์ `render.yaml` ใน repository และตั้งค่า Web Service ให้โดยอัตโนมัติ!
5. เมื่อการ deploy สำเร็จ ให้คัดลอก URL ของ backend (เช่น `https://ames-home-prediction-api.onrender.com`)

*วิธีตั้งค่าแบบแมนนวลบน Render (กรณีไม่ใช้ Blueprint):*
- เลือก **Web Service** -> เชื่อมต่อ repo
- Root Directory: `backend`
- Environment: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 2. การ Deploy Frontend (บน Vercel)

Vercel เหมาะที่สุดสำหรับการโฮสต์ frontend framework อย่าง Vite & React

1. ไปที่ [Vercel](https://vercel.com/) และล็อกอินด้วย GitHub
2. คลิก **Add New** -> **Project**
3. Import repository นี้
4. **สำคัญมาก**: ในหน้าจอการตั้งค่า (Configuration):
   - ขยายเมนู **Build and Output Settings**
   - ตั้งค่า **Root Directory** เป็น `frontend` (คลิก edit และพิมพ์ `frontend`)
   - ขยายเมนู **Environment Variables** และเพิ่มค่าตามนี้:
     - Name: `VITE_API_URL`
     - Value: `[URL Backend บน Render ของคุณ]` (เช่น `https://ames-home-prediction-api.onrender.com`)
5. คลิก **Deploy**

## 💻 การรันโปรเจคในเครื่อง (Local Development)

**สำหรับ Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**สำหรับ Frontend:**
```bash
cd frontend
npm install
npm run dev
```
