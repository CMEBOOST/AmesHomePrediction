# Ames Home Prediction Web App 🏡

โปรเจคนี้คือเว็บแอปพลิเคชันแบบ Full Stack สำหรับการทำนายราคาบ้านใน Ames, Iowa โดยใช้โมเดล Machine Learning Ensemble (ประกอบด้วย Ridge, Lasso, Gradient Boosting และ LightGBM)

## โครงสร้างโฟลเดอร์ (Folder Structure)

- `backend/`: แอปพลิเคชัน Python ที่สร้างด้วย FastAPI ทำหน้าที่เป็น API ในการรันโมเดล (`model_artifacts.joblib`) ที่เทรนจากชุดข้อมูล Ames
- `frontend/`: เว็บแอปพลิเคชัน React + Vite ที่ออกแบบด้วยสไตล์ Glassmorphism สวยงาม สำหรับเชื่อมต่อและแสดงผลจาก API
- `render.yaml`: ไฟล์การตั้งค่าสำหรับการนำ backend ไปรันบน Render ได้อย่างง่ายดาย

## 🧠 โมเดล Machine Learning ของเรา

โปรเจคนี้ใช้โมเดลแบบ **Ensemble Learning** ที่เกิดจากการนำโมเดล 4 ตัวมารวมกันเพื่อเพิ่มความแม่นยำในการทำนายราคาบ้าน (`SalePrice`) โดยมีรายละเอียดดังนี้:

### 1. การเตรียมข้อมูลและสร้าง Feature ใหม่ (Data Preprocessing & Feature Engineering)
- **จัดการค่าว่าง (Missing Values)**: เติมค่าว่างด้วย `None` สำหรับข้อมูลหมวดหมู่ (Categorical) เช่น ประเภทโรงรถ และเติม `0` หรือค่ามัธยฐาน (`Median`) สำหรับข้อมูลเชิงตัวเลข
- **สร้างฟีเจอร์ใหม่ (Feature Engineering)**: เช่น คำนวณอายุของบ้าน ณ วันที่ขาย (`AgeAtSale`), รวมพื้นที่บ้านทั้งหมด (`TotalSF`), รวมพื้นที่ระเบียง (`TotalPorchSF`), จำนวนห้องน้ำรวม (`TotalBathrooms`), และเพิ่มฟีเจอร์บ่งชี้ว่าบ้านมีสระว่ายน้ำ โรงรถ หรือเตาผิงหรือไม่
- **การแปลงข้อมูล (Encoding)**: ทำการแปลงข้อมูลเชิงอันดับ (Ordinal Encoding) ให้กับฟีเจอร์ประเมินคุณภาพต่างๆ (เช่น `ExterQual`) และใช้ One-Hot Encoding สำหรับข้อมูลหมวดหมู่อื่นๆ

### 2. การคัดเลือกฟีเจอร์ (Feature Selection)
- ใช้โมเดล **Lasso Regression** เป็นตัวช่วยคัดกรอง (Feature Selection) เพื่อเลือกเฉพาะตัวแปรที่มีผลต่อราคาบ้านมากที่สุดเข้าสู่กระบวนการเรียนรู้

### 3. โมเดล Ensemble ที่ใช้
เราเทรนโมเดลทั้งหมด 4 ชนิด และนำผลการทำนายมาผสมผสานด้วยการถ่วงน้ำหนัก (Weighted Average) ดังนี้:
- **Ridge Regression (น้ำหนัก 15%)**: โมเดลความสัมพันธ์เชิงเส้นที่มีการทำ L2 Regularization
- **Lasso Regression (น้ำหนัก 15%)**: โมเดลความสัมพันธ์เชิงเส้นที่มี L1 Regularization 
- **Gradient Boosting Regressor (น้ำหนัก 35%)**: โมเดลกลุ่ม Tree-based แบบ Boosting ที่ค่อยๆ เรียนรู้เพื่อลดข้อผิดพลาด
- **LightGBM (น้ำหนัก 35%)**: โมเดล Boosting Tree-based ที่มีความรวดเร็วสูงและจับรูปแบบความสัมพันธ์ซับซ้อนได้ดี

> **Note:** ตัวแปรเป้าหมาย (ราคาบ้าน หรือ `SalePrice`) ถูกนำมาแปลงด้วย Log-Transformation (`log1p`) ก่อนนำไปให้โมเดลเรียนรู้ และถูกตัดค่า Outliers ที่ผิดปกติออกไป เพื่อให้ตัวโมเดลทำงานได้อย่างมีประสิทธิภาพและแม่นยำที่สุด

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
