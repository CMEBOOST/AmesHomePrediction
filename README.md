# Ames Home Prediction Web App 🏡

This repository contains the full stack web application for predicting house prices in Ames, Iowa, based on a machine learning ensemble model (Ridge, Lasso, Gradient Boosting, LightGBM).

## Folder Structure

- `backend/`: FastAPI Python application. Serves the `model_artifacts.joblib` trained on the Ames dataset.
- `frontend/`: React + Vite web application with a beautiful Glassmorphism design to interact with the API.
- `render.yaml`: Configuration for easy deployment of the backend to Render.

## 🚀 Deployment Instructions

### 1. Deploying the Backend (Render)

Render is perfect for hosting our FastAPI backend for free.

1. Go to [Render](https://render.com/) and sign in with GitHub.
2. Click **New +** and select **Blueprint**.
3. Connect this GitHub repository (`CMEBOOST/AmesHomePrediction`).
4. Render will automatically read the `render.yaml` file in this repository and set up the Web Service for you!
5. Once deployment is complete, copy the backend URL (e.g., `https://ames-home-prediction-api.onrender.com`).

*Alternative manual way on Render:*
- Select **Web Service** -> Connect repo.
- Root Directory: `backend`
- Environment: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 2. Deploying the Frontend (Vercel)

Vercel is optimized for frontend frameworks like Vite & React.

1. Go to [Vercel](https://vercel.com/) and sign in with GitHub.
2. Click **Add New** -> **Project**.
3. Import this repository (`CMEBOOST/AmesHomePrediction`).
4. **Important**: In the configuration screen:
   - Expand **Build and Output Settings**.
   - Set **Root Directory** to `frontend` (Click edit and type `frontend`).
   - Expand **Environment Variables** and add:
     - Name: `VITE_API_URL`
     - Value: `[Your Render Backend URL]` (e.g., `https://ames-home-prediction-api.onrender.com`)
5. Click **Deploy**.

## 💻 Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
