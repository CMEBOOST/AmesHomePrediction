from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from ml_pipeline import preprocess_data
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ames Housing Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change to the Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.joblib")

# Load model globally
try:
    artifacts = joblib.load(MODEL_PATH)
    models = artifacts['models']
    weights = artifacts['weights']
    training_columns = artifacts['columns']
    print("Model artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None

class HomeData(BaseModel):
    # Defining only the most important features to simplify the UI
    # We will fill the rest with default values (medians/modes)
    GrLivArea: int = 1500
    OverallQual: int = 6
    YearBuilt: int = 2000
    TotalBsmtSF: int = 1000
    GarageCars: int = 2
    FullBath: int = 2
    Neighborhood: str = "CollgCr"

@app.get("/")
def read_root():
    return {"message": "Welcome to Ames Home Prediction API"}

@app.post("/predict")
def predict_price(data: HomeData):
    if models is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Create a dataframe with the input data
    input_dict = data.dict()
    
    # We need to supply missing columns for the preprocessor. 
    # For a real app, we might ask for all columns or use a predefined template.
    # Here we create a template row with generic default values
    # These should ideally match median/mode of training data.
    # To keep it simple, we initialize an empty dataframe and then add our values.
    # Actually, pandas can be created with one row and lots of NaNs, then our 
    # preprocess_data function will fill those NaNs with mode/0!
    
    # Create empty dataframe with a few essential columns to prevent errors
    df_input = pd.DataFrame([input_dict])
    
    # Preprocess
    try:
        X_processed = preprocess_data(df_input, is_train=False, training_columns=training_columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # Predict
    # test_preds has shape (1, 4) since we have 4 models
    test_preds = np.column_stack([pipe.predict(X_processed) for pipe in models.values()])
    
    # Calculate final weighted prediction
    final_log_pred = np.dot(test_preds, weights)[0]
    final_price = np.expm1(final_log_pred)
    
    # Calculate confidence based on Ensemble Agreement
    # We convert log predictions to dollar prices for all 4 models
    dollar_preds = np.expm1(test_preds[0])
    # Calculate standard deviation among the 4 predictions
    std_dev = np.std(dollar_preds)
    
    # Coefficient of Variation (CV) = Standard Deviation / Mean
    cv = std_dev / final_price
    
    # Map CV to a confidence score (0 to 100)
    # If standard deviation is 5% of the price (cv = 0.05), confidence is 95%
    confidence_score = max(0, min(99.9, 100 * (1 - cv)))

    return {
        "predicted_price": round(final_price, 2),
        "confidence_score": round(confidence_score, 1)
    }
