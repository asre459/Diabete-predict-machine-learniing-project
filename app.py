

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
<<<<<<< HEAD
from fastapi.responses import HTMLResponse
=======
from fastapi.responses import HTMLResponse  # Import HTMLResponse
>>>>>>> d1a964b62f2d6a20e8fb42c07215627b80ebf02f
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler with error handling
try:
    model = joblib.load("diabetes_model.pkl")  # Ensure your model file is in the same directory
    scaler = joblib.load("scaler.pkl")  # Ensure your scaler file is available
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model, scaler = None, None  # Set to None to prevent crashes

# Initialize FastAPI
app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request body structure
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

@app.post("/predict")
def predict_diabetes(input: DiabetesInput):
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded.")
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Check server logs.")

    try:
        # Create a DataFrame with the same columns used during training
        input_data = pd.DataFrame([{
            "pregnancies": input.pregnancies,
            "glucose": input.glucose,
            "blood_pressure": input.blood_pressure,
            "skin_thickness": input.skin_thickness,
            "insulin": input.insulin,
            "bmi": input.bmi,
            "diabetes_pedigree_function": input.diabetes_pedigree_function,
            "age": input.age
        }])

        logger.info(f"Input data shape: {input_data.shape}")

        # Scale input data using the scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing prediction.")

# Serve the HTML file at the root URL
@app.get("/")
def read_root():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

