from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback # Add this for detailed traceback

app = FastAPI(
    title="🔥 Obesity Prediction API",
    description="Welcome to the Smart Health Checker! 🧠💪 Predict your obesity level in seconds using machine learning.", # API description
    version="1.0.0"
)

# 📦 Load Pretrained XGBoost Pipeline
model_pipeline = None # Initialize to None
try:
    model_pipeline = joblib.load("xgb_best_pipeline.pkl") # Load the pre-trained model pipeline from file
    print("INFO: Model pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: xgb_best_pipeline.pkl not found. Make sure the file is in the correct directory.")
    raise # Re-raise to make sure the app doesn't start with a missing model
except Exception as e:
    print(f"ERROR: Failed to load model pipeline: {e}")
    traceback.print_exc() # Print full traceback
    raise # Re-raise to make sure the app doesn't start

# Define input schema and target names (as you already have them)
class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

original_columns_order = [ # Define the expected order of columns for the DataFrame
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

obesity_map = { # Mapping from obesity labels to numerical codes
    'insufficient weight': 0,
    'normal weight': 1,
    'overweight level i': 2,
    'overweight level ii': 3,
    'obesity type i': 4,
    'obesity type ii': 5,
    'obesity type iii': 6
}
reverse_obesity_map = {v: k for k, v in obesity_map.items()} # Reverse mapping for displaying labels

@app.get("/") # Define a GET endpoint for the root URL
def read_root():
    return {"message": "🎉 Welcome to the Obesity Prediction API! 🚀 Submit your health data to get instant insights."}

@app.post("/predict")
def predict_obesity(data: ObesityInput):
    if model_pipeline is None: # Check if model loaded successfully during startup
        return {"error": "Model not loaded. Server initialization failed."}, 500

    try:
        input_data_dict = data.dict()

        # --- Reapply preprocessing steps done *outside* the pipeline in the notebook ---
        # Height conversion from meters to cm and rounding
        input_data_dict['Height'] = input_data_dict['Height'] * 100
        input_data_dict['Height'] = round(input_data_dict['Height'], 1)
        # Weight and Age rounding
        input_data_dict['Weight'] = round(input_data_dict['Weight'], 1)
        input_data_dict['Age'] = round(input_data_dict['Age'], 1)

        # Rounding FCVC, NCP, CH2O, FAF, TUE to integers
        for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
            if input_data_dict[col] is None:
                # If you allowed NaNs in your training data for these, keep as pd.NA or an imputed value
                input_data_dict[col] = pd.NA
            else:
                input_data_dict[col] = int(round(input_data_dict[col]))
                
        # Standardize MTRANS values
        if input_data_dict['MTRANS'] == 'automobile':
            input_data_dict['MTRANS'] = 'car'
        elif input_data_dict['MTRANS'] == 'motorbike':
            input_data_dict['MTRANS'] = 'motorcycle'
        elif input_data_dict['MTRANS'] == 'bike':
            input_data_dict['MTRANS'] = 'bicycle'
        elif input_data_dict['MTRANS'] == 'public_transportation':
            input_data_dict['MTRANS'] = 'public transport'
        # Other MTRANS values like 'walking' are already fine

        # Standardize CAEC and CALC values
        if input_data_dict['CAEC'] == 'no':
            input_data_dict['CAEC'] = 'never'
        if input_data_dict['CALC'] == 'no':
            input_data_dict['CALC'] = 'never'

        input_df = pd.DataFrame([input_data_dict], columns=original_columns_order)
        
        # Ensure correct dtypes for the DataFrame before passing to pipeline
        for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
            input_df[col] = input_df[col].astype('Int64') # Using 'Int64' for nullable integers
        for col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
             input_df[col] = input_df[col].astype(str) # Ensure these are treated as strings
        
        prediction = model_pipeline.predict(input_df)[0] # Make prediction using the loaded pipeline
        label = reverse_obesity_map.get(prediction, "Unknown Prediction") # Get the human-readable label for the prediction

        return { # Return prediction result
            "prediction_code": int(prediction),
            "prediction_label": label
        }
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        traceback.print_exc() # This will print the full error stack to your Uvicorn terminal
        return {"error": "An error occurred during prediction. Check server logs for details."}, 500