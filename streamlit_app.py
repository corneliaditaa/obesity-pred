import streamlit as st
import requests
import json

# 🌐 API URL - Adjust for localhost or tunnel
# IMPORTANT: Replace with your actual deployed FastAPI endpoint URL
# If running FastAPI locally: "http://127.0.0.1:8000/predict"
# If running FastAPI in Colab via localtunnel: "https://your-fastapi-url.loca.lt/predict"
API_URL = "http://127.0.0.1:8000/predict" 


# 🎨 Page Config
st.set_page_config(page_title="Obesity Prediction", page_icon="🍔", layout="centered") # Configure Streamlit page settings

st.title("💡 Smart Obesity Prediction") # Display main title
st.markdown("Predict your obesity level using a machine learning model powered by FastAPI!") # Display subtitle/description
st.divider()

# 🧾 Input Form
st.header("🧍 Personal Information") # Section header for personal info
col1, col2 = st.columns(2) # Create two columns for layout
with col1:
    # Send gender as string, FastAPI will handle encoding
    gender = st.selectbox("Gender", ["male", "female"]).lower().strip() 
    age = st.slider("Age", 10, 100, 25) # Send as float
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01) # Send as float
with col2:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1) # Send as float
    # Send family_history as string, FastAPI will handle encoding
    family_history = st.selectbox("Family History of Overweight", ["yes", "no"]).lower().strip() 
    # Send favc as string, FastAPI will handle encoding
    favc = st.selectbox("Frequent High-Calorie Food Consumption (FAVC)", ["yes", "no"]).lower().strip() 

st.header("🍽️ Eating & Drinking Habits")
col3, col4 = st.columns(2)
with col3:
    # Send as float, FastAPI will handle rounding to Int64
    fcvc = st.slider("Vegetable Consumption (1–3)", 1.0, 3.0, 2.0) 
    # Send as float, FastAPI will handle rounding to Int64
    ncp = st.slider("Number of Meals per Day", 1.0, 4.0, 3.0) 
    # Send as string, FastAPI will handle standardization and one-hot encoding
    caec_options = ["frequently", "sometimes", "always", "no"] # "no" maps to "never" in FastAPI
    caec = st.selectbox("Food Between Meals (CAEC)", caec_options).lower().strip() 
with col4:
    # Send as float, FastAPI will handle rounding to Int64
    ch2o = st.slider("Water Intake (1–3)", 1.0, 3.0, 2.0) 
    # Send as string, FastAPI will handle standardization and one-hot encoding
    calc_options = ["never", "sometimes", "frequently", "always"] # "no" maps to "never" in FastAPI
    calc = st.selectbox("Alcohol Consumption (CALC)", calc_options).lower().strip() 
    # Send scc as string, FastAPI will handle encoding
    scc = st.selectbox("Do You Monitor Calories? (SCC)", ["yes", "no"]).lower().strip() 

st.header("🏃 Lifestyle & Transport")
col5, col6 = st.columns(2)
with col5:
    # Send smoke as string, FastAPI will handle encoding
    smoke = st.selectbox("Do You Smoke?", ["yes", "no"]).lower().strip() 
    # Send as float, FastAPI will handle rounding to Int64
    faf = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0) 
    # Send as float, FastAPI will handle rounding to Int64
    tue = st.slider("Technology Use (hours/day)", 0.0, 3.0, 1.0) 
with col6:
    # Send as string, FastAPI will handle standardization and one-hot encoding
    mtrans_options = ["walking", "public transport", "motorcycle", "car", 
                      "automobile", "motorbike", "bike", "public_transportation"] # Include original names for user choice
    mtrans = st.selectbox("Transportation Mode", mtrans_options).lower().strip() 

st.divider()

# 🔄 Prepare Payload for FastAPI
input_data = {
    "Gender": gender,
    "Age": float(age), # Ensure it's float as per BaseModel
    "Height": float(height), # Ensure it's float
    "Weight": float(weight), # Ensure it's float
    "family_history_with_overweight": family_history,
    "FAVC": favc,
    "FCVC": float(fcvc), # Ensure it's float
    "NCP": float(ncp), # Ensure it's float
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": float(ch2o), # Ensure it's float
    "SCC": scc,
    "FAF": float(faf), # Ensure it's float
    "TUE": float(tue), # Ensure it's float
    "CALC": calc,
    "MTRANS": mtrans
}

# 🚀 Prediction Trigger
st.subheader("🔍 Prediction Result")
if st.button("Predict Obesity Level"):
    with st.spinner("Sending data to FastAPI..."):
        try:
            response = requests.post(API_URL, json=input_data)
            
            # Check for HTTP errors (4xx or 5xx)
            response.raise_for_status() 
            
            # If successful, parse the JSON response
            result = response.json()
            
            # Display prediction
            st.success(f"🎯 Predicted Class: **{result['prediction_label']}** (Code: {result['prediction_code']})")
            
            # Optional: Provide insights based on prediction code
            prediction_code = result.get('prediction_code', -1)
            st.write("---")
            st.subheader("What does this mean?")
            if prediction_code == 0:
                st.info("Your weight is below the healthy range. Consider consulting a nutritionist.")
            elif prediction_code == 1:
                st.info("You are within a healthy weight range. Keep up the good work!")
            elif prediction_code == 2:
                st.warning("You are in the Overweight Level I range. Focus on balanced diet and regular physical activity.")
            elif prediction_code == 3:
                st.warning("You are in the Overweight Level II range. It's recommended to adopt healthier eating habits and increase physical activity.")
            elif prediction_code >= 4:
                st.error("Your weight is in the Obesity range (Type I, II, or III). It's highly recommended to consult a healthcare professional for a tailored plan.")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Connection Error: Could not connect to the FastAPI server. Please ensure the server is running and the URL is correct.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: Received status code {e.response.status_code}. Response: {e.response.text}")
        except json.JSONDecodeError:
            st.error("⚠️ Data Error: Failed to decode JSON response from the server. Server might have returned invalid JSON.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Optional preview of the payload being sent
with st.expander("📋 Preview JSON Payload (for debugging)"):
    st.json(input_data) # Display input data as JSON