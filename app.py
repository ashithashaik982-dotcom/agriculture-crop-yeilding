import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

st.title("🌾 Crop Yield Prediction App")
st.markdown("### Enter input values to predict crop yield")

input_data = {}

for col in columns[:5]:
    nice_name = col.replace("_", " ").title()
    input_data[col] = st.number_input(nice_name, min_value=0, value=0)

input_df = pd.DataFrame([input_data])

# Fill missing columns
for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]

if st.button("🚀 Predict Yield"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌟 Predicted Crop Yield: {int(prediction)}")