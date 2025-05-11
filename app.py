import gradio as gr
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model (e.g., RandomForestClassifier)
model = joblib.load("rf_model.pkl")

# Sample data format: [N, P, K, temperature, humidity, pH, rainfall]
CROP_LABELS = ["apple",
"banana",
"blackgram",
"chickpea",
"coconut",
"coffee",
"cotton",
"grapes",
"jute",
"kidneybeans",
"lentil ",
"maize ",
"mango ",
"mothbeans ",
"mungbean ",
"muskmelon ",
"orange ",
"papaya ",
"pigeonpeas ",
"pomegranate ",
"rice ",
"watermelon"] 

# Create SHAP explainer
explainer = shap.Explainer(model)


def predict_with_explanation(N, P, K, temperature, humidity, ph, rainfall):
    # Define the feature names exactly as they were used during training
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    
    # Prepare the input as a DataFrame with the correct column names (case-sensitive)
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
    
    # Perform prediction using the model
    pred = model.predict(input_data)[0]
    label = CROP_LABELS[pred] if isinstance(pred, int) else pred  # Assuming CROP_LABELS maps to crop names
    
    # SHAP explanation for the input data
    shap_values = explainer.shap_values(input_data)
    shap_value = shap_values[0]  # We are explaining the first instance

    # Generate SHAP explanation as text (no plot)
    explanation = f"Explanation for predicted crop: {label}\n"
    
    # Loop through each feature to display its contribution (positive or negative)
    for feature, shap_val in zip(feature_names, shap_value.values):
        explanation += f"{feature}: {'Increased' if shap_val > 0 else 'Decreased'} by {abs(shap_val):.4f}\n"

    return explanation, shap_value




# Gradio inputs
inputs = [
    gr.Number(label="Nitrogen (N)"),
    gr.Number(label="Phosphorus (P)"),
    gr.Number(label="Potassium (K)"),
    gr.Number(label="Temperature (°C)"),
    gr.Number(label="Humidity (%)"),
    gr.Number(label="pH"),
    gr.Number(label="Rainfall (mm)")
]

# Gradio interface
demo = gr.Interface(
    fn=predict_with_explanation,
    inputs=inputs,
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(type="filepath", label="SHAP Explanation")
    ],
    title="Crop Recommendation System with SHAP",
    description="Predict the best crop and see which features contributed most to the prediction."
)

demo.launch()
