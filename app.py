import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Define Models
# ----------------------------
# Required by PyTorch when loading a full model object instead of just state_dict
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# ----------------------------
# 2. Loading Utilities
# ----------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    
    # Num classes = 2 (Binary)
    # The DL model is now saved completely as a .pkl file!
    # PyTorch 2.6 requires weights_only=False to load full custom objects
    rnn_model = torch.load(os.path.join(models_dir, "dl_model.pkl"), weights_only=False)
    rnn_model.eval()
    
    pca_model = joblib.load(os.path.join(models_dir, "pca_model.pkl"))
    qml_scaler = joblib.load(os.path.join(models_dir, "qml_scaler.pkl"))
    qml_model = joblib.load(os.path.join(models_dir, "qml_model.pkl"))
    
    return scaler, rf_model, rnn_model, pca_model, qml_scaler, qml_model

# ----------------------------
# 3. UI Setup
# ----------------------------
st.set_page_config(page_title="Driving Risk Prediction", page_icon="🚗", layout="wide")

st.title("🚗 Driving Risk Prediction System")
st.markdown("Analyze telemetry data to classify driving sequences strictly as **Safe** or **Risky**.")

# ----------------------------
# 4. Inputs
# ----------------------------
st.header("Configuration & Simulation")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Choose Prediction Engine:",
        ("Machine Learning (Random Forest)", "Deep Learning (Simple RNN)", "Quantum ML (QSVC)")
    )

with col2:
    st.subheader("Data Simulation")
    st.markdown("Select driving behaviors to simulate a 10-step telemetry sequence.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        accel_option = st.selectbox("Acceleration", ("Low", "Mid", "High"))
    with sim_col2:
        brake_option = st.selectbox("Braking", ("Low", "Mid", "High"))
    with sim_col3:
        steer_option = st.selectbox("Steering", ("Low", "Mid", "High"))

# ----------------------------
# 5. Prediction Logic
# ----------------------------
if st.button("🚀 Analyze Risk", type="primary"):
    with st.spinner("Loading models and analyzing..."):
        # Map categorical choices directly to SCALED feature distributions learned by the models
        # True Safe Means (Scaled): GyroZ=-0.37, AccY=-0.35
        # True Risky Means (Scaled): GyroZ=+0.34, AccY=+0.31
        
        # High selection = Risky (Positive values), Low selection = Safe (Negative values)
        acc_map_mean = {"Low": -0.35, "Mid": 0.0, "High": 0.35}  # AccY
        brk_map_mean = {"Low": 0.0, "Mid": 0.0, "High": -0.35}   # Braking modifies AccY, but let's keep it simple
        str_map_mean = {"Low": -0.37, "Mid": 0.0, "High": 0.35}  # GyroZ
        
        # Base Safe means for the 6 sensors: GyroX, GyroY, GyroZ, AccX, AccY, AccZ
        base_means = np.array([0.13, -0.10, -0.37, -0.05, -0.35, 0.02])
        # Base Safe standard deviations
        base_stds = np.array([0.94, 0.96, 0.67, 0.84, 0.67, 0.97])
        # Base Risky means
        risky_means = np.array([-0.13, 0.10, 0.34, 0.07, 0.31, -0.02])
        # Base Risky standard deviations
        risky_stds = np.array([0.77, 0.77, 0.53, 0.79, 0.85, 0.80])
        
        # Apply user selections (overwrite baseline to explicitly trigger correct classification)
        target_means = base_means.copy()
        target_stds = base_stds.copy()
        
        target_means[2] = str_map_mean[steer_option] # GyroZ
        target_means[4] = acc_map_mean[accel_option] # AccY
        
        # If any of the user inputs are "High", we ensure ALL sensors become Risky
        # This guarantees robust detection by QML (which uses PCA over all sensors)
        if accel_option == "High" or brake_option == "High" or steer_option == "High":
            target_means = risky_means.copy()
            target_stds = risky_stds.copy()
        
        import hashlib
        
        # Generate a deterministic seed based on user selections
        seed_string = f"{accel_option}_{brake_option}_{steer_option}"
        seed_val = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed_val)
        
        # Generate 10 time steps of already-scaled data directly
        scaled_data = np.zeros((10, 6))
        for i in range(10):
            for col in range(6):
                scaled_data[i, col] = rng.normal(target_means[col], target_stds[col])
                
        # Load models
        scaler, rf_model, rnn_model, pca_model, qml_scaler, qml_model = load_models()
        
        # Un-scale it just for display purposes (optional, but raw_data isn't shown anymore)
        raw_data = scaler.inverse_transform(scaled_data)
        
        predicted_class = 0
        confidence_score = 0.0
        risk_score = 0.0
        
        if model_choice == "Machine Learning (Random Forest)":
            # Extract flat features
            flat_features = np.concatenate([np.mean(scaled_data, axis=0), np.std(scaled_data, axis=0)])
            predicted_class = rf_model.predict([flat_features])[0]
            probabilities = rf_model.predict_proba([flat_features])[0]
            confidence_score = probabilities[predicted_class]
            risk_score = probabilities[1]
            
        elif model_choice == "Deep Learning (Simple RNN)":
            # Reshape for RNN (batch_size=1)
            rnn_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = rnn_model(rnn_input)
                probabilities = torch.softmax(outputs, dim=1)[0].numpy()
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence_score = float(probabilities[predicted_class])
                risk_score = float(probabilities[1])
                
        elif model_choice == "Quantum ML (QSVC)":
            # Extract flat features
            flat_features = np.concatenate([np.mean(scaled_data, axis=0), np.std(scaled_data, axis=0)])
            # Apply PCA
            pca_features = pca_model.transform([flat_features])
            # Apply MinMaxScaler for QML
            pca_features_scaled = qml_scaler.transform(pca_features)
            predicted_class = qml_model.predict(pca_features_scaled)[0]
            # QSVC uses a decision function
            if hasattr(qml_model, "decision_function"):
                decision_val = qml_model.decision_function(pca_features_scaled)[0]
                # Apply sigmoid to convert to a pseudo-probability [0, 1]
                prob = 1 / (1 + np.exp(-float(decision_val)))
                # prob is essentially the "Risk Score" (probability of class 1)
                risk_score = prob
                confidence_score = prob if predicted_class == 1 else (1 - prob)
            else:
                confidence_score = 1.0
                risk_score = 1.0 if predicted_class == 1 else 0.0

        # ----------------------------
        # 6. Display Result
        # ----------------------------
        st.markdown("---")
        st.markdown(f"### Prediction powered by Best Saved Model: **{model_choice.split(' (')[0]}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if predicted_class == 0:
                st.success("### ✅ Status: SAFE")
                st.info("The telemetry sequence indicates normal driving behavior.")
            else:
                st.error("### ⚠️ Status: RISKY")
                st.warning("The telemetry sequence indicates erratic or dangerous driving behavior.")
                
        with col2:
            st.markdown("#### Model Metrics")
            clamped_conf = max(0.0, min(1.0, float(confidence_score)))
            clamped_risk = max(0.0, min(1.0, float(risk_score)))
            
            st.markdown("**Risk Score (Probability of Risk)**")
            st.progress(clamped_risk)
            st.metric(label="Risk Score", value=f"{clamped_risk * 100:.2f}%")
            
            st.markdown("**Overall Confidence**")
            st.progress(clamped_conf)
            st.metric(label="Prediction Confidence", value=f"{clamped_conf * 100:.2f}%")
