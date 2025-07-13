import streamlit as st
import torch
import numpy as np
from preprocess import all_symptoms, le
from model import DiagnosisModel
from data_loader import get_description, get_precautions
import torch.nn as nn
import matplotlib.pyplot as plt

# Streamlit rerun fix for latest versions
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import get_script_run_ctx

def reset_app():
    raise RerunException(get_script_run_ctx())

# ---------- Load the Model ----------
input_size = len(all_symptoms)
num_classes = len(le.classes_)

class DiagnosisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DiagnosisModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = DiagnosisModel(input_size, num_classes)
model.load_state_dict(torch.load("diagnosis_model.pt"))
model.eval()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Medical Diagnosis Assistant", layout="centered")
st.title("🩺 Medical Diagnosis Assistant")
st.markdown("Select your symptoms from the list below to get a disease prediction.")

# Symptom selection
selected_symptoms = st.multiselect(
    "🔍 Select symptoms:",
    options=all_symptoms,
    help="Choose multiple symptoms from the list."
)

# Reset button
if st.button("🔄 Reset"):
    reset_app()

# Diagnose button
if st.button("🧠 Diagnose"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Convert symptoms to input vector
        input_symptoms = [s.lower() for s in selected_symptoms]
        input_vector = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]
        input_tensor = torch.tensor([input_vector], dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            top_indices = np.argsort(probabilities)[::-1][:3]

        # Show top predictions
        st.markdown("### 🧪 Top 3 Predictions:")
        for idx in top_indices:
            disease = le.classes_[idx]
            prob = probabilities[idx] * 100
            st.write(f"🔹 **{disease}** – {prob:.2f}%")

        # Plot bar chart
        st.markdown("### 📊 Probability Chart:")
        top_diseases = [le.classes_[i] for i in top_indices]
        top_probs = [probabilities[i]*100 for i in top_indices]

        fig, ax = plt.subplots()
        bars = ax.barh(top_diseases[::-1], top_probs[::-1], color='skyblue')
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.bar_label(bars, fmt='%.2f%%', padding=3)  # ✅ FIXED
        st.pyplot(fig)

        # Show full info for top-1 prediction
        top_disease = le.classes_[top_indices[0]]
        st.success(f"✅ Most likely disease: **{top_disease}**")

        # Description
        st.markdown("### 📝 Disease Description:")
        st.info(get_description(top_disease))

        # Precautions
        st.markdown("### 🛡️ Recommended Precautions:")
        for p in get_precautions(top_disease):
            st.write(f"• {p}")
