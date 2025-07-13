import torch
import numpy as np
from preprocess import all_symptoms, le
from model import DiagnosisModel
import torch.nn as nn

# Load model info
input_size = len(all_symptoms)
num_classes = len(le.classes_)

# Step 1: Define the same model structure
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

# Step 2: Load trained model
model = DiagnosisModel(input_size, num_classes)
model.load_state_dict(torch.load("diagnosis_model.pt"))
model.eval()

# Step 3: Input symptoms from user
input_symptoms = ["itching", "skin_rash", "dischromic _patches"]

# Step 4: Convert input symptoms to binary vector
def symptoms_to_vector(symptoms, all_symptoms):
    symptoms_set = set([s.strip().lower() for s in symptoms])
    vector = [1 if symptom in symptoms_set else 0 for symptom in all_symptoms]
    return torch.tensor([vector], dtype=torch.float32)

# Step 5: Predict
input_vector = symptoms_to_vector(input_symptoms, all_symptoms)
with torch.no_grad():
    output = model(input_vector)
    predicted_index = torch.argmax(output, dim=1).item()
    predicted_disease = le.classes_[predicted_index]

print("\n🔬 Predicted disease:", predicted_disease)
