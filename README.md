# 🩺 Medical Diagnosis Assistant

A smart, interactive disease prediction system built with **PyTorch** and **Streamlit**.  
Users select symptoms from a list, and the app predicts the most likely disease using a trained neural network.

![App Preview](https://user-images.githubusercontent.com/abdulrahmanhamdi/medical_diagnosis/demo.gif)

## 🚀 Features

- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/ui-checks.svg" width="16"/> **Symptom Selection**: Choose from a list instead of typing manually  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/cpu.svg" width="16"/> **AI-Based Prediction**: Uses PyTorch to classify 41 diseases based on 17 symptoms  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/list-stars.svg" width="16"/> **Top 3 Predictions**: Shows the top 3 possible diseases with confidence scores  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/bar-chart-line.svg" width="16"/> **Interactive Probability Chart**: Visual bar chart using `matplotlib`  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/book.svg" width="16"/> **Disease Descriptions**: Detailed explanation from medical dataset  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/heart-pulse.svg" width="16"/> **Precaution Suggestions**: Health advice for predicted disease  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/arrow-counterclockwise.svg" width="16"/> **One-Click Reset**: Easily clear selections and try again

## 🌐 Live Preview

> Coming soon via Streamlit Cloud or HuggingFace Spaces  
For now, clone and run locally ⬇️

## 📁 Dataset

This app uses the [Disease Symptom Description Dataset on Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset), which includes:

- `dataset.csv`: Symptoms & disease mappings  
- `symptom_description.csv`: Definitions for each disease  
- `symptom_precaution.csv`: Recommended precautions

## 🧠 Technologies Used

| Tool/Library  | Purpose                          |
|---------------|----------------------------------|
| PyTorch       | Deep learning classification     |
| Streamlit     | Web interface                    |
| Pandas        | Data processing                  |
| Matplotlib    | Plotting confidence probabilities|
| scikit-learn  | Encoding disease labels          |

## 💻 How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/abdulrahmanhamdi/medical_diagnosis
cd medical_diagnosis

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python model.py

# 5. Launch the app
streamlit run app.py
```

## 📸 Screenshots

### 🔍 Disease Prediction
![Prediction Screenshot](img/predict.png)

### 📊 Confidence Chart
![Chart Screenshot](img/chart.png)

## ✨ Future Enhancements

- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/robot.svg" width="16"/> Add natural language chatbot input (e.g. "I have a fever and cough")  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/cloud-arrow-down.svg" width="16"/> Connect to real medical APIs (MedlinePlus, WHO)  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/phone.svg" width="16"/> Mobile-optimized version  
- <img src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/icons/person-vcard.svg" width="16"/> Patient history tracking (login system)

## 🤝 Acknowledgments

Special thanks to the contributors of the original [Kaggle dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) and the open-source community.

## 🧾 License

This project is licensed under the MIT License. See `LICENSE` for more info.
