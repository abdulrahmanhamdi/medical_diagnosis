import pandas as pd

# Load description data
desc_df = pd.read_csv("symptom_Description.csv")
desc_dict = dict(zip(desc_df["Disease"].str.strip(), desc_df["Description"]))

# Load precaution data
prec_df = pd.read_csv("symptom_precaution.csv")
prec_dict = {}

for _, row in prec_df.iterrows():
    disease = row["Disease"].strip()
    precautions = [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    prec_dict[disease] = precautions

def get_description(disease):
    return desc_dict.get(disease.strip(), "No description available.")

def get_precautions(disease):
    return prec_dict.get(disease.strip(), ["No precautions found."])
