import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

# Step 2: Get the list of symptom columns (Symptom_1 to Symptom_17)
symptom_columns = df.columns[1:]

# Step 3: Collect all unique symptoms from all rows and all symptom columns
all_symptoms = pd.unique(df[symptom_columns].values.ravel())

# Step 4: Clean the symptoms list (remove NaNs, strip whitespaces)
all_symptoms = [s.strip() for s in all_symptoms if isinstance(s, str)]

# Step 5: Remove duplicates and sort the symptoms alphabetically
all_symptoms = sorted(list(set(all_symptoms)))

print(f"Total unique symptoms: {len(all_symptoms)}")
print("First 10 symptoms:", all_symptoms[:10])

# Step 6: Encode each row into a binary vector indicating which symptoms are present
X = []
y = []

for idx, row in df.iterrows():
    # Collect symptoms for this row, remove whitespace
    symptoms = set([str(s).strip() for s in row[1:] if isinstance(s, str)])
    
    # Create a binary vector where 1 means symptom is present
    vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    
    X.append(vector)
    y.append(row['Disease'])

X = np.array(X)
print("Shape of X (input vectors):", X.shape)
print("Sample input vector:", X[0])

# Step 7: Encode disease labels as integers using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Number of unique diseases:", len(le.classes_))
print("Encoded disease classes:", list(le.classes_))
print("Sample encoded label:", y_encoded[0])

from collections import Counter

# Count how many times each disease appears
print("\nDisease distribution:")
disease_counts = Counter(y_encoded)
for label, count in disease_counts.items():
    print(f"{le.classes_[label]}: {count}")

# Total number of samples
print(f"\nTotal samples: {len(X)}")

# Number of unique diseases
print(f" Unique diseases: {len(le.classes_)}")
