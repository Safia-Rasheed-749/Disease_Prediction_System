import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')
print(df.head(5))
print(df.shape)
print(df.isnull().sum())

# Symptoms columns
symptom_cols = [col for col in df.columns if col != "Disease"]

# =====================
# Data cleaning (IMPORTANT)
# =====================
# - strip disease labels to avoid duplicates like "Diabetes" vs "Diabetes "
# - normalize symptom tokens to avoid feature fragmentation due to whitespace
# - treat empty cells as missing (then drop them)
df['Disease'] = df['Disease'].astype(str).str.strip()

for col in symptom_cols:
    # treat empty cells as missing
    df[col] = df[col].replace({"": np.nan})
    # strip string symptoms; keep NaN as NaN
    df[col] = df[col].apply(lambda x: np.nan if pd.isna(x) else str(x).strip())

# Fill NaNs with 0 for later comparisons (we do not want them as symptoms)
df.fillna(0, inplace=True)

print(df.head())
print(df.isnull().sum())

# Unique symptoms (exclude 0)
all_symptoms = sorted(set(df[symptom_cols].stack()) - {0})
print("Total unique symptoms:", len(all_symptoms))

X = pd.DataFrame(0, index=df.index, columns=all_symptoms, dtype=np.uint8)

# Fast encoding
for i, row in df[symptom_cols].iterrows():
    for symptom in row:
        if symptom != 0:
            X.at[i, symptom] = 1
print(X.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])
print(f"Total unique disease : {len(le.classes_)}")
if 'Disease_Encoded' in df.columns:  # legacy safeguard
    df = df.drop('Disease_Encoded', axis=1)
print(df.tail())
from sklearn.model_selection import train_test_split
X = X
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y)
print("......")
print(f"Total samples: {len(df)}")
print(f"Training target samples: {len(y_train)}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {len(X_test)}")
# Model Training
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf = RandomForestClassifier(n_estimators=200,      # Zyada trees
    max_depth=20,          # Depth limit
    random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Performance")
print(f"Training Accuracy: {rf.score(X_train, y_train)*100:.2f}%")
print(f"Testing Accuracy:  {accuracy*100:.2f}%\n")

# 5. Check real insights to crack the 100% mystery
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
# Saving model
import joblib
import pickle
joblib.dump(rf, 'models/disease_model.joblib')
print("Model saved: models/disease_model.joblib")

# 9. Save Label Encoder
joblib.dump(le, 'models/label_encoder.joblib')
print("Label Encoder saved: models/label_encoder.joblib")

# 10. Save Symptoms List
with open('models/all_symptoms.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)
print(" Symptoms list saved: models/all_symptoms.pkl")

