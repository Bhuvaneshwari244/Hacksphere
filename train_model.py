import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example synthetic training data (replace with your dataset)
# X = features, y = labels (0 = healthy, 1 = Parkinson's)
np.random.seed(42)
X = np.random.rand(100, 30)  # 100 samples, 30 features (matches feature_extraction.py)
y = np.random.randint(0, 2, 100)

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained successfully! Accuracy:", accuracy_score(y_test, y_pred))

# Save both model and scaler
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/parkinson_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("💾 Saved model and scaler to /models/")
