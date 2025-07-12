import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- Load dataset ---
df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# --- Preprocessing ---
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# --- Model pipeline with RandomForest ---
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit the model ---
model.fit(X_train, y_train)

# --- Accuracy on test set ---
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# --- Cross-validation score ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
cv_mean = cv_scores.mean()

# --- Display results ---
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Cross-Validation Accuracy (5-fold): {cv_mean * 100:.2f}%")

# --- Save model ---
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

# --- Save accuracy metrics ---
with open("model_accuracy.txt", "w") as f:
    f.write(str(test_accuracy))

with open("cv_accuracy.txt", "w") as f:
    f.write(str(cv_mean))

print("✅ Model and both accuracy scores saved.")
