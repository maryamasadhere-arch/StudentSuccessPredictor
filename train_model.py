import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_excel("student_performance_dataset.xlsx")

# Encode categorical columns
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Parental_Education_Level"] = le.fit_transform(df["Parental_Education_Level"])
df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"Yes": 1, "No": 0})
df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"Yes": 1, "No": 0})

# Define features and target
X = df.drop("Pass_Fail", axis=1)
y = df["Pass_Fail"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model using current version of scikit-learn
joblib.dump(model, "student_success_model.pkl")
print("✅ Model retrained and saved successfully!")
