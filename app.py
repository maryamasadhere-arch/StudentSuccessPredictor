from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)  # ← This must be named 'app'

# Load trained model
model = joblib.load("student_success_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = 1 if request.form["gender"] == "Male" else 0
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    past_score = float(request.form["past_score"])
    final_score = float(request.form["final_score"])
    parent_edu = {"High School": 1, "Bachelors": 0, "Masters": 2, "PhD": 3}[request.form["parent_edu"]]
    internet = 1 if request.form["internet"] == "Yes" else 0
    activity = 1 if request.form["activity"] == "Yes" else 0

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Study_Hours_per_Week": study_hours,
        "Attendance_Rate": attendance,
        "Past_Exam_Scores": past_score,
        "Parental_Education_Level": parent_edu,
        "Internet_Access_at_Home": internet,
        "Extracurricular_Activities": activity,
        "Final_Exam_Score": final_score
    }])

    prediction = model.predict(input_df)[0]
    result = "Pass" if prediction == 1 else "Fail"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
