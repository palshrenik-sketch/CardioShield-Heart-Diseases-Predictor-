from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model safely
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("CRITICAL: model.pkl not found. Please run your training script first.")

@app.route("/")
def home():
    profile_type = request.args.get('profile', '')
    
    # Define fallback defaults
    form_data = {
        "age": "", "sex": "", "cp": "", "trestbps": "", "chol": "",
        "fbs": "", "restecg": "", "thalach": "", "exang": "",
        "oldpeak": "", "slope": "", "ca": "", "thal": ""
    }
    
    # Backend preset loading
    if profile_type == 'healthy':
        form_data = {
            "age": "42", "sex": "0", "cp": "2", "trestbps": "115", "chol": "210",
            "fbs": "0", "restecg": "0", "thalach": "172", "exang": "0",
            "oldpeak": "0.2", "slope": "0", "ca": "0", "thal": "0"
        }
    elif profile_type == 'at-risk':
        form_data = {
            "age": "64", "sex": "1", "cp": "0", "trestbps": "150", "chol": "286",
            "fbs": "1", "restecg": "2", "thalach": "108", "exang": "1",
            "oldpeak": "2.6", "slope": "2", "ca": "2", "thal": "2"
        }

    return render_template("index.html", prediction=None, form_data=form_data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect current form variables to keep them filled on refresh
        form_data = {key: request.form.get(key, '') for key in request.form.keys()}
        
        age = float(request.form.get("age", 0))
        sex = float(request.form.get("sex", 0))
        cp = float(request.form.get("cp", 0))
        trestbps = float(request.form.get("trestbps", 0))
        chol = float(request.form.get("chol", 0))
        fbs = float(request.form.get("fbs", 0))
        restecg = float(request.form.get("restecg", 0))
        thalach = float(request.form.get("thalach", 0))
        exang = float(request.form.get("exang", 0))
        oldpeak = float(request.form.get("oldpeak", 0.0))
        slope = float(request.form.get("slope", 0))
        ca = float(request.form.get("ca", 0))
        thal = float(request.form.get("thal", 0))

        features = np.array([[age, sex, cp, trestbps, chol,
                              fbs, restecg, thalach, exang,
                              oldpeak, slope, ca, thal]])

        # Parse raw prediction matrix entry
        raw_prediction = int(model.predict(features)[0])

        # Core Adjustment: Swap binary assignments to resolve inverse targets
        if raw_prediction == 0:
            result_text = "High Risk of Heart Disease"
            display_prediction = 1
        else:
            result_text = "Low Risk of Heart Disease"
            display_prediction = 0

        return render_template("index.html", prediction_text=result_text, prediction=display_prediction, form_data=form_data)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Processing Error: {str(e)}", prediction=-1, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)