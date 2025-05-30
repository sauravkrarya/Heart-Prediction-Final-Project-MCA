from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
with open("trained_models.pkl", "rb") as file:
    all_models = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input
        name = request.form['name']
        email = request.form['email']
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form.get('trestbps', 95))
        chol = int(request.form.get('chol', 150))
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form.get('thalach', 72))
        exang = int(request.form['exang'])
        oldpeak = float(request.form.get('oldpeak', 2.0))
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Feature vector
        features = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]

        # For displaying user input
        input_data = {
            "Age": age, "Sex": "Male" if sex == 1 else "Female", "Chest Pain Type": cp,
            "Resting BP": trestbps, "Cholesterol": chol, "FBS": fbs,
            "Resting ECG": restecg, "Max Heart Rate": thalach,
            "Exercise Angina": exang, "Old Peak": oldpeak,
            "Slope": slope, "Major Vessels": ca, "Thal": thal
        }

        predictions = {}
        high_risk_count = 0

        for model_name, model in all_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([features])[0][1]
                result = 1 if proba >= 0.5 else 0
                predictions[model_name] = f"{'High Risk' if result == 1 else 'Low Risk'} ({round(proba*100, 1)}%)"
            else:
                result = model.predict([features])[0]
                predictions[model_name] = "High Risk" if result == 1 else "Low Risk"

            high_risk_count += result

        # Calculate overall high-risk percentage
        accuracy = round((high_risk_count / len(all_models)) * 100, 2)

        personal_info = [name, email]
        responses = [input_data, predictions, personal_info, accuracy]

        return render_template("result.html", result=responses)

    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
