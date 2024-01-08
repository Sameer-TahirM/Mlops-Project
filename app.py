from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

import pickle

# Replace "path/to/your/xgboost_model.pkl" with the actual path to your pickle file
model_path = "xgboost_model.pkl"

# Load the XGBoost model from the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        features = [float(request.form['satisfaction_level']),
                    float(request.form['last_evaluation']),
                    int(request.form['number_project']),
                    int(request.form['average_monthly_hours']),
                    int(request.form['time_spend_company']),
                    int(request.form['work_accident']),
                    int(request.form['promotion_last_5years']),
                    int(request.form['low']),
                    int(request.form['medium'])]

        # Make a prediction using the model
        prediction = model.predict(np.array(features).reshape(1, -1))[0]

        # Map the prediction to a meaningful result
        result = "Left" if prediction == 1 else "Stayed"

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
