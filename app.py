from flask import Flask, render_template, request
import numpy as np
import os
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # Initializing a Flask app

@app.route('/', methods=['GET', 'POST'])  # Single route for home and predictions
def home():
    prediction = None  # Default value for prediction

    if request.method == 'POST':
        try:
            # Reading user inputs from form
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            # Prepare data for model
            data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                             chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                             pH, sulphates, alcohol]).reshape(1, -1)

            # Load prediction pipeline and make prediction
            obj = PredictionPipeline()
            prediction = obj.predict(data)

        except Exception as e:
            print('The Exception message is:', e)
            prediction = "Error in prediction"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
