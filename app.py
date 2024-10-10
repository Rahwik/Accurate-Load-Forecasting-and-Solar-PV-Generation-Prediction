from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models
load_model = joblib.load('models/load_forecasting_model.pkl')
solar_model = joblib.load('models/solar_pv_generation_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values for load forecasting
        hour = int(request.form['hour'])
        day_of_week = int(request.form['day_of_week'])
        month = int(request.form['month'])

        # Get input values for solar PV generation
        day_of_year = int(request.form['day_of_year'])
        cloud_cover = float(request.form['cloud_cover'])
        temperature = float(request.form['temperature'])

        # Load forecasting prediction
        load_input = np.array([[hour, day_of_week, month]])
        load_prediction = load_model.predict(load_input)[0]

        # Solar PV generation prediction
        solar_input = np.array([[hour, day_of_year, cloud_cover, temperature]])
        solar_prediction = solar_model.predict(solar_input)[0]

        return render_template('results.html', load_prediction=load_prediction, solar_prediction=solar_prediction)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
