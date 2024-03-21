import re
from flask import Flask, request, render_template, redirect, url_for
import pickle

import re
from flask import Flask, request, render_template, redirect, url_for
import csv

app = Flask(__name__)

# Load vectorizer and model
vectorizer1 = pickle.load(open('vectorizer.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('rain.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    location = request.form.get('location')
    sublocation = request.form.get('sublocation')
    body_part = request.form.get('bodypart')
    clinical_history = request.form.get('clinical_history')

    # Process and vectorize the clinical history
    processed_text = clinical_history.lower().replace('[^a-zA-Z\s]', '')
    clinical_history_vectorized = vectorizer1.transform([processed_text])

    # Make prediction
    prediction = rf_model.predict(clinical_history_vectorized)[0]

    # Render the feedback template with the prediction and input data
    return render_template('nw_feedback.html', location=location, sublocation=sublocation, body_part=body_part, clinical_history=clinical_history, predicted_protocol=prediction)

@app.route('/feedback', methods=['POST'])
def feedback():
    # Get input data from the form
    location = request.form.get('location')
    sublocation = request.form.get('sublocation')
    body_part = request.form.get('bodypart')
    clinical_history = request.form.get('clinical_history')
    predicted_protocol = request.form.get('predicted_protocol')
    real_protocol = request.form.get('real_protocol')
    is_prediction_correct = request.form.get('prediction_correct') == 'yes'

    # Save input, output, and feedback data to CSV
    with open('data.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_prediction_correct:
            writer.writerow([location, sublocation, body_part, clinical_history, predicted_protocol, real_protocol, "Correct"])
        else:
            writer.writerow([location, sublocation, body_part, clinical_history, predicted_protocol, real_protocol, "Incorrect"])

    # Redirect back to the home page
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, port=8010)
