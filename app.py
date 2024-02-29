import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
vectorizer1 = pickle.load(open('vectorizer.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', predicted_protocol='')

@app.route('/predict', methods=['POST'])
def predict():
    clinical_history = request.form['clinical history']

    # Clean and preprocess the input clinical_history
    processed_text = clinical_history.lower().replace('[^a-zA-Z\s]', '')

    # Vectorize the input clinical_history
    clinical_history_vectorized = vectorizer1.transform([processed_text])

    # Make predictions
    prediction = rf_model.predict(clinical_history_vectorized)[0]

    # Return the prediction to the index.html template
    return render_template('index.html', predicted_protocol='The predicted protocol is {}'.format(prediction))

if __name__ == "__main__":
    app.run()
