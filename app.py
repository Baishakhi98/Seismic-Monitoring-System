from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import ensemble
import os
import pandas as pd
import numpy as np
import statistics
import pickle
from flask import Flask, request, render_template, redirect, url_for
from mcculw import ul
from mcculw.enums import ULRange
from mcculw.ul import ULError

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model_path = "rfc_model.pkl" 
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    return redirect(url_for('predict'))

@app.route('/predict')
def predict():
    def extract_features(data):
        maxamp = max(data)
        mean_val = statistics.mean(data)
        std_dev = statistics.stdev(data)
        energy = np.sum(np.square(np.abs(data)))
        avg_power = energy / (2 * len(data) + 1)
        #entropy = -np.sum(data * np.log2(data))
        features = {
            "Maximum Amplitude": maxamp,
            "Mean": mean_val,
            "S.D": std_dev,
            "Energy": energy,
            "Average Power": avg_power,
            #"Entropy": entropy,
            "Label": 'Run'
        }
        return features

    boardnum = 0
    airange = ULRange.BIP10VOLTS
    channel = 1
    data = []

    while True:
        try:
            value = ul.a_in(boardnum, channel, airange)
            eng_value = ul.to_eng_units(boardnum, airange, value)
            data.append(float('{:.6f}'.format(eng_value)))

            if len(data) == 100:
                features = extract_features(data)
                df = pd.DataFrame([features], columns=["Maximum Amplitude", "Mean", "S.D", "Energy", "Average Power", "Label"])
                prediction = model.predict(df.drop(columns=["Label"]))
                result = str(prediction[0])

                if result == "Movement":
                    source = "Movement"
                elif result == "No Movement":
                    source = "No Movement"
                
                return render_template('disturbance.html', prediction_text=source)
                data = []
        except ULError as e:
            return render_template('disturbance.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
