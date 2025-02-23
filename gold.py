from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
gold_data = pd.read_csv('data/gld_price_data.csv')
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Train the model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X, Y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    spx = float(request.form['spx'])
    uso = float(request.form['uso'])
    slv = float(request.form['slv'])
    eur_usd = float(request.form['eur_usd'])
    
    features = np.array([[spx, uso, slv, eur_usd]])
    prediction = regressor.predict(features)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Gold Price: ${prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
