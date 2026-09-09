import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load pickle model safely relative to script path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "linear.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Estimator</title>
    <style>
        :root {
            --bg-color: #f4f7fe;
            --card-bg: #ffffff;
            --primary: #4318ff;
            --primary-hover: #3311cc;
            --text-dark: #2b2d42;
            --text-muted: #a3ed2d;
            --border-color: #e0e5f2;
            --shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-dark);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 550px;
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .container:hover {
            box-shadow: 0px 20px 45px rgba(112, 144, 176, 0.2);
        }

        h2 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 26px;
            font-weight: 700;
            color: var(--text-dark);
            text-align: center;
        }

        p.subtitle {
            text-align: center;
            color: #7090b0;
            margin-bottom: 28px;
            font-size: 14px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .full-width {
            grid-column: span 2;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #2b2d42;
        }

        input, select {
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #ffffff;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(67, 24, 255, 0.15);
        }

        button {
            margin-top: 20px;
            width: 100%;
            padding: 14px;
            background-color: var(--primary);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0px 10px 20px rgba(67, 24, 255, 0.2);
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 25px;
            padding: 18px;
            background: #f4f7fe;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #e0e5f2;
        }

        .result-card span {
            display: block;
            font-size: 13px;
            color: #7090b0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-card strong {
            font-size: 24px;
            color: var(--primary);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Property Valuation</h2>
        <p class="subtitle">Enter specs below to predict property price</p>
        
        <form action="/predict" method="POST">
            <div class="grid">
                <div class="form-group">
                    <label>Square Footage</label>
                    <input type="number" step="any" name="Square_Footage" required placeholder="e.g. 2000">
                </div>
                
                <div class="form-group">
                    <label>Bedrooms</label>
                    <input type="number" name="Num_Bedrooms" required placeholder="e.g. 3">
                </div>

                <div class="form-group">
                    <label>Bathrooms</label>
                    <input type="number" name="Num_Bathrooms" required placeholder="e.g. 2">
                </div>

                <div class="form-group">
                    <label>Year Built</label>
                    <input type="number" name="Year_Built" required placeholder="e.g. 2015">
                </div>

                <div class="form-group">
                    <label>Lot Size (sq ft)</label>
                    <input type="number" step="any" name="Lot_Size" required placeholder="e.g. 5000">
                </div>

                <div class="form-group">
                    <label>Garage Size (cars)</label>
                    <input type="number" name="Garage_Size" required placeholder="e.g. 2">
                </div>

                <div class="form-group full-width">
                    <label>Neighborhood Quality</label>
                    <select name="Neighborhood_Quality" required>
                        <option value="" disabled selected>Select category...</option>
                        <option value="1">Low Quality</option>
                        <option value="2">Medium Quality</option>
                        <option value="3">High Quality</option>
                        <option value="4">Premium Quality</option>
                    </select>
                </div>
            </div>

            <button type="submit">Calculate Prediction</button>
        </form>

        {% if prediction %}
        <div class="result-card">
            <span>Estimated Price</span>
            <strong>${{ prediction }}</strong>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Features matching model training order:
        # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
        features = [
            float(request.form["Square_Footage"]),
            float(request.form["Num_Bedrooms"]),
            float(request.form["Num_Bathrooms"]),
            float(request.form["Year_Built"]),
            float(request.form["Lot_Size"]),
            float(request.form["Garage_Size"]),
            float(request.form["Neighborhood_Quality"]),
        ]
        
        prediction_val = model.predict([features])[0]
        formatted_prediction = f"{prediction_val:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction=formatted_prediction)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

# Vercel Serverless Handler
app_handler = app
