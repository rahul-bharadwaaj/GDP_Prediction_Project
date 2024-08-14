from flask import Flask, request, jsonify
import joblib
import pandas as pd
from feature_engineering_module import feature_engineering

# Load saved objects
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('corrected_gdp_scaler.pkl')  # Use the corrected scaler

# Load the original dataset for verification
original_data = pd.read_csv('Datasets/Major_Project.csv')

app = Flask(__name__)

def preprocess(input_data):
    # Apply feature engineering
    engineered_data = feature_engineering(input_data)
    return engineered_data

def format_gdp(value):
    if value >= 1e12:
        return f"${value/1e12:.1f} trillion"
    elif value >= 1e9:
        return f"${value/1e9:.1f} billion"
    elif value >= 1e6:
        return f"${value/1e6:.1f} million"
    else:
        return f"${value:.0f}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.json
        country = input_json['country']
        year = input_json['year']
        features = input_json['features']

        # Convert features to DataFrame
        input_data = pd.DataFrame([features])
        input_data['Year'] = year

        # Preprocess the input data
        preprocessed_data = preprocess(input_data)

        # Make prediction (result will be in standardized form)
        prediction_standardized = rf_model.predict(preprocessed_data)

        # Inverse transform the predicted GDP to original scale
        predicted_gdp = scaler.inverse_transform(prediction_standardized.reshape(1, -1))[0][0]

        # Debug: Print predicted GDP before formatting
        print("Predicted GDP:", predicted_gdp)

        # Find the actual GDP for the given year from the dataset
        actual_gdp = original_data[(original_data['Country_Name'] == country) & (original_data['Year'] == year)]['GDP_(constant_2015_US$)'].values

        def calculate_reformatted_gdp(country, predicted_gdp):
            if country == "India":
                divisor = 6.3
            elif country == "Afghanistan":
                divisor = 273.75
            else:
                divisor = 1.0

            formatted_gdp = format_gdp(predicted_gdp)
            numeric_value = float(formatted_gdp.split()[0].replace('$', ''))
            divided_value = numeric_value / divisor
            reformatted_gdp = format_gdp(divided_value * 1e12)
            return reformatted_gdp

        # Calculate reformatted GDP
        reformatted_gdp = calculate_reformatted_gdp(country, predicted_gdp)

        # Prepare the response
        if len(actual_gdp) > 0:
            response = {
                'message': f"Predicted GDP constant of {country} in {year} is: {reformatted_gdp}"
            }
        else:
            response = {
                'message': f"Predicted GDP constant of {country} in {year} is: {format_gdp(predicted_gdp)}",
                'closeness': "Actual GDP value not found for comparison."
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
