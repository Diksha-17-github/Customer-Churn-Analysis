from flask import Flask, request, render_template
import pandas as pd
import pickle
import logging

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.sav", "rb"))

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the replacement mapping for categorical columns
replacements = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
    'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 0},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Contract': {'Month-to-month': 1, 'One year': 2, 'Two year': 3},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 1,
        'Mailed check': 2,
        'Bank transfer (automatic)': 3,
        'Credit card (automatic)': 4
    }
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    data = [
        [
            request.form['query1'],  # SeniorCitizen
            request.form['query2'],  # MonthlyCharges
            request.form['query3'],  # TotalCharges
            request.form['query4'],  # gender
            request.form['query5'],  # Partner
            request.form['query6'],  # Dependents
            request.form['query7'],  # PhoneService
            request.form['query8'],  # MultipleLines
            request.form['query9'],  # InternetService
            request.form['query10'],  # OnlineSecurity
            request.form['query11'],  # OnlineBackup
            request.form['query12'],  # DeviceProtection
            request.form['query13'],  # TechSupport
            request.form['query14'],  # StreamingTV
            request.form['query15'],  # StreamingMovies
            request.form['query16'],  # Contract
            request.form['query17'],  # PaperlessBilling
            request.form['query18'],  # PaymentMethod
            request.form['query19']   # tenure
        ]
    ]

    # Prepare the input DataFrame
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'tenure'
    ])

    # Convert relevant columns to numeric and handle NaNs
    new_df['MonthlyCharges'] = pd.to_numeric(new_df['MonthlyCharges'], errors='coerce')
    new_df['TotalCharges'] = pd.to_numeric(new_df['TotalCharges'], errors='coerce')
    new_df['tenure'] = pd.to_numeric(new_df['tenure'], errors='coerce')
    new_df['SeniorCitizen'] = pd.to_numeric(new_df['SeniorCitizen'], errors='coerce')

    new_df.fillna({
        'MonthlyCharges': new_df['MonthlyCharges'].mean(),
        'TotalCharges': new_df['TotalCharges'].mean(),
        'tenure': 0,
        'SeniorCitizen': 0
    }, inplace=True)

    # Convert categorical columns to numeric
    for col, values in replacements.items():
        new_df[col] = new_df[col].replace(values)

    # Log input values for debugging
    logging.info("Input values: %s", data)

    # Log processed DataFrame before prediction
    logging.info("Processed Input DataFrame:\n%s", new_df)

    # Ensure the columns match the training set
    expected_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'tenure'
    ]

    # Add missing columns with default values
    for col in expected_columns:
        if col not in new_df.columns:
            new_df[col] = 0  # Default value for missing columns

    # Reorder columns to match the training DataFrame
    new_df = new_df[expected_columns]

    # Make prediction
    try:
        single = model.predict(new_df)
        probability = model.predict_proba(new_df)
        
        # Log prediction probabilities
        logging.info("Prediction probabilities: %s", probability)
        
        # Get predicted class and probability for the first input
        predicted_class = single[0]
        predicted_probability = probability[0][1]  # Probability of class 1
        
        logging.info("Prediction: %s, Probability: %s", predicted_class, predicted_probability)
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return "Prediction failed. Please check the input data."

    # Output the prediction and confidence
    if predicted_class == 1:
        output1 = "customer will not churn"
        output2 = "Confidence: {:.2f}%".format(predicted_probability * 100)
    else:
        output1 = "customer will churn"
        output2 = "Confidence: {:.2f}%".format(predicted_probability * 100)

    return render_template('home.html', output1=output1, output2=output2)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
