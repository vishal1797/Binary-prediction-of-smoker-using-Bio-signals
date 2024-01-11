from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('smoker_prediction.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Receive input from the form and make a prediction
# Receive input from the form and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        features = [float(request.form[str(i)]) for i in range(1, 23)]
        input_data = pd.DataFrame([features], columns=[
            'Age', 'Height', 'Weight', 'Waist', 'Eyesight_left', 'Eyesight_right',
            'Hearing_left', 'Hearing_right', 'Systolic', 'Diastolic',
            'Fasting_blood_sugar', 'Cholesterol_total', 'Triglyceride', 'HDL',
            'LDL', 'Hemoglobin', 'Urine_protein', 'Serum_creatinine', 'AST', 'ALT',
            'Gtp', 'Dental_caries'
        ])

        # Make a prediction
        prediction = pipeline.predict(input_data)[0]

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction="Error occurred. Please check input values.")


if __name__ == '__main__':
    app.run(debug=True)

























