# Binary-prediction-of-smoker-using-Bio-signals

## Overview

This project is part of a Kaggle competition aiming to predict whether an individual is a smoker or not based on bio signals. The dataset includes various bio-related features such as age, height, weight, blood pressure, cholesterol levels, etc.

## Dataset Overview

The dataset consists of both training and testing datasets. The features include age, height, weight, blood pressure, cholesterol levels, and more. The target variable is 'smoking,' indicating whether an individual is a smoker or not.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution of variables, check for missing values, and explore correlations. Visualizations such as correlation heatmaps and histograms were used to gain insights into the dataset.

## Model Development

The RandomForestClassifier was chosen as the predictive model. The development process includes hyperparameter tuning using GridSearchCV, model training, and evaluation.

### Hyperparameter Tuning

A subset of the dataset was used for hyperparameter tuning to reduce computation time. The best hyperparameters were determined using GridSearchCV.

### Model Training

The final model was trained on the entire training dataset using the selected hyperparameters.

### AUC-ROC Evaluation

The model's performance was evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) to assess its ability to discriminate between classes.

## Deployment

The project was deployed using Flask, and the model was saved as a pickle file for ease of use.

## Files in the Repository

- `train.csv`: Training dataset
- `test.csv`: Testing dataset
- `app.py`: Flask application for deployment
- `index.html`: HTML file for the main page
- `result.html`: HTML file for displaying predictions
- `predictions.csv`: CSV file containing the model predictions
- `requirements.txt`: List of project dependencies

## Usage

To use the project, follow these steps:

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Flask application using `python app.py`.
3. Access the application through the provided URL.

## Results

The model achieved a certain accuracy score on the validation set and was further evaluated using the AUC-ROC metric.

