# Spam-Ham-Email-SMS-detection-Model-
## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Model Details](#model-details)
6. [Results](#results)


## Introduction
This project aims to build a machine-learning model to detect spam in emails and SMS messages. The model is trained on a labelled dataset of emails and SMS messages to classify them as either "Spam" or "Ham" (not spam).

## Installation
To use this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Iamjuhwan/Spam-Ham-Email-SMS-detection-Model.git
cd Spam-Ham-Email-SMS-detection-Model
pip install -r requirements.txt

Training the Model: Use the Jupyter notebook Spam Detection.ipynb to train the model on your dataset. Open the notebook and follow the instructions to preprocess the data, train the model, and evaluate it.

Predicting: Use the stacking_class.pkl model to classify new messages. Example usage in Python:

## Usage 

import pickle

# Load the model
with open('models/stacking_class.pkl', 'rb') as file:
    model = pickle.load(file)

# Example message
message = "Your free lottery ticket is waiting for you!"

# Preprocess the message (apply the same preprocessing steps used during training)
# Example preprocessing function
def preprocess_message(message):
    # Add your preprocessing code here
    return processed_message

processed_message = preprocess_message(message)

# Make prediction
prediction = model.predict([processed_message])
print("Prediction:", "Spam" if prediction == 1 else "Ham")

## Project Structure

Spam-Ham-Email-SMS-detection-Model/
│
├── data/
│   └── emails_and_sms.csv       # Dataset
│
├── models/
│   └── stacking_class.pkl       # Trained model
│
├── notebooks/

│   └── Spam Detection.ipynb     # Jupyter notebook for model training
│
├── requirements.txt             # Dependencies

## Model Details
The model is built using the following steps:

Data Preprocessing: Text cleaning, tokenization, and vectorization.
Model Training: Training using a stacking classifier combining multiple algorithms.
Evaluation: Assessing the model's performance using accuracy, precision, recall, and F1-score metrics.

## Results
The model achieves an amazing accuracy. Detailed results and evaluation metrics are in the Spam Detection.ipynb notebook.
└── README.md                    # This file


