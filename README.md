# Fraud Detection

An AI-powered web application that detects potentially fraudulent credit card transactions using machine learning. The project uses a Random Forest classifier to analyze transaction patterns and predict whether a transaction is likely fraudulent.

## Overview
Fraud detection is a critical problem in financial systems, where even a small percentage of false negatives can result in substantial losses. This project aims to build a robust fraud detection system that combines machine learning with an interactive dashboard for real-time transaction analysis.

The application uses:
- Random Forest for binary classification
- FastAPI backend for model serving and API endpoints
- HTML, CSS, and JavaScript for the frontend dashboard
- Real-time fraud probability prediction and transaction monitoring

## Features
- Detects fraudulent transactions using a trained machine learning model
- Predicts fraud probability for new transactions
- Provides a cybersecurity-style interactive dashboard
- Displays transaction insights in a user-friendly interface
- FastAPI-based backend for efficient API handling
- Suitable for real-time analysis and demo deployment

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Random Forest
- Pandas
- HTML
- CSS
- JavaScript

## Project Workflow
1. Load and preprocess transaction data
2. Train a Random Forest model on historical data
3. Evaluate model performance
4. Use the trained model to predict fraud probability
5. Expose predictions through a FastAPI API
6. Visualize results through a web dashboard

## Installation
```bash
git clone https://github.com/ABASKAR-N/Fraud-Detection.git
cd Fraud-Detection
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:8000`.

## Deployment
This application is deployed on Render: https://fraud-detection-11ar.onrender.com/

## Model Performance
The Random Forest model is trained on historical transaction data and achieves high accuracy in detecting fraudulent transactions.

## License
MIT License
