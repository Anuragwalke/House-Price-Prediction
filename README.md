
# House Price Prediction

A machine learning project that estimates house prices from property features using Python, Pandas, Scikit-learn, and Flask.

## Features

- Housing dataset exploration and preprocessing
- Categorical feature encoding
- Random Forest regression model
- Train/test split and model evaluation
- Flask web interface for price predictions

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Joblib

## Project Structure

```text
House_price_prediction/
├── Housing.csv
├── explore_data.py
├── train_model.py
├── house_price_model.pkl
├── app.py
├── templates/
│   └── index.html
└── README.md
```

## Model Evaluation

Current test-set results:

- R² score: 0.614
- MAE: 1,013,596.82
- RMSE: 1,396,543.92

These are results from one train/test split, not a guarantee of real-world performance.

## How to Run

1. Create and activate a Python virtual environment.
2. Install the required packages.
3. Run `python train_model.py`.
4. Run `python app.py`.
5. Open http://127.0.0.1:5000 in your browser.

## Dataset

The model uses the Housing.csv dataset. See the dataset source and license before redistributing the data.