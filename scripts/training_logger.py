import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data", "model_inputs")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

def load_data(filename):
    """Load the data from the model_inputs folder."""
    filepath = os.path.join(DATA_DIR, filename)
    logging.info(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath, sep=";", decimal=",")
    logging.info(f"Data loaded successfully! Shape: {data.shape}")
    return data

def preprocess_data(data, target_variable):
    """Preprocess the data by separating features (X) and target (y)."""
    logging.info("Preprocessing data...")
    X = data.drop(columns=target_variable)
    y = data[target_variable]
    logging.info("Preprocessing completed!")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split completed! Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_filename="model.pkl", load_model=False):
    """Train or load a Random Forest model."""
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if load_model:
        logging.info(f"Loading model from {model_path}...")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info("Model loaded successfully!")
        else:
            logging.warning(f"Model file not found at {model_path}. Training a new model...")
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            logging.info(f"New model trained and saved to {model_path}.")
    else:
        logging.info("Training a new Random Forest model...")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        logging.info(f"Model trained and saved to {model_path}.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Evaluation Metrics -> MSE: {mse}, MAE: {mae}, R²: {r2}")
    return mse, mae, r2

def cross_validate_model(model, X_train, y_train):
    """Perform cross-validation."""
    logging.info("Performing cross-validation...")
    cv_results = cross_validate(model, X_train, y_train, cv=5,
                                 scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'))
    mean_r2 = cv_results['test_r2'].mean()
    mean_mse = -cv_results['test_neg_mean_squared_error'].mean()
    mean_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    logging.info(f"Cross-Validation Results -> Mean R²: {mean_r2}, Mean MSE: {mean_mse}, Mean MAE: {mean_mae}")
    return mean_r2, mean_mse, mean_mae

if __name__ == "__main__":
    input_filename = "ml_inputs.csv"
    target_variable = "VL_RECEITA_BRUTA"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = load_data(input_filename)
    X, y = preprocess_data(data, target_variable)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, model_filename="model.pkl", load_model=True)

    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    mean_r2, mean_mse, mean_mae = cross_validate_model(model, X_train, y_train)
