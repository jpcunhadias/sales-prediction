# sales-prediction

# Training Logger

This project is designed to load input data, preprocess it, train a predictive model (or load an existing one), and evaluate its performance using Python and Docker. The pipeline processes data stored in `data/model_inputs` and saves the model in the `models` directory.

---

## **Project Structure**

```
.
├── README.md # Project documentation

├── data # Directory for all data files
│   ├── model_inputs # Input data for the model
│   ├── processed # (Optional) Intermediate processed data
│   └── raw # (Optional) Raw unprocessed data

├── dockerfile # Docker configuration for containerizing the project

├── models # Directory to save/load trained models

├── notebooks # Jupyter Notebooks for analysis and modeling
│   ├── 01-eda.ipynb # Exploratory Data Analysis notebook
│   └── 02-modeling.ipynb # Model building and experimentation notebook

├── outputs # Directory for storing results, logs, and metrics

├── requirements.txt # Python dependencies for the project

└── scripts # Directory for scripts
    └── training_logger.py # Main script for the ML pipeline
```

---

## **Pipeline Overview**

1. **Load Data:**
   - Input data must be placed in the `data/model_inputs` directory as a `.csv` file.

2. **Preprocess Data:**
   - Separates features (`X`) and target (`y`) and splits data into training and testing sets.

3. **Train or Load Model:**
   - Trains a new model if no existing model is found in the `models` directory.
   - Saves the trained model as `model.pkl`.

4. **Evaluate Model:**
   - Evaluates performance using metrics such as MSE, MAE, and R².

---

## **How to Run**

### **With Python**

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
2. Run the pipeline

   ```bash
   python scripts/training_logger.py

### **With Docker**

1. Build Docker Image

   ```bash
   docker build -t training-logger .
2. Run the container

   ```bash
   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models training-logger
