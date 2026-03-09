**House Price Prediction using Machine Learning**

***Overview***

This project builds an **end-to-end machine learning pipeline** to predict housing prices using the Ames Housing dataset. The workflow covers data preprocessing, feature engineering, model comparison, hyperparameter tuning, and feature importance analysis.

The final model uses a **Random Forest Regressor and achieves an R² score of ~0.91**, explaining over 90% of the variance in housing prices.

**Project Structure**

house-price-prediction

├── data

     └── raw.csv


├── models

    ├── house_price_model.pkl

    └── scaler.pkl

├── notebooks

    └── house_price_prediction.ipynb


├── src

     └── train_model.py


├── requirements.txt

└── README.md

**WHERE**

1) data/ – dataset used for training
2) models/ – saved trained model and preprocessing objects
3) notebooks/ – exploratory analysis and experimentation
4) src/ – reproducible training pipeline
5) requirements.txt – project dependencies


**DATASET**

The project uses the Ames Housing dataset, which contains detailed information about residential homes such as:

Lot size

Living area

Garage capacity

Quality ratings

Basement features

Year built and renovation information

The dataset includes 80+ features describing different aspects of a property.

Target variable: SalePrice

***Machine Learning Pipeline***

The following steps were implemented in the training pipeline:

1. Data Cleaning

Removed identifier columns (e.g., PID)

Checked and handled missing values

2. Feature Engineering

Ordinal encoding for ordered categorical variables

One-hot encoding for nominal categorical variables

Conversion of numerical features to categorical where appropriate

3. Data Preprocessing

Feature scaling using StandardScaler

Handling categorical and numerical features separately

4. Model Training

Multiple regression models were evaluated:

Linear Regression

Support Vector Regression

SGD Regressor

K-Nearest Neighbors

Decision Tree Regressor

Gradient Boosting Regressor

Random Forest Regressor

Neural Network Regressor

5. Model Selection

Random Forest performed best among the evaluated models.

6. Hyperparameter Tuning

Randomized search with cross-validation was used to optimize model parameters.

7. Model Evaluation

Evaluation metric: R² Score

Final Performance R² ≈ 0.91

**FEATURE IMPORTANCE**

The model identified the most influential features affecting house prices:

| Feature                  | Importance  |
| ------------------------ | ----------- |
| Overall Quality          | Highest     |
| Above Ground Living Area | High        |
| First Floor Area         | Significant |
| Garage Capacity          | Moderate    |
| Lot Area                 | Moderate    |


These findings align with real estate expectations where property quality and living space strongly influence value.

***INSTALLATION***

clone the repo:  
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction

Install dependencies: pip install -r requirements.txt

**TRAIN THE MODEL**

python src/train_model.py

This will:

load the dataset

preprocess the data

train the model

evaluate performance

save the trained model in the models/ directory

***Technologies Used***

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

XGBoost

Joblib


**Key Learning Outcomes**

This project demonstrates:

End-to-end machine learning workflow

Data preprocessing and feature engineering

Model comparison across multiple algorithms

Cross-validation and hyperparameter tuning

Model interpretation using feature importance

Reproducible training pipelines

**Future Improvements**

Possible enhancements include:

Implementing XGBoost or LightGBM for improved performance

Creating a prediction API using FastAPI

Deploying the model using Docker or cloud services

Building an interactive dashboard for predictions


