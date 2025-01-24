Second-Hand Car Price Prediction using Machine Learning

Project Overview

This project aims to predict the selling price of second-hand cars based on various features such as brand, model, age, kilometers driven, fuel type, transmission, mileage, engine capacity, and power. The dataset was sourced from Kaggle, and multiple machine learning models were trained and evaluated. The project was implemented using Google Colab and is based on fundamental machine learning concepts like regression, data preprocessing, and model evaluation.

Dataset & Features
The dataset was downloaded from Kaggle and contains various details about second-hand cars. The target variable is selling price, and the features are categorized as follows:

1️⃣ Categorical Features (Non-Numeric Data)
These features were encoded using OneHotEncoding to convert them into numerical form:

Brand – The car's manufacturer (e.g., Maruti, Hyundai, Honda).
Model – The specific model of the car.
Seller Type – Whether the seller is an Individual or Dealer.
Fuel Type – The type of fuel used: Petrol, Diesel, CNG, Electric.
Transmission – Whether the car has Manual or Automatic transmission.

2️⃣ Numerical Features (Continuous & Discrete Values)
These features were imputed using SimpleImputer (Mean Strategy) to handle missing values:

Vehicle Age – Age of the car in years (calculated from the manufacturing year).
Kilometers Driven – The total distance traveled by the car.
Mileage (km/l) – Fuel efficiency of the car.
Engine (cc) – Engine displacement in cubic centimeters.
Max Power (bhp) – Maximum power output of the engine.
Seats – Number of seats in the car.

Machine Learning Models Used
This project was developed using basic machine learning techniques, focusing on supervised learning (regression). The following models were trained and evaluated:

Linear Regression – Used as a baseline model to understand linear relationships.
Decision Tree Regressor – Captures non-linear patterns but prone to overfitting.
Random Forest Regressor – Achieved the best accuracy by reducing overfitting and improving generalization.
Support Vector Regression (SVR) – Explored how well margin-based learning works for this problem.

Preprocessing & Model Training


🔹 Data Preprocessing Steps:
OneHotEncoding was applied to categorical features.
Missing values in numerical columns were filled using SimpleImputer (Mean Strategy).
Feature scaling and transformations were handled within a Scikit-learn Pipeline.

🔹 Model Evaluation:
Mean Squared Error (MSE) and R-squared (R²) were used as evaluation metrics.
Random Forest Regressor achieved the highest R² score and lowest MSE, making it the best model.

Results & Real-World Testing
The trained model was tested in a local shop, where it provided nearly accurate price estimates, demonstrating its practical applicability.

Tools & Technologies Used
Google Colab – Used for development and model training.
Pandas & NumPy – For data handling and preprocessing.
Scikit-learn – For machine learning models and evaluation.
Matplotlib & Seaborn – For data visualization and analysis.
