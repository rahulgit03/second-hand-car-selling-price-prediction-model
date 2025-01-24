# second-hand-car-selling-price-prediction-model
Project Title:
Second-Hand Car Price Prediction using Machine Learning

Project Overview:
This project aims to predict the selling price of second-hand cars based on various features such as brand, model, age, kilometers driven, fuel type, and transmission. The dataset was sourced from Kaggle, and multiple machine learning models were trained and evaluated.

Dataset:
Used a dataset from Kaggle containing details of second-hand cars.
Features include brand, model, vehicle age, kilometers driven, seller type, fuel type, transmission, mileage, engine capacity, max power, and seats.
Algorithms Used & Why:
Linear Regression – Used as a baseline model to understand linear relationships.
Decision Tree Regressor – Helps capture non-linear relationships but prone to overfitting.
Random Forest Regressor – Performed the best by reducing overfitting and improving generalization.
Support Vector Regression (SVR) – Used to explore how well margin-based learning works for this problem.
Preprocessing Steps:
Categorical Encoding: OneHotEncoding for categorical variables (brand, model, fuel type, etc.).
Missing Value Handling: Used SimpleImputer to replace missing numerical values with the mean.
Feature Scaling & Transformation: Applied necessary transformations to optimize model performance.
Model Evaluation:
Mean Squared Error (MSE) and R-squared were used as performance metrics.
Random Forest achieved the best balance between accuracy and generalization.
Results & Real-World Testing:
When tested in a local shop, the model’s predictions were nearly accurate compared to actual second-hand car prices.
How to Use the Model:
The script includes a user input-based prediction function where users can enter car details to get an estimated price.
Run predict_car_price() to interactively predict the price based on input features.
Future Improvements:
Enhancing feature engineering (e.g., considering car condition, region, demand-supply factors).
Experimenting with deep learning models for better prediction accuracy.
Deploying the model as a web application for real-time price estimation.
