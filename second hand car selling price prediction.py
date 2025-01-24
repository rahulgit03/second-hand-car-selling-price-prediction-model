# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Loading Data
X = df[['brand', 'model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission', 'mileage', 'engine', 'max_power', 'seats']]
y = df['selling_price']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission']
numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model Training
pipeline_lr = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
pipeline_dt = Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))])
pipeline_rf = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))])
pipeline_svr = Pipeline([('preprocessor', preprocessor), ('model', SVR())])

# Model Evaluation
models = [
    ('Linear Regression', pipeline_lr),
    ('Decision Tree', pipeline_dt),
    ('Random Forest', pipeline_rf),
    ('Support Vector Regression', pipeline_svr)
]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R-squared: {r2}\n")

# Prediction Function
def predict_car_price():
    brand = input("Enter car brand: ") 
    model = input("Enter car model: ")
    vehicle_age = int(input("Enter vehicle age: "))
    km_driven = float(input("Enter kilometers driven: "))
    seller_type = input("Enter seller type (Individual/Dealer): ")
    fuel_type = input("Enter fuel type (Petrol/Diesel): ")
    transmission = input("Enter transmission (Manual/Automatic): ")
    mileage = float(input("Enter mileage: "))
    engine = int(input("Enter engine capacity (cc): "))
    max_power = float(input("Enter maximum power (bhp): "))
    seats = int(input("Enter number of seats: "))

    sample_data = pd.DataFrame({
        'brand': [brand],
        'model': [model],
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'seller_type': [seller_type],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    best_model = pipeline_rf 

    try:
        predicted_price = best_model.predict(sample_data)[0]
        print(f"Predicted car price: ₹{predicted_price:.2f}") 
    except ValueError as e:
        print(f"Error: {e}") 

# Calling Prediction Function
predict_car_price()
