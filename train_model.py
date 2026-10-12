
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("Housing.csv")

# 2. Separate features (X) and target (y)
X = df.drop(columns=["price"])
y = df["price"]

# 3. Identify numeric and text columns
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# 4. Convert text categories into numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),
        ("numeric", "passthrough", numeric_columns),
    ]
)

# 5. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Create the machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )),
    ]
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate it on data it has not trained on
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R² score: {r2:.3f}")

# 9. Save the trained model for our future website
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")