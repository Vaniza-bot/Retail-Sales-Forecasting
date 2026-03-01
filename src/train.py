from preprocessing import load_data, clean_data
from feature_engineering import create_features
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from xgboost import XGBRegressor

# Load and process data
df = load_data("data/train.csv")
df = clean_data(df)
df = create_features(df)

# Target variable (log transform)
y = np.log1p(df['Sales'])

# Feature columns (now exist)
X = df[[
    'Order Month',
    'Order Year',
    'Order Quarter',
    'Order Day',
    'Day of Week',
    'Shipping Days',
    'Category_encoded',
    'Region_encoded',
    'Sub_Category_encoded',
    'Segment_encoded'
]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (XGBoost)
model = XGBRegressor(
    n_estimators=408,
    max_depth=4,
    learning_rate=0.01039,
    subsample=0.9438,
    colsample_bytree=0.6201
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
evaluate_model(y_test, y_pred)

# Save model
joblib.dump(model, "models/sales_model.pkl")

print("Model saved successfully!")