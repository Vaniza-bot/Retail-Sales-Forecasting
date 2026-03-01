import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from preprocessing import load_data, clean_data
from feature_engineering import create_features
from evaluate import evaluate_model
import numpy as np

# Load data
df = load_data("data/train.csv")
df = clean_data(df)
df = create_features(df)

# Target and features
y = np.log1p(df['Sales'])
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

# Objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Return error (Optuna minimizes)
    return evaluate_model(y_test, y_pred)

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)