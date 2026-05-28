import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/matches_clean.csv')

print("Building win probability model...")

# Features for win prediction
features = ['team1', 'team2', 'toss_won', 'decision', 'venue_name', 'season']
target = 'winner'

model_df = df[features + [target]].dropna()

# Encode categorical columns
encoders = {}
for col in ['team1', 'team2', 'toss_won', 'decision', 'venue_name', 'winner']:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {round(acc * 100, 1)}%")

# Save model and encoders
joblib.dump(model, 'models/win_model.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
print("Model saved to models/")