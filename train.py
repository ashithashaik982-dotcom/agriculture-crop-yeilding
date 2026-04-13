import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/Egypt_Crop_Yield_Processed_Pivot.csv")

df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

y = df["Yield"]
X = df.drop("Yield", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("✅ Model trained successfully!")