# train_and_export.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).parent
DATA = BASE / "data" / "Food_Delivery_Times.csv"
MODEL_DIR = BASE / "model"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model_linreg_pipeline.pkl"
CATS_PATH  = MODEL_DIR / "categories.json"

# kolom sesuai proyekmu
target = "Delivery_Time_min"
num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

df = pd.read_csv(DATA)
X = df[num_cols + cat_cols].copy()
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
])

pipe = Pipeline([
    ("preprocessor", pre),
    ("regressor", LinearRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)
print(f"MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.2f}")

joblib.dump(pipe, MODEL_PATH)
print("Saved:", MODEL_PATH)

# simpan kategori buat dropdown UI
ohe = None; used_cat_cols = None
for name, trans, cols in pipe.named_steps["preprocessor"].transformers_:
    if name == "cat":
        ohe, used_cat_cols = trans, cols
        break
cats = {c: cat.tolist() for c, cat in zip(used_cat_cols, ohe.categories_)}
with open(CATS_PATH, "w") as f:
    json.dump(cats, f)
print("Saved:", CATS_PATH)
