# ⏱️ Optimizing Food Delivery Time Prediction

# 📂 Dataset
  - Sumber: Kaggle
  - Jumlah data: 1000 baris, 9 fitur
  - Fitur utama: Distance, Preparation Time, Courier Experience, Weather, Traffic Level, Vehicle Type, Time of Day
  - Target: Delivery Time (minutes)

# ⚙️ Modeling Approach
  - Preprocessing: Handling missing values, drop duplicates, outlier check
  - Encoding & Scaling dengan ColumnTransformer
  - Algorithms: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost
  - Hyperparameter Tuning

# 📊 Results

| Model                      | MAE   | RMSE  | R²         | MAPE   |   |
| -------------------------- | ----- | ----- | ---------- | ------ | - |
| Linear Regression          | 5.90  | 8.83  | 0.83       | 10.41% |   |
| Ridge / Lasso / ElasticNet | ~5.90 | ~8.83 | ~0.82–0.83 | ~10.4% |   |
| Random Forest              | 6.99  | 10.13 | 0.77       | 13.27% |   |
| XGBoost (tuned)            | 6.50  | 9.47  | 0.80       | 12.16% |   |

  - Model Terbaik: Linear Regression → MAE ~6 menit, R² = 0.83.
  - Faktor paling berpengaruh: Distance, Weather, Preparation Time, Traffic Level

![Predict vs Actual]
