import sys
import os
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python main.py <path_to_parquet_file>")
    sys.exit(1)

FILE_NAME = sys.argv[1]
if not os.path.exists(FILE_NAME):
    print(f"File not found: {FILE_NAME}")
    sys.exit(1)

df = pd.read_parquet(FILE_NAME)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

target_candidates = [
    c for c in df.columns
    if "sample" in c.lower() and "purchase" in c.lower()
]

print("\nTarget candidates:", target_candidates)
assert len(target_candidates) == 1, "Could not uniquely detect purchase column"

TARGET_COL = target_candidates[0]
print("Using purchase column:", TARGET_COL)

df["purchased"] = (df[TARGET_COL] > 0).astype(int)

print("\nPurchase distribution:")
print(df["purchased"].value_counts())

features = [
    "sample_price",
    "cust_orders_lifetime",
    "cust_spend_lifetime",
    "cust_avg_order_value",
    "cust_recency_days",
    "cust_items_lifetime",
    "cust_distinct_brands",
    "cust_distinct_categories",
    "orders_in_category",
    "spend_in_category",
    "distinct_products_in_category",
    "days_since_category_purchase",
    "basket_value_current_order",
    "basket_size",
    "num_samples_in_order",
    "dow",
    "hour_of_day"
]

missing = [f for f in features if f not in df.columns]
print("\nMissing features:", missing)

for col in features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df[features] = df[features].fillna(0)

from sklearn.model_selection import train_test_split

X = df[features]
y = df["purchased"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Valid:", X_valid.shape)

import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50)]
)

from sklearn.metrics import roc_auc_score

val_preds = model.predict_proba(X_valid)[:,1]
print("\nValidation AUC:", roc_auc_score(y_valid, val_preds))

df["purchase_probability"] = model.predict_proba(X)[:,1]

print("\nSample scores:")
print(df[["purchase_probability"]].head())

ranked_customers = df.sort_values(
    "purchase_probability",
    ascending=False
)

# Optional: mark real buyers
ranked_customers["customer_type"] = ranked_customers.apply(
    lambda x: "Actual Buyer" if x[TARGET_COL] > 0 else "Predicted Buyer",
    axis=1
)

print("\nTop 10 customers:")
ranked_customers.head(10)

output_file = "ranked_customers_by_purchase_probability.csv"

ranked_customers.to_csv(output_file, index=False)

print("\nSaved:", output_file)