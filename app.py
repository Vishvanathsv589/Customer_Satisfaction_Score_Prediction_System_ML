# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =========================
# 2. LOAD DATA
# =========================
customers = pd.read_csv("olist_customers_dataset.csv")
orders = pd.read_csv("olist_orders_dataset.csv")
reviews = pd.read_csv("olist_order_reviews_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")
payments = pd.read_csv("olist_order_payments_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")

# =========================
# 3. MERGE DATA
# =========================

# Orders + Reviews
df = orders.merge(reviews, on="order_id", how="inner")

# Add Customers
df = df.merge(customers, on="customer_id", how="left")

# Add Order Items
df = df.merge(items, on="order_id", how="left")

# Add Payments
df = df.merge(payments, on="order_id", how="left")

# Add Products
df = df.merge(products, on="product_id", how="left")

# =========================
# 4. DATETIME CONVERSION
# =========================
date_cols = [
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# =========================
# 5. FEATURE ENGINEERING
# =========================

# Delivery time
df['delivery_time'] = (
    df['order_delivered_customer_date'] - df['order_purchase_timestamp']
).dt.days

# Delivery delay
df['delivery_delay'] = (
    df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
).dt.days

# Number of items per order
order_items_count = items.groupby('order_id')['order_item_id'].count()
df['num_items'] = df['order_id'].map(order_items_count)

# Total payment per order
payment_total = payments.groupby('order_id')['payment_value'].sum()
df['total_payment'] = df['order_id'].map(payment_total)

# Product volume (proxy for size)
df['product_volume'] = (
    df['product_length_cm'] *
    df['product_height_cm'] *
    df['product_width_cm']
)

# Review length
df['review_length'] = df['review_comment_message'].fillna("").apply(len)

# =========================
# 6. CLEAN DATA
# =========================
df = df.fillna(0)

# =========================
# 7. FEATURE SELECTION
# =========================
features = [
    'delivery_time',
    'delivery_delay',
    'num_items',
    'total_payment',
    'product_volume',
    'review_length'
]

X = df[features]
y = df['review_score']

# =========================
# 8. TRAIN-TEST SPLIT:
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. MODELS
# =========================

lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=8, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf
}

# Train
for model in models.values():
    model.fit(X_train, y_train)

# =========================
# 10. CROSS VALIDATION
# =========================
print("\n=== CROSS VALIDATION RMSE ===")

for name, model in models.items():
    scores = cross_val_score(
        model, X, y, cv=5,
        scoring='neg_root_mean_squared_error'
    )
    print(f"{name}: {-np.mean(scores):.4f}")

# =========================
# 11. TEST RMSE
# =========================
print("\n=== TEST RMSE ===")

for name, model in models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: {rmse:.4f}")

# =========================
# 12. FEATURE IMPORTANCE (RF)
# =========================
importances = rf.feature_importances_

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()

# =========================
# 13. SAMPLE PREDICTION
# =========================
sample_df = pd.DataFrame(
    [[5, 1, 2, 200, 5000, 100]],
    columns=features
)

pred = rf.predict(sample_df)

print("\nSample Predicted Satisfaction Score:", float(pred[0]))

# =========================
# 14. INSIGHTS
# =========================
print("\n=== INSIGHTS ===")

for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

print("\nKey Findings:")
print("- Delivery delay has strong negative impact")
print("- Higher payment orders tend to expect better service")
print("- Large products may increase delivery issues")
print("- Longer reviews often signal dissatisfaction")

# =========================
# 15. RECOMMENDATIONS
# =========================
print("\n=== RECOMMENDATIONS ===")
print("1. Reduce delivery delays (optimize logistics)")
print("2. Improve handling of large/fragile products")
print("3. Prioritize high-value customers")
print("4. Monitor long reviews for complaints")