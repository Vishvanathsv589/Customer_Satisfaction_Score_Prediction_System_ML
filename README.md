📊 Customer Satisfaction Prediction System
🚀 Problem Statement

In today’s competitive business environment, customer satisfaction directly impacts retention, loyalty, and revenue.

Organizations collect large volumes of data from:

  Customer feedback (reviews, surveys, comments)
  Service quality metrics (delivery time, delays, etc.)
  Complaint history
  Product/service usage patterns

However, traditional feedback systems are:

  Reactive
  Slow
  Unable to identify root causes of dissatisfaction
  🎯 Objective

Build an AI-powered system that:

  Predicts customer satisfaction scores
  Identifies key drivers of dissatisfaction
  Provides actionable insights
  Enables proactive decision-making
  
💡 Solution Overview

This project uses the Olist Brazilian E-commerce dataset to build a machine learning pipeline that predicts customer review scores.

The system:

  Merges multiple datasets
  Engineers meaningful features
  Trains multiple regression models
  Evaluates performance using RMSE
  Extracts insights and recommendations
  
📁 Dataset

Source: Olist Brazilian E-commerce Dataset (Kaggle)

Data Files Used:
  Customers
  Orders
  Reviews
  Order Items
  Payments
  Products
  
⚙️ Feature Engineering

Key features created:

  📦 Delivery Time → Purchase → Delivery duration
  ⏱️ Delivery Delay → Actual vs Estimated delivery
  🛒 Number of Items → Items per order
  💰 Total Payment → Order value
  📐 Product Volume → Size proxy
  📝 Review Length → Customer sentiment indicator
  
🧠 Models Implemented

As required, the following models were used:

  ✅ Linear Regression
  ✅ Decision Tree Regressor
  ✅ Random Forest Regressor
  
🧪 Model Evaluation
📉 Metrics Used:
  Root Mean Squared Error (RMSE)
  5-Fold Cross Validation

This ensures:

Reliable performance comparison
Reduced overfitting
📊 Results
✔️ Outputs Generated:
  Satisfaction prediction model
  Predicted satisfaction score
  Cross-validation RMSE
  Test RMSE
  Feature importance (Random Forest)
  
📈 Feature Importance Insights

Top influencing factors:

  Delivery delay (strong negative impact)
  Total payment (higher expectations)
  Product volume (logistics complexity)
  Delivery time
  
🔮 Sample Prediction

Example input:

[delivery_time, delivery_delay, num_items, total_payment, product_volume, review_length]

Output:

Predicted Satisfaction Score: ~X.X

💡 Key Insights
  🚚 Delivery delays significantly reduce satisfaction
  💰 High-value customers demand better service
  📦 Large products increase delivery challenges
  📝 Long reviews often indicate dissatisfaction
  
🧾 Recommendations
  Optimize logistics to reduce delays
  Improve handling of large/fragile products
  Prioritize high-value customers
  Monitor long reviews for early issue detection
  
🛠️ Tech Stack
  Python
  Pandas
  NumPy
  Scikit-learn
  Matplotlib

(As per tool restrictions — no deep learning used)

▶️ How to Run
  git clone https://github.com/your-username/your-repo.git
  cd your-repo
  pip install pandas numpy matplotlib scikit-learn
  python main.py
  
📌 Project Requirements Checklist

  ✔ Feature Engineering
  ✔ Minimum 3 Models
  ✔ Cross Validation
  ✔ RMSE Evaluation
  ✔ Feature Importance
  ✔ Insights & Recommendations
  ✔ No Deep Learning

🔮 Future Improvements
  Hyperparameter tuning
  Advanced models (XGBoost, LightGBM)
  Real-time prediction system
  Dashboard for business users
