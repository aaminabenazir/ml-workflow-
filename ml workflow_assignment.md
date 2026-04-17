

## Task 1: Label and Data Leakage Identification

*Label (Target Variable):* repeat_purchase_flag  
This is the label because it directly indicates the outcome we want to predict — whether a customer will make a repeat purchase within 30 days (1) or not (0).

*Column that would introduce data leakage if used as a feature:* days_since_last_order  
Including days_since_last_order as a feature would cause data leakage. This column contains information that is too closely tied to the target (or potentially unavailable at prediction time). If we are predicting whether a repeat purchase will happen within the next 30 days from a given point, knowing how many days have already passed since the last order makes the prediction partially trivial or uses information that wouldn't realistically be available in the same way during real-time inference, leading to overly optimistic model performance that won't generalize.

## Task 2: Steps to Complete Before Training a Gradient Boosting Model

1. *Perform Exploratory Data Analysis (EDA)*  
   Before jumping into a complex model like gradient boosting, we should thoroughly explore the dataset. This includes checking for missing values, outliers, class imbalance in the target (repeat_purchase_flag), feature distributions, and correlations. EDA helps identify data quality issues, understand business context, and guide proper feature engineering — preventing wasted effort on a model built on flawed data.

2. *Data Splitting and Preprocessing (with proper validation strategy)*  
   We must split the data into training and validation/test sets (ideally using a time-based or customer-based split to respect the temporal nature of orders). Preprocessing steps such as handling missing values, scaling numerical features (avg_order_value, order_count_last_90d, etc.), and ensuring no leakage during feature engineering should be done only on the training set. This ensures honest model evaluation and prevents overfitting or unrealistic performance estimates when moving to gradient boosting.

These foundational steps ensure the model is built on clean, well-understood data and evaluated fairly, which is critical before applying any advanced algorithm.
