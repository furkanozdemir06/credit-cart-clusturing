# Credit Card Customer Segmentation

An unsupervised machine learning project that segments credit card customers by their usage behavior. The pipeline covers data exploration, feature engineering, scaling, PCA, K-Means clustering, and a business interpretation of the resulting segments.

## Overview

Credit card companies hold rich behavioral data but no ready-made customer groups. This project uses clustering to discover groups of customers who spend, borrow, and pay in similar ways, which can support targeted marketing, credit risk analysis, and customer relationship management.

## Dataset

The dataset (`CC GENERAL.csv`) contains 8,950 credit card holders and 18 columns describing their behavior over the previous six months.

| Group | Example columns |
|-------|-----------------|
| Balance and limit | `BALANCE`, `BALANCE_FREQUENCY`, `CREDIT_LIMIT` |
| Purchases | `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_FREQUENCY`, `PURCHASES_TRX` |
| Cash advances | `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX` |
| Payments | `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT` |
| Other | `CUST_ID`, `TENURE` |

Missing values: `MINIMUM_PAYMENTS` (313) and `CREDIT_LIMIT` (1).

## Workflow

1. **Exploratory data analysis**
   - Checked shape, data types, missing values, and summary statistics.
   - Plotted distributions of balance, purchases, credit limit, and purchase frequency, a purchases vs. payments scatter plot, and a correlation heatmap.
2. **Preprocessing and feature engineering**
   - Dropped `CUST_ID` and filled missing values with the median.
   - Engineered 7 ratio features: `PURCHASES_PER_TRX`, `CASH_ADVANCE_PER_TRX`, `PAYMENT_TO_BALANCE`, `MINPAY_TO_BALANCE`, `CASH_ADVANCE_SHARE`, `INSTALLMENT_SHARE`, and `ONEOFF_SHARE`.
   - Applied a `log1p` transform to reduce skewness, then scaled with `RobustScaler` to limit the effect of outliers.
3. **Dimensionality reduction (PCA)**
   - Examined cumulative explained variance, then projected the data onto 3 principal components for visualization.
4. **Clustering (K-Means)**
   - Evaluated 2 to 9 clusters using silhouette scores and the elbow method (`yellowbrick`).
   - Trained the final model with 3 clusters on the scaled feature set.
5. **Cluster profiling**
   - Computed the average value of each original feature per cluster to interpret the segments.

## Results

Silhouette scores for different numbers of clusters:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| Silhouette | 0.443 | 0.259 | 0.300 | 0.296 | 0.262 | 0.250 | 0.252 | 0.251 |

The final model uses k = 3. Note that k = 3 does not have the highest silhouette score (k = 2 does), so the choice reflects a trade-off between statistical separation and having more than two segments to act on.

### Customer segments

| Cluster | Customers | Avg. balance | Avg. purchases | Purchase frequency | Avg. cash advance | Profile |
|---------|-----------|--------------|----------------|--------------------|-------------------|---------|
| 0 | 1,247 | 98.9 | 347.1 | 0.29 | 339.3 | **Low-activity customers:** very low balances, rarely update their balance (balance frequency 0.35), and make few purchases |
| 1 | 5,274 | 1,432.6 | 1,596.2 | 0.73 | 525.6 | **Active shoppers:** high and frequent purchases (about 23 transactions on average), including both one-off and installment purchases |
| 2 | 2,429 | 2,603.2 | 52.5 | 0.07 | 2,291.3 | **Cash-advance users:** carry high balances and rely heavily on cash advances (about 7.7 advance transactions) while barely using the card for purchases |

The segment names are interpretations based on the cluster averages. Possible uses: re-engagement offers for cluster 0, rewards and installment promotions for cluster 1, and closer risk monitoring or lower-cost credit products for cluster 2.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (K-Means, PCA, scaling, silhouette score)
- yellowbrick (elbow visualization)
- matplotlib, seaborn
## Author

Furkan
