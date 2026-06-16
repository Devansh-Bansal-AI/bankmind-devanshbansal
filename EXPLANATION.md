# EXPLANATION.md

## Mandatory Questions

### 1. What percentage of customers in your dataset have y = yes? What does this imbalance mean for how you'd evaluate a model?

Only **11.70%** of customers subscribed (`y = yes`), while 88.30% did not. The imbalance ratio is 7.5:1.

This means a naive model that always predicts "no" would achieve 88.3% accuracy — while identifying exactly zero potential subscribers. Accuracy is therefore misleading as a primary metric. For this dataset I evaluate using **F1-score** (which penalises both false positives and false negatives) and **PR-AUC** (Precision-Recall Area Under Curve), both of which force the model to prove it can actually detect the minority "yes" class.

### 2. Which job category had the highest subscription rate? Does this make sense to you intuitively?

**Students** had the highest subscription rate at **28.68%**, followed by **retired** at **22.79%**.

This is intuitive. Students are at the beginning of their financial journey — they typically have fewer existing products and lower financial complexity, making them receptive to straightforward savings instruments like term deposits. Retirees, on the other hand, tend to favour low-risk, guaranteed-return products to preserve capital, which is exactly what a term deposit offers.

---

## Track B Questions

### 3. Which feature had the highest importance in your tree-based model? Why do you think that is?

The two most important features in the XGBoost model were:

| Feature            | Importance |
|--------------------|------------|
| `contact_unknown`  | 0.153      |
| `poutcome_success` | 0.144      |

`contact_unknown` is highest because if the bank doesn't even have a valid contact method for the customer, the chance of them subscribing through a campaign is negligible — it's a strong negative signal. `poutcome_success` is the second because if a customer already subscribed in a previous campaign, they have a proven track record of converting. These two features essentially split customers into "unreachable / unlikely" vs. "has said yes before" — the most decisive information the model has.

**Important caveat on `duration`:** The `duration` feature (call length) ranks 6th in importance at 0.041. However, the UCI dataset documentation explicitly warns: *"this attribute highly affects the output target... duration is not known before a call is performed."* In a real production model deployed to Relationship Managers *before* they make the call, `duration` would not exist as an input. Its presence here inflates model performance and constitutes a form of **data leakage**. For a truly production-ready system, the model should be retrained without `duration` and evaluated on its remaining features alone. I kept it here for benchmark comparability as the dataset authors intended, but any deployment would require its removal.

### 4. Why is F1 a better metric than accuracy for this particular dataset?


Because the dataset is 88/12 imbalanced, accuracy rewards a model for simply predicting the majority class. F1-score is the harmonic mean of Precision and Recall — it forces the model to simultaneously:
- **Precision**: not waste Relationship Managers' time on false positives
- **Recall**: not miss customers who would actually subscribe

A model with 88% accuracy and 0% recall on the "yes" class is operationally useless. F1 catches this failure; accuracy hides it.

### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call? Walk through your thinking.

I'll examine **Sample 4 (Customer 1392)**:

| Feature   | Value        |
|-----------|--------------|
| Age       | 40           |
| Job       | blue-collar  |
| Marital   | married      |
| Education | primary      |
| Balance   | 640          |
| Housing   | yes          |
| Loan      | yes          |
| Contact   | unknown      |
| Duration  | 347          |
| Poutcome  | unknown      |
| **Predicted** | **no (0.16%)** |
| **Actual**    | **no**         |

**I fully agree with this prediction.** Here's my reasoning:

1. **Both housing and personal loan active** — this customer is already heavily leveraged. Taking on a new financial product (a term deposit that locks up cash) is unlikely when existing debt obligations are consuming disposable income.
2. **Contact method is unknown** — the model's #1 feature. The bank couldn't reliably reach this person through the campaign.
3. **Previous campaign outcome is unknown** — no positive history of conversion.
4. **Balance of only 640** — with two active loans and only 640 average yearly balance, there is minimal surplus to commit to a deposit product.
5. **Blue-collar job with primary education** — this demographic had one of the lowest subscription rates (7.27%).

Every signal points the same way. The model's 0.16% confidence is not just correct — it's well-calibrated.
