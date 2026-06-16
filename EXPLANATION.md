# EXPLANATION.md

## Mandatory Questions

### 1. What percentage of customers in your dataset have y = yes? What does this imbalance mean for how you'd evaluate a model?

**11.70%** subscribed (5,289 out of 45,211). The imbalance ratio is **7.5:1**.

This makes standard accuracy a trap metric. A model that blindly predicts "no" for every customer achieves 88.3% accuracy while identifying zero subscribers — operationally useless for a bank trying to allocate Relationship Manager time. The only metrics that matter for business impact are:

- **F1-Score** — forces the model to balance precision (not wasting RM time on false leads) and recall (not missing actual subscribers).
- **PR-AUC** (Precision-Recall Area Under Curve) — evaluates the model's ranking ability across all decision thresholds on the minority class specifically, unlike ROC-AUC which is inflated by the dominant negative class.

Any evaluation that leads with accuracy on this dataset is a red flag for production readiness.

### 2. Which job category had the highest subscription rate? Does this make sense to you intuitively?

**Students** at **28.68%**, followed by **retired** at **22.79%**.

This maps directly to financial liquidity and risk appetite:

- **Students** have minimal existing financial obligations (no mortgages, no dependents). They have low balances but high marginal utility from even small, safe savings products like term deposits. Banks target them as lifetime customer acquisition.
- **Retirees** are capital-preservation oriented. They actively seek low-risk, guaranteed-return instruments. A term deposit is exactly the product profile that matches their risk tolerance — fixed returns, no market exposure.

The lowest rate — **blue-collar** at 7.27% — aligns with higher existing debt loads (housing loans, personal loans) consuming disposable income.

---

## Track B Questions

### 3. Which feature had the highest importance in your tree-based model? Why do you think that is?

The top features in the XGBoost model:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `contact_unknown` | 0.153 |
| 2 | `poutcome_success` | 0.144 |
| 3 | `poutcome_unknown` | 0.093 |
| 4 | `month_mar` | 0.047 |
| 5 | `housing_yes` | 0.044 |
| 6 | `duration` | 0.041 |

**`contact_unknown`** dominates because it is a hard negative signal — if the bank has no valid contact method, the customer is effectively unreachable during a campaign. **`poutcome_success`** is the strongest positive signal — a customer who already converted in a previous campaign has a proven track record.

**Critical caveat on `duration` (Data Leakage):**

The `duration` feature (call length in seconds) ranks 6th at 0.041 importance. While mathematically predictive, it represents a **catastrophic data leakage risk** in any production deployment. Here is why:

- `duration` is the length of the telemarketing call **in seconds**. It is only known **after** the call ends.
- In a real bank, Relationship Managers use this model **before** making the call to decide **who to call**. At that point, `duration` does not exist.
- A model that relies on `duration` for its predictions cannot be used for **pre-call lead prioritization** — the exact use case this system is built for.
- The UCI dataset documentation explicitly warns: *"this attribute highly affects the output target... duration is not known before a call is performed."*

For benchmark purposes (as intended by the dataset authors), I kept `duration` in training. For a production deployment, it must be **removed entirely** and the model retrained and re-evaluated on the remaining features. Any model shipped with `duration` as an input is fundamentally broken for the stated business objective.

### 4. Why is F1 a better metric than accuracy for this particular dataset?

Because the dataset is 88/12 imbalanced, accuracy rewards a model for predicting the majority class. A model with 88.3% accuracy and 0% recall on subscribers is worse than useless — it creates a false sense of performance while missing every potential sale.

F1-score is the harmonic mean of Precision and Recall. It forces the model to simultaneously:
- **Precision**: Not waste RM time calling people who will not subscribe (false positives cost call-hours).
- **Recall**: Not miss customers who would subscribe (false negatives cost revenue).

In banking operations, the cost asymmetry between these two errors depends on RM capacity. F1 treats them equally, making it the baseline metric. For fine-tuning the trade-off, the **Precision-Recall curve** (saved in `images/precision_recall_curve.png`) provides the full picture.

### 5. Pick one of your 5 sample predictions. Do you actually agree with the model's call? Walk through your thinking.

I am analyzing a **borderline** prediction to demonstrate model calibration rather than picking an obvious case.

**Customer 3098** — Predicted: **no** | Probability: **17.49%** | Actual: **no**

| Feature | Value |
|---------|-------|
| Age | 30 |
| Job | blue-collar |
| Marital | single |
| Education | secondary |
| Balance | 609 |
| Housing | yes |
| Loan | no |
| Contact | unknown |
| Duration | 402s |
| Poutcome | unknown |

**Why this is interesting:** This customer has **conflicting signals**. The call duration of 402 seconds is well above the average for "no" customers (221s), which would normally push toward a "yes" prediction. The customer is also single (fewer financial obligations) with no personal loan — both mild positive signals.

**Why the model correctly predicted "no" at 17.49%:**

1. **`contact_unknown`** — the model's #1 feature. The bank doesn't have a reliable contact channel for this customer. This is the single strongest negative signal in the entire model.
2. **`poutcome_unknown`** — no prior campaign history. No conversion track record.
3. **`housing = yes`** — existing mortgage obligations reduce disposable income available for a deposit product.
4. **`job = blue-collar`** — the lowest-converting job category at 7.27%. Combined with secondary education, this demographic profile has the weakest subscription rate in the dataset.
5. **`balance = 609`** — below the dataset median (448 is the median, but this customer also has a mortgage eating into it).

The model correctly weighted the strong negatives (`contact_unknown`, `poutcome_unknown`, blue-collar demographic) over the single conflicting signal (`duration`). The 17.49% probability is not a confident "no" — it reflects genuine uncertainty. This is good calibration: the model isn't overconfident, it just correctly determined that the negative signals outweigh the positives. In production, this customer would sit below any reasonable decision threshold (typically 40-50%) and would not be prioritized for outreach.
