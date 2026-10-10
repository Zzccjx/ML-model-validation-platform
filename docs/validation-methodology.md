# Validation Methodology

## Overview
The AI/ML Model Validation & Assurance Platform performs rigorous, multi-dimensional empirical validation across seven standard pillars to assess model readiness, fairness, reliability, and security.

---

## 7 Core Validation Modules

### 1. Performance Validation (`performance`)
Evaluates core task predictive efficacy against ground truth labels:
- **Classification**: Accuracy, Balanced Accuracy, Macro/Weighted F1 Score, Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix, and per-class metrics.
- **Regression**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score, Mean Absolute Percentage Error (MAPE), and residual analysis.

### 2. Data Quality & Schema Integrity (`data_quality`)
Audits test dataset hygiene, schema consistency, and potential training artifacts:
- **Missingness**: Per-column and overall missing cell densities.
- **Duplicates**: Row duplication and identical duplicate feature detection.
- **Variance**: Constant ($Var=0$) and near-zero variance feature identification.
- **Outliers**: Extreme values and IQR distribution anomalies.
- **Class Balance**: Representation ratios and extreme imbalance detection.
- **Target Leakage**: Linear/Spearman correlation against target ($|r| \ge 0.95$).

### 3. Fairness & Demographic Parity (`fairness`)
Audits predictive disparity across sensitive group attributes (e.g., gender, race, age):
- **EEOC 80% Rule**: Disparate Impact Ratio evaluation ($\text{Selection Rate}_{\min} / \text{Selection Rate}_{\max}$).
- **Demographic Parity Difference**: Absolute acceptance rate difference across subgroups.
- **Equal Opportunity Difference**: True Positive Rate ($\Delta TPR$) parity across groups.
- **Equalized Odds Difference**: Maximum disparity across both TPR and FPR ($\max(\Delta TPR, \Delta FPR)$).
- **Subgroup Sample Warnings**: Flags statistical vulnerability when subgroup sample sizes $N < 30$.

### 4. Data & Prediction Drift Detection (`drift`)
Detects statistical divergence between baseline/reference distributions and current evaluation data:
- **Numerical Distribution Divergence**: Two-sample Kolmogorov-Smirnov (KS) test ($p$-value $< 0.05$).
- **Population Stability Index (PSI)**: 10 uniform quantile intervals ($\text{PSI} < 0.10$ Stable, $\text{PSI} \ge 0.25$ Actionable Shift).
- **Categorical Drift**: Chi-Square contingency testing and categorical PSI.
- **Prediction Drift**: Detects target and prediction output distribution shifts between sequential partitions.

### 5. Adversarial Robustness & Noise Sensitivity (`robustness`)
Assesses predictive stability against random perturbations and corrupted inputs:
- **Gaussian Jitter Perturbations**: Noise injection across $\sigma \in \{1\%, 5\%, 10\%, 20\%\}$ of feature standard deviation.
- **Prediction Flip Rate**: Measures fraction of decisions flipped under perturbation.
- **Missingness Tolerance**: Tests performance degradation under random feature masking ($10\%, 20\%$).
- **Stability Tiers**: Categorizes resilience into `RESILIENT`, `MODERATE`, or `VULNERABLE`.

### 6. Explainability & Feature Attribution (`explainability`)
Audits decision transparency and detects single-factor dominance:
- **Adaptive Multi-Model Attribution**: Tree `feature_importances_`, linear coefficients, or sample permutation importance fallback.
- **Normalized Relative Contribution**: Relative percentage share summing strictly to $100\%$.
- **Dominance Detection**: Flags models where a single feature dominates $> 60\%$ or $> 80\%$ of decisions.
- **Dead Feature Identification**: Highlights uninformative features ($\le 0.05\%$ attribution) as candidates for pipeline pruning.

### 7. Reliability & Probability Calibration (`reliability`)
Verifies predictive certainty and calibration against real empirical frequencies:
- **Brier Score**: Mean squared error of probabilistic predictions ($\frac{1}{N}\sum (p_i - y_i)^2$).
- **Expected Calibration Error (ECE)**: 10-interval discretization measuring calibration gaps.
- **Maximum Calibration Error (MCE)**: Maximum observed gap across probability intervals.
- **Overconfident Misclassifications**: Detects decisions made with $\ge 90\%$ certainty that were wrong.
- **Continuous Dispersion**: Residual standard error dispersion ratio and 95% empirical interval coverage for regression.

---

## Findings & Severity Framework
- **CRITICAL**: Immediate blockers (e.g., severe overconfidence $> 15\%$, potential target leakage $|r| \ge 0.98$, severe bias).
- **HIGH**: Substantial issues requiring remediation (e.g., $\text{ECE} > 0.15$, Brier $> 0.25$, significant drift $\text{PSI} \ge 0.25$).
- **MEDIUM**: Noticeable degradations or moderate calibration drift ($0.08 < \text{ECE} \le 0.15$).
- **LOW / INFO**: Minor observations and best-practice optimization recommendations.

---

## Responsible AI Disclaimer
This platform provides automated empirical indicators and statistical verification based on the supplied test artifacts. It does not provide legal certification of model compliance or guarantee safety under unseen real-world shifts.

