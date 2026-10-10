"""Data Quality Validator module.

Analyzes dataset integrity, missing value rates, duplicate rows/columns,
feature variance, numerical extremes, outliers, class balance, and potential
target leakage indicators.
"""
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


class DataQualityValidator(BaseValidator):
    """Validator assessing dataset quality and statistical health."""

    name = "data_quality"
    version = "1.0.0"
    category = "data_quality"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when a non-empty dataset is present."""
        return context.dataset is not None and len(context.dataset) > 0

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute comprehensive data quality validation."""
        df: pd.DataFrame = context.dataset
        target_col = context.target_column
        feature_cols = [c for c in df.columns if c != target_col]

        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        total_rows = len(df)
        total_cells = df.size
        total_missing = int(df.isna().sum().sum())
        overall_missing_pct = float((total_missing / total_cells) * 100) if total_cells > 0 else 0.0

        metrics["row_count"] = total_rows
        metrics["column_count"] = len(df.columns)
        metrics["overall_missing_percentage"] = round(overall_missing_pct, 2)

        # -------------------------------------------------------------
        # 1. Missing Values Analysis
        # -------------------------------------------------------------
        missing_by_col = {}
        high_missing_cols = []
        mod_missing_cols = []
        for col in df.columns:
            m_count = int(df[col].isna().sum())
            if m_count > 0:
                m_pct = round((m_count / total_rows) * 100, 2)
                missing_by_col[col] = {"count": m_count, "percentage": m_pct}
                if m_pct >= 20.0:
                    high_missing_cols.append((col, m_pct))
                elif m_pct >= 5.0:
                    mod_missing_cols.append((col, m_pct))

        metrics["missing_values"] = missing_by_col

        for col, pct in high_missing_cols:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="data_quality",
                    title=f"High Missing Rate in Column '{col}' ({pct}%)",
                    description=f"Column '{col}' is missing {pct}% of its values ({missing_by_col[col]['count']}/{total_rows} rows).",
                    metric_name=f"Missing % ({col})",
                    metric_value=pct,
                    threshold="< 20.0%",
                    recommendation="Consider imputing missing values or dropping column if missingness is completely random.",
                )
            )

        for col, pct in mod_missing_cols:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="data_quality",
                    title=f"Moderate Missing Rate in Column '{col}' ({pct}%)",
                    description=f"Column '{col}' has {pct}% missing values.",
                    metric_name=f"Missing % ({col})",
                    metric_value=pct,
                    threshold="< 5.0%",
                    recommendation="Apply robust imputation (median, mean, or KNN imputer) prior to training.",
                )
            )

        # -------------------------------------------------------------
        # 2. Duplicate Records & Duplicate Columns
        # -------------------------------------------------------------
        dup_rows = int(df.duplicated().sum())
        dup_pct = round((dup_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0
        metrics["duplicate_rows_count"] = dup_rows
        metrics["duplicate_rows_percentage"] = dup_pct

        if dup_pct >= 10.0:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="data_quality",
                    title=f"Severe Record Duplication: {dup_pct}% Duplicate Rows",
                    description=f"Dataset contains {dup_rows} duplicate rows ({dup_pct}%). This may artificially inflate validation scores.",
                    metric_name="Duplicate Rows %",
                    metric_value=dup_pct,
                    threshold="< 2.0%",
                    recommendation="Deduplicate dataset records to prevent data contamination across splits.",
                )
            )
        elif dup_pct >= 2.0:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="data_quality",
                    title=f"Moderate Record Duplication ({dup_pct}%)",
                    description=f"{dup_rows} identical records detected in dataset.",
                    metric_name="Duplicate Rows %",
                    metric_value=dup_pct,
                    threshold="< 2.0%",
                    recommendation="Verify whether duplicate rows represent organic repeat measurements or collection artifacts.",
                )
            )

        # Check duplicate columns (identical data across columns)
        duplicate_col_pairs = []
        cols_list = list(df.columns)
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                if df[c1].equals(df[c2]):
                    duplicate_col_pairs.append((c1, c2))

        metrics["duplicate_column_pairs"] = duplicate_col_pairs
        if duplicate_col_pairs:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="data_quality",
                    title=f"Duplicate Columns Detected: {len(duplicate_col_pairs)} pair(s)",
                    description=f"Columns share identical values: {duplicate_col_pairs}.",
                    metric_name="Duplicate Column Pairs",
                    metric_value=float(len(duplicate_col_pairs)),
                    threshold="0 pairs",
                    recommendation="Drop redundant identical columns to streamline feature space.",
                )
            )

        # -------------------------------------------------------------
        # 3. Variance & Constant Features
        # -------------------------------------------------------------
        constant_cols = []
        near_zero_variance_cols = []
        feature_stats = {}

        for col in feature_cols:
            s = df[col].dropna()
            n_unique = int(s.nunique())
            is_num = bool(pd.api.types.is_numeric_dtype(df[col]))

            variance = None
            if is_num and len(s) > 1:
                variance = float(s.var())

            feature_stats[col] = {
                "dtype": str(df[col].dtype),
                "unique_count": n_unique,
                "variance": round(variance, 6) if variance is not None else None,
                "is_numeric": is_num,
            }

            if n_unique <= 1:
                constant_cols.append(col)
            elif is_num and variance is not None and variance < 1e-5:
                near_zero_variance_cols.append((col, variance))
            elif not is_num and n_unique > 1:
                top_freq = s.value_counts().iloc[0]
                if (top_freq / len(s)) > 0.98:
                    near_zero_variance_cols.append((col, top_freq / len(s)))

        metrics["constant_columns"] = constant_cols
        metrics["near_zero_variance_columns"] = [c[0] for c in near_zero_variance_cols]

        if constant_cols:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="data_quality",
                    title=f"Constant Feature(s) Detected: {constant_cols}",
                    description=f"{len(constant_cols)} feature column(s) have zero variance and carry no discriminative signal.",
                    metric_name="Constant Column Count",
                    metric_value=float(len(constant_cols)),
                    threshold="0",
                    recommendation="Remove constant features from the input pipeline.",
                )
            )

        if near_zero_variance_cols:
            findings.append(
                ValidationFinding(
                    severity="low",
                    category="data_quality",
                    title=f"Near-Zero Variance in {len(near_zero_variance_cols)} Feature(s)",
                    description=f"Features with almost no variation: {[c[0] for c in near_zero_variance_cols]}.",
                    metric_name="Near-Zero Variance Count",
                    metric_value=float(len(near_zero_variance_cols)),
                    threshold="0",
                    recommendation="Consider variance thresholding or principal component selection.",
                )
            )

        # -------------------------------------------------------------
        # 4. Outliers, Infinite & Extreme Values
        # -------------------------------------------------------------
        infinite_cols = []
        outlier_counts = {}
        total_outliers = 0
        num_feature_count = 0

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_feature_count += 1
                s = df[col].dropna()
                # Infinite values check
                inf_count = int(np.isinf(s).sum())
                if inf_count > 0:
                    infinite_cols.append((col, inf_count))

                # IQR Outlier check
                if len(s) >= 4:
                    q1 = float(s.quantile(0.25))
                    q3 = float(s.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 1e-7:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        o_count = int(((s < lower_bound) | (s > upper_bound)).sum())
                        if o_count > 0:
                            outlier_counts[col] = o_count
                            total_outliers += o_count

        metrics["infinite_columns"] = infinite_cols
        metrics["outlier_counts"] = outlier_counts
        avg_outlier_pct = round((total_outliers / (num_feature_count * total_rows)) * 100, 2) if num_feature_count * total_rows > 0 else 0.0
        metrics["outlier_percentage"] = avg_outlier_pct

        if infinite_cols:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="data_quality",
                    title="Infinite Values in Features",
                    description=f"Infinite values detected in columns: {infinite_cols}.",
                    metric_name="Infinite Column Count",
                    metric_value=float(len(infinite_cols)),
                    threshold="0",
                    recommendation="Clip numerical values or investigate division-by-zero occurrences.",
                )
            )

        if avg_outlier_pct >= 10.0:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="data_quality",
                    title=f"High Outlier Density ({avg_outlier_pct}% of Numerical Cells)",
                    description="Multiple features display significant outlier concentrations under standard 1.5*IQR bounds.",
                    metric_name="Outlier %",
                    metric_value=avg_outlier_pct,
                    threshold="< 5.0%",
                    recommendation="Apply robust scalers (RobustScaler) or winsorization if training sensitive linear models.",
                )
            )

        # -------------------------------------------------------------
        # 5. Potential Target Leakage Detection
        # -------------------------------------------------------------
        leakage_cols = []
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            # If target can be cast to float/int
            target_num = None
            if pd.api.types.is_numeric_dtype(target_series):
                target_num = target_series.astype(float)
            elif target_series.nunique() == 2:
                # Binary string/categorical converted to 0/1
                vals = target_series.unique()
                target_num = (target_series == vals[1]).astype(float)

            if target_num is not None:
                for col in feature_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        valid_idx = df[col].notna() & df[target_col].notna()
                        if valid_idx.sum() > 10:
                            s_feat = df.loc[valid_idx, col].astype(float)
                            s_targ = target_num.loc[valid_idx]
                            if s_feat.std() > 1e-6 and s_targ.std() > 1e-6:
                                corr = abs(float(np.corrcoef(s_feat, s_targ)[0, 1]))
                                feature_stats[col]["target_correlation"] = round(corr, 4)
                                if corr >= 0.95:
                                    leakage_cols.append((col, round(corr, 4)))

        metrics["potential_leakage_columns"] = leakage_cols
        for col, corr_val in leakage_cols:
            sev = "critical" if corr_val >= 0.98 else "high"
            findings.append(
                ValidationFinding(
                    severity=sev,
                    category="data_quality",
                    title=f"Potential Data Leakage Detected: '{col}' (|r|={corr_val})",
                    description=(
                        f"Feature '{col}' exhibits an extraordinarily high linear association ({corr_val}) "
                        f"with target '{target_col}'. This indicates potential target leakage, duplicate target column, "
                        "or a variable recorded post-decision."
                    ),
                    metric_name=f"Correlation ({col} ~ {target_col})",
                    metric_value=corr_val,
                    threshold="< 0.95",
                    recommendation=f"Review feature origin for '{col}'. Exclude it if it cannot be observed prior to inference time.",
                )
            )

        # -------------------------------------------------------------
        # 6. Class Imbalance (Classification)
        # -------------------------------------------------------------
        imbalance_level = "N/A"
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            is_class = context.model_type == "classification" or target_series.nunique() <= 20
            if is_class and target_series.nunique() > 1:
                counts = target_series.value_counts()
                majority_cnt = int(counts.iloc[0])
                minority_cnt = int(counts.iloc[-1])
                imb_ratio = round(majority_cnt / max(1, minority_cnt), 2)
                minority_pct = round((minority_cnt / len(target_series)) * 100, 2)

                if minority_pct < 5.0 or imb_ratio >= 15.0:
                    imbalance_level = "EXTREME"
                    findings.append(
                        ValidationFinding(
                            severity="high",
                            category="data_quality",
                            title=f"Extreme Class Imbalance (Ratio {imb_ratio}:1)",
                            description=(
                                f"Minority class '{counts.index[-1]}' represents only {minority_pct}% "
                                f"({minority_cnt}/{len(target_series)} rows)."
                            ),
                            metric_name="Imbalance Ratio",
                            metric_value=imb_ratio,
                            threshold="< 10.0:1",
                            recommendation="Use stratified sampling, balanced class weights, or focal loss to prevent majority bias.",
                        )
                    )
                elif minority_pct < 15.0 or imb_ratio >= 4.0:
                    imbalance_level = "HIGH"
                    findings.append(
                        ValidationFinding(
                            severity="medium",
                            category="data_quality",
                            title=f"Significant Class Imbalance (Ratio {imb_ratio}:1)",
                            description=f"Minority class '{counts.index[-1]}' comprises {minority_pct}% of samples.",
                            metric_name="Imbalance Ratio",
                            metric_value=imb_ratio,
                            threshold="< 4.0:1",
                            recommendation="Monitor balanced accuracy and consider SMOTE or threshold optimization.",
                        )
                    )
                elif imb_ratio >= 1.5:
                    imbalance_level = "MODERATE"
                else:
                    imbalance_level = "BALANCED"

                metrics["class_imbalance"] = {
                    "level": imbalance_level,
                    "ratio": imb_ratio,
                    "minority_percentage": minority_pct,
                    "class_counts": {str(k): int(v) for k, v in counts.items()},
                }

        metrics["imbalance_level"] = imbalance_level
        metrics["feature_stats"] = feature_stats

        # -------------------------------------------------------------
        # 7. Category Score Calculation (0 - 100)
        # -------------------------------------------------------------
        score = 100.0
        score -= min(25.0, overall_missing_pct * 2.5)
        score -= min(15.0, dup_pct * 1.5)
        score -= min(20.0, len(constant_cols) * 10.0)
        score -= min(10.0, len(near_zero_variance_cols) * 2.5)
        score -= min(10.0, avg_outlier_pct * 0.5)
        if leakage_cols:
            score -= 25.0
        if infinite_cols:
            score -= 15.0
        if imbalance_level == "EXTREME":
            score -= 10.0
        elif imbalance_level == "HIGH":
            score -= 5.0

        final_score = float(np.clip(score, 0.0, 100.0))

        # Recommendations
        recommendations.append(
            f"Dataset Quality Summary: Missing={overall_missing_pct:.1f}%, Duplicates={dup_pct:.1f}%, "
            f"Constant Columns={len(constant_cols)}, Outlier Density={avg_outlier_pct:.1f}%, Imbalance={imbalance_level}."
        )

        return ValidatorResult(
            validator_name=self.name,
            validator_version=self.version,
            status=ValidatorStatus.SUCCESS,
            score=round(final_score, 1),
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            category=self.category,
        )
