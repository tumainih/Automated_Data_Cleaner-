"""
pipeline.py
Core data-cleaning / ML engine for the Smart Data Cleaner platform.

This module consolidates and refactors the original Group 13 scripts
(Data_import.py, Data_consistency.py, Duplicates.py, FillContinous.py,
FillCategorical.py, FillDate.py, FillMissingToid.py, OutlierDetection.py,
Missing_summary.py, Sorting_missing.py) into pure, side-effect-free functions
that can be safely called from a Streamlit app (no input()/print() calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)


# =========================================================
# 1. DATASET PROFILING (column type detection)
# =========================================================

def detect_identity_columns(df: pd.DataFrame) -> list:
    ids = []
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue

        date_ratio = pd.to_datetime(s, errors="coerce", format="mixed").notna().mean()
        if date_ratio > 0.8:
            continue

        unique_ratio = s.nunique() / len(s)
        name = col.lower()
        name_check = any(x in name for x in ["id", "code", "key", "email", "phone", "mobile", "account", "name"])
        person_name_check = "name" in name and not pd.api.types.is_numeric_dtype(df[col])
        email_check = s.str.match(r"^[\w\.-]+@[\w\.-]+\.\w+$").mean() > 0.8
        phone_check = s.str.match(r"^[+0-9][0-9\-\s]+$").mean() > 0.8

        if (unique_ratio >= 0.90 and (name_check or email_check or phone_check)) or person_name_check:
            ids.append(col)
    return ids


def detect_date_columns(df: pd.DataFrame) -> list:
    dates = []
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        # Date logic (per feedback): use ONLY parsing ratio; do NOT rely on name keywords.
        date_ratio = pd.to_datetime(s, errors="coerce", format="mixed").notna().mean()
        if date_ratio >= 0.90:
            dates.append(col)
    return dates



def detect_continuous_columns(df: pd.DataFrame, exclude: list = None) -> list:
    exclude = exclude or []
    continuous = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() < 0.90:
            continue
        numeric = numeric.dropna()
        if len(numeric) == 0:
            continue
        unique = numeric.nunique()
        unique_ratio = unique / len(numeric)
        top_frequency = numeric.value_counts().max() / len(numeric)
        if unique > 10 and unique_ratio >= 0.10 and top_frequency < 0.20:
            continuous.append(col)
    return continuous


def detect_categorical_columns(df: pd.DataFrame, exclude: list = None) -> list:
    exclude = exclude or []
    categorical = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_ratio = numeric.notna().mean()
        unique = s.nunique()
        unique_ratio = unique / len(s)
        top_frequency = s.value_counts().max() / len(s)

        text_category = numeric_ratio < 0.90 and unique_ratio <= 0.50
        numeric_category = numeric_ratio >= 0.90 and unique <= 10
        repeated_category = top_frequency >= 0.05 and unique <= 50

        if text_category or numeric_category or repeated_category:
            categorical.append(col)
    return categorical


def profile_dataset(df: pd.DataFrame) -> dict:
    """Detect column roles and cast dtypes accordingly. Returns dict with keys
    Identity, Date, Continuous, Categorical, Data."""
    df = df.copy()

    identity_cols = detect_identity_columns(df)
    date_cols = detect_date_columns(df)
    continuous_cols = detect_continuous_columns(df, exclude=identity_cols + date_cols)
    categorical_cols = detect_categorical_columns(
        df, exclude=identity_cols + date_cols + continuous_cols
    )

    for col in identity_cols:
        df[col] = df[col].astype("string")
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return {
        "Identity": identity_cols,
        "Date": date_cols,
        "Continuous": continuous_cols,
        "Categorical": categorical_cols,
        "Data": df,
    }


# =========================================================
# 2. MISSING VALUE SUMMARY
# =========================================================

def missing_value_report(df: pd.DataFrame):
    column_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isna().sum().values,
        "Missing (%)": (df.isna().mean() * 100).round(2).values,
    }).sort_values("Missing Values", ascending=False).reset_index(drop=True)

    row_missing = df.isna().sum(axis=1)
    row_summary = (
        row_missing.value_counts().sort_index()
        .rename_axis("Missing Values per Row")
        .reset_index(name="Number of Rows")
    )
    return column_report, row_summary


# =========================================================
# 3. DUPLICATE HANDLING
# =========================================================

def review_duplicates(df: pd.DataFrame, identity_cols: list, resolve_identity_duplicates: bool = False):
    """Removes full-row duplicates always. Optionally resolves identity-column
    duplicates by keeping the most complete row. Returns (df, report dict)."""
    df = df.copy()
    report = {}

    row_duplicates = int(df.duplicated().sum())
    report["full_row_duplicates_removed"] = row_duplicates
    if row_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    duplicate_ids = {}
    for col in identity_cols:
        dup_mask = df[col].duplicated(keep=False)
        count = int(dup_mask.sum())
        if count > 0:
            duplicate_ids[col] = df.loc[dup_mask, col].astype(str).unique().tolist()[:20]

    report["identity_duplicates_found"] = duplicate_ids
    report["identity_duplicates_resolved"] = {}

    if duplicate_ids and resolve_identity_duplicates:
        df["_missing_count"] = df.isna().sum(axis=1)
        for col in duplicate_ids.keys():
            before = len(df)
            df = (
                df.sort_values("_missing_count")
                .drop_duplicates(subset=[col], keep="first")
            )
            removed = before - len(df)
            report["identity_duplicates_resolved"][col] = removed
        df = df.drop(columns="_missing_count").reset_index(drop=True)

    return df, report


# =========================================================
# 4. DATA CONSISTENCY / STANDARDIZATION
# =========================================================

def standardize_dataset(
    df: pd.DataFrame,
    identity_cols: list,
    date_cols: list,
    continuous_cols: list,
    categorical_cols: list,
    clip_negative_continuous: bool = False,
    max_categories: int = 20,
):
    """Standardizes text case, dates, numeric symbols, and category cardinality.
    clip_negative_continuous: if True, converts negative numeric values to NaN
    (useful for columns that should never be negative, e.g. age, price)."""
    df = df.copy()
    log = []

    # Identity / text cleaning
    for col in identity_cols:
        s = df[col].astype("string")
        s = s.str.replace(r"[^a-zA-Z0-9\s@.+-]", "", regex=True)
        upper_ratio = s.dropna().str.isupper().mean() if s.dropna().shape[0] else 0
        lower_ratio = s.dropna().str.islower().mean() if s.dropna().shape[0] else 0
        if upper_ratio > 0.7:
            s = s.str.upper()
        elif lower_ratio > 0.7:
            s = s.str.lower()
        df[col] = s

    # Date cleaning
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    # Continuous cleaning
    for col in continuous_cols:
        original = df[col].astype("string")
        cleaned = original.str.replace(r"[,()%$]", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")

        invalid_count = int(numeric.isna().sum() - df[col].isna().sum())
        if invalid_count > 0:
            log.append(f"{col}: {invalid_count} invalid values converted to NaN")

        negative_count = int((numeric < 0).sum())
        if negative_count > 0:
            if clip_negative_continuous:
                numeric.loc[numeric < 0] = np.nan
                log.append(f"{col}: {negative_count} negative values replaced with NaN")
            else:
                log.append(f"{col}: {negative_count} negative values retained")

        df[col] = numeric.astype(float)

    # Categorical cleaning
    for col in categorical_cols:
        s = df[col].astype("string")
        s = s.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        s = s.str.lower()
        if s.dropna().empty:
            df[col] = s.astype("category")
            continue
        categories = s.value_counts().head(max_categories).index
        rare_count = int((~s.isin(categories) & s.notna()).sum())
        if rare_count > 0:
            log.append(f"{col}: {rare_count} rare category values grouped as missing")
        s = s.where(s.isin(categories), np.nan)
        df[col] = s.astype("category")

    return df, log


# =========================================================
# 5. MACHINE-LEARNING IMPUTATION
# =========================================================

def _select_features(df, predictors, target_train_idx, importance_model, y_train, rows_total):
    X_all = df[predictors].copy()
    for col in X_all.columns:
        if X_all[col].dtype.name == "category":
            X_all[col] = X_all[col].cat.codes
    X_all = X_all.select_dtypes(include=np.number)
    if X_all.shape[1] == 0:
        return None, None

    X_train = X_all.loc[target_train_idx].fillna(X_all.loc[target_train_idx].median())
    importance_model.fit(X_train, y_train)
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    keep = max(1, int(len(importance) * (0.20 if rows_total < 1000 else 0.30)))
    selected = importance.head(keep)["Feature"].tolist()
    return X_all, selected


def fill_continuous_missing_rf(df, continuous_cols, identity_cols, date_cols, min_training_ratio=0.0):
    df = df.copy()
    log = []
    for target in continuous_cols:
        available_rows = df[target].notna().sum()
        total_rows = len(df)
        training_ratio = available_rows / total_rows if total_rows else 0
        if training_ratio == 0:
            log.append(f"{target}: skipped (no training data available)")
            continue
        if training_ratio < min_training_ratio:
            log.append(f"{target}: skipped (only {training_ratio:.1%} training data available)")
            continue

        predictors = [c for c in df.columns if c not in identity_cols + date_cols + [target]]
        train_idx = df[target].notna()
        y_train = df.loc[train_idx, target]

        selector = RandomForestRegressor(n_estimators=100, random_state=42)
        X_all, selected_features = _select_features(df, predictors, train_idx, selector, y_train, total_rows)
        if X_all is None:
            df[target] = df[target].fillna(df[target].median())
            log.append(f"{target}: no numeric predictors — filled with median")
            continue

        X_train_final = X_all.loc[train_idx, selected_features].fillna(X_all[selected_features].median())
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train_final, y_train)

        missing_rows = df[df[target].isna()].index
        if len(missing_rows) == 0:
            continue

        X_missing = X_all.loc[missing_rows, selected_features].fillna(X_train_final.median())
        predictions = model.predict(X_missing)

        original_values = df[target].dropna()
        is_integer = np.all(original_values % 1 == 0) if len(original_values) else False
        if is_integer:
            predictions = np.round(predictions).astype(int)
            df[target] = df[target].astype("Int64")
        df.loc[missing_rows, target] = predictions

        log.append(f"{target}: filled {len(missing_rows)} values using predictors {selected_features}")
    return df, log


def fill_categorical_missing_rf(df, categorical_cols, identity_cols, date_cols, min_training_ratio=0.0):
    df = df.copy()
    log = []
    for target in categorical_cols:
        available_rows = df[target].notna().sum()
        total_rows = len(df)
        training_ratio = available_rows / total_rows if total_rows else 0
        if training_ratio == 0:
            log.append(f"{target}: skipped (no training data available)")
            continue
        if training_ratio < min_training_ratio:
            log.append(f"{target}: skipped (only {training_ratio:.1%} training data available)")
            continue

        predictors = [c for c in df.columns if c not in identity_cols + date_cols + [target]]
        train_idx = df[target].notna()
        y_train = df.loc[train_idx, target].astype(str)

        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        X_all, selected_features = _select_features(df, predictors, train_idx, selector, y_train, total_rows)
        if X_all is None:
            mode = df[target].mode()
            df[target] = df[target].fillna(mode.iloc[0] if not mode.empty else "Unknown")
            log.append(f"{target}: no usable predictors — filled with mode")
            continue

        X_train_final = X_all.loc[train_idx, selected_features].fillna(X_all[selected_features].median())
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train_final, y_train)

        missing_rows = df[df[target].isna()].index
        if len(missing_rows) == 0:
            continue

        X_missing = X_all.loc[missing_rows, selected_features].fillna(X_train_final.median())
        predictions = model.predict(X_missing)
        df.loc[missing_rows, target] = predictions
        df[target] = df[target].astype("category")

        log.append(f"{target}: filled {len(missing_rows)} values using predictors {selected_features}")
    return df, log


def fill_missing_dates(df, date_cols, strategy="unknown"):
    """strategy: 'unknown' -> mark as NaT/'Unknown', 'ffill', 'bfill', or 'mode'."""
    df = df.copy()
    log = []
    for col in date_cols:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        if strategy == "ffill":
            df[col] = df[col].ffill().bfill()
        elif strategy == "bfill":
            df[col] = df[col].bfill().ffill()
        elif strategy == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        log.append(f"{col}: {missing} missing dates handled using '{strategy}' strategy")
    return df, log


def fill_missing_identity(df, identity_cols):
    df = df.copy()
    log = []
    for col in identity_cols:
        missing_rows = df[df[col].isna()].index.tolist()
        if len(missing_rows) == 0:
            continue
        for row in missing_rows:
            df.at[row, col] = f"Unknown_{row}"
        log.append(f"{col}: filled {len(missing_rows)} missing identity values")
    return df, log


# =========================================================
# 6. OUTLIER DETECTION (Isolation Forest)
# =========================================================

def detect_outliers_isolation_forest(
    df, continuous_cols, categorical_cols, contamination=0.05
):
    df = df.copy()
    results = {}

    if len(continuous_cols) > 0:
        X_cont = df[continuous_cols].copy()
        X_cont = X_cont.fillna(X_cont.median())
        if X_cont.shape[1] > 0 and len(X_cont) > 1:
            model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            prediction = model.fit_predict(X_cont)
            scores = model.decision_function(X_cont)
            df["continuous_outlier"] = prediction == -1
            df["continuous_anomaly_score"] = scores
            results["continuous"] = int(df["continuous_outlier"].sum())

    if len(categorical_cols) > 0:
        X_cat = pd.DataFrame(index=df.index)
        for col in categorical_cols:
            freq = df[col].value_counts(normalize=True)
            X_cat[col] = df[col].map(freq)
        X_cat = X_cat.fillna(0)
        if X_cat.shape[1] > 0 and len(X_cat) > 1:
            model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            prediction = model.fit_predict(X_cat)
            scores = model.decision_function(X_cat)
            df["categorical_outlier"] = prediction == -1
            df["categorical_anomaly_score"] = scores
            results["categorical"] = int(df["categorical_outlier"].sum())

    flags = [c for c in ["continuous_outlier", "categorical_outlier"] if c in df]
    if flags:
        df["overall_outlier"] = df[flags].any(axis=1)
        results["overall"] = int(df["overall_outlier"].sum())

    return df, results


# =========================================================
# 7. ROW COMPLETENESS SORTING
# =========================================================

def sort_rows_by_completeness(df, identity_cols, date_cols, continuous_cols, categorical_cols):
    temp = df.copy()
    temp["total_missing"] = temp.isna().sum(axis=1)
    temp["identity_missing"] = temp[identity_cols].isna().sum(axis=1) if identity_cols else 0
    temp["date_missing"] = temp[date_cols].isna().sum(axis=1) if date_cols else 0
    temp["continuous_missing"] = temp[continuous_cols].isna().sum(axis=1) if continuous_cols else 0
    temp["categorical_missing"] = temp[categorical_cols].isna().sum(axis=1) if categorical_cols else 0

    sorted_df = temp.sort_values(
        by=["total_missing", "identity_missing", "date_missing", "continuous_missing", "categorical_missing"]
    ).drop(columns=["total_missing", "identity_missing", "date_missing", "continuous_missing", "categorical_missing"])
    return sorted_df


# =========================================================
# 8. DATA QUALITY SCORE
# =========================================================

def _mask_values_for_evaluation(df: pd.DataFrame, cols: list, mask_fraction: float, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Create a synthetic evaluation scenario by masking (setting to NaN) a fraction
    of existing non-null values for the provided columns.

    Returns: (masked_df, mask_meta) where mask_meta maps col -> list of row indices masked."""
    rng = np.random.default_rng(seed)
    masked_df = df.copy()
    mask_meta: dict[str, list[int]] = {}
    for col in cols:
        if col not in masked_df.columns:
            continue
        non_null_idx = masked_df.index[masked_df[col].notna()].to_numpy()
        if non_null_idx.size == 0:
            continue
        n_mask = max(1, int(round(mask_fraction * non_null_idx.size)))
        chosen = rng.choice(non_null_idx, size=min(n_mask, non_null_idx.size), replace=False)
        chosen = chosen.tolist()
        masked_df.loc[chosen, col] = np.nan
        mask_meta[col] = chosen
    return masked_df, mask_meta


def evaluate_imputation_synthetic(
    df_before_imputation: pd.DataFrame,
    df_after_imputation: pd.DataFrame,
    identity_cols: list,
    date_cols: list,
    continuous_cols: list,
    categorical_cols: list,
    mask_fraction: float = 0.1,
    seed: int = 42,
    min_samples: int = 10,
) -> dict:
    """Synthetic imputation evaluation via random masking.

    We assume `df_after_imputation` is the output of the imputation pipeline.
    We will:
      1) mask a fraction of *known* values in `df_after_imputation` (treat it as ground truth)
      2) run the existing imputation functions to estimate the masked values
      3) compare estimates with the ground truth (the original values before masking).

    For simplicity and robustness, we reuse existing pipeline functions:
    - fill_continuous_missing_rf
    - fill_categorical_missing_rf
    - fill_missing_dates (mode/ffill/bfill/unknown handled as 'mode' here)
    - fill_missing_identity (keeps 'Unknown_<row>' placeholder)

    Returns dict with computed metrics."""
    # Use post-imputation df as ground-truth for evaluation
    df_gt = df_after_imputation.copy()

    # Choose evaluation columns
    cont_eval = [c for c in continuous_cols if c in df_gt.columns]
    cat_eval = [c for c in categorical_cols if c in df_gt.columns]

    rng_cols = cont_eval + cat_eval
    if not rng_cols:
        return {"html_block": "<p>No continuous/categorical columns available for synthetic imputation evaluation.</p>"}

    masked_df, mask_meta = _mask_values_for_evaluation(df_gt, rng_cols, mask_fraction, seed=seed)

    # Run imputations on masked_df
    df_work = masked_df.copy()
    # identity and dates are not usually masked for evaluation; but their functions are safe.
    if identity_cols:
        df_work, _ = fill_missing_identity(df_work, identity_cols)
    if date_cols:
        df_work, _ = fill_missing_dates(df_work, date_cols, strategy="mode")
    if cont_eval:
        df_work, _ = fill_continuous_missing_rf(df_work, cont_eval, identity_cols, date_cols, min_training_ratio=0.0)
    if cat_eval:
        df_work, _ = fill_categorical_missing_rf(df_work, cat_eval, identity_cols, date_cols, min_training_ratio=0.0)

    # Metrics
    metrics = {}

    # Continuous metrics
    cont_metrics = []
    for col in cont_eval:
        if col not in mask_meta:
            continue
        rows = mask_meta[col]
        if len(rows) < min_samples:
            continue
        y_true = df_gt.loc[rows, col].astype(float).to_numpy()
        y_pred = df_work.loc[rows, col].astype(float).to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        cont_metrics.append({"column": col, "mae": mae, "rmse": rmse, "n": len(rows)})

    if cont_metrics:
        metrics["continuous"] = cont_metrics
        metrics["continuous_overall_mae"] = float(np.mean([m["mae"] for m in cont_metrics]))
        metrics["continuous_overall_rmse"] = float(np.mean([m["rmse"] for m in cont_metrics]))

    # Categorical metrics
    cat_metrics = []
    for col in cat_eval:
        if col not in mask_meta:
            continue
        rows = mask_meta[col]
        if len(rows) < min_samples:
            continue
        y_true = df_gt.loc[rows, col].astype(str).to_numpy()
        y_pred = df_work.loc[rows, col].astype(str).to_numpy()
        acc = float(np.mean(y_true == y_pred))
        cat_metrics.append({"column": col, "accuracy": acc, "n": len(rows)})

    if cat_metrics:
        metrics["categorical"] = cat_metrics
        metrics["categorical_overall_accuracy"] = float(np.mean([m["accuracy"] for m in cat_metrics]))

    # Render HTML block
    def _fmt_float(x):
        return f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else str(x)

    html = ""
    if cont_metrics:
        html += f"<p><b>Continuous imputation (synthetic)</b>: overall MAE={_fmt_float(metrics['continuous_overall_mae'])}, overall RMSE={_fmt_float(metrics['continuous_overall_rmse'])}</p>"
    else:
        html += "<p>No sufficient continuous samples for synthetic imputation evaluation.</p>"

    if cat_metrics:
        html += f"<p><b>Categorical imputation (synthetic)</b>: overall accuracy={_fmt_float(metrics['categorical_overall_accuracy'])}</p>"
    else:
        html += "<p>No sufficient categorical samples for synthetic imputation evaluation.</p>"

    # Also include per-column details (limited)
    if cont_metrics:
        html += "<h4>Continuous per-column</h4><ul>" + "".join(
            f"<li>{m['column']}: n={m['n']}, MAE={_fmt_float(m['mae'])}, RMSE={_fmt_float(m['rmse'])}</li>" for m in cont_metrics[:10]
        ) + "</ul>"
    if cat_metrics:
        html += "<h4>Categorical per-column</h4><ul>" + "".join(
            f"<li>{m['column']}: n={m['n']}, accuracy={_fmt_float(m['accuracy'])}</li>" for m in cat_metrics[:10]
        ) + "</ul>"

    return {**metrics, "html_block": html}


def evaluate_outlier_detection_synthetic(
    df_after_imputation: pd.DataFrame,
    identity_cols: list,
    date_cols: list,
    continuous_cols: list,
    categorical_cols: list,
    contamination: float = 0.05,
    outlier_fraction: float = 0.05,
    seed: int = 42,
    min_injected: int = 5,
) -> dict:
    """Synthetic outlier evaluation by injecting outliers into continuous columns.

    We create a copy of df_after_imputation (treated as ground truth),
    inject outliers by amplifying selected rows in continuous columns,
    then run Isolation Forest detection and compare detected rows with injected labels.

    Returns dict with precision/recall/F1 for row-level outlier detection."""
    rng = np.random.default_rng(seed)
    df_work = df_after_imputation.copy()

    cont_eval = [c for c in continuous_cols if c in df_work.columns]
    if not cont_eval:
        return {"html_block": "<p>No continuous columns available for synthetic outlier evaluation.</p>"}

    n_rows = len(df_work)
    if n_rows == 0:
        return {"html_block": "<p>Empty dataset; no outlier evaluation.</p>"}

    n_inject = max(min_injected, int(round(outlier_fraction * n_rows)))
    n_inject = min(n_inject, n_rows)
    injected_rows = rng.choice(np.arange(n_rows), size=n_inject, replace=False)
    injected_rows = injected_rows.tolist()
    injected_set = set(injected_rows)

    # Inject: for each selected continuous column, multiply by a large factor
    factor = 6.0
    for col in cont_eval:
        col_vals = df_work[col]
        # only modify non-null
        idx = [i for i in injected_rows if pd.notna(col_vals.iloc[i])]
        if not idx:
            continue
        df_work.loc[idx, col] = df_work.loc[idx, col] * factor

    detected_df, outlier_results = detect_outliers_isolation_forest(
        df_work, cont_eval, categorical_cols, contamination=contamination
    )

    if "overall_outlier" not in detected_df.columns:
        return {"html_block": "<p>Outlier detector did not produce overall_outlier flags.</p>"}

    detected_rows = set(detected_df.index[detected_df["overall_outlier"]].tolist())

    tp = len(detected_rows & injected_set)
    fp = len(detected_rows - injected_set)
    fn = len(injected_set - detected_rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    html = (
        f"<p><b>Outlier detection (synthetic)</b>: injected={len(injected_set)}, detected={len(detected_rows)}</p>"
        f"<p>Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}</p>"
    )
    return {
        "injected_outlier_rows": len(injected_set),
        "detected_outlier_rows": len(detected_rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "html_block": html,
    }


def data_quality_score(df: pd.DataFrame) -> dict:

    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 1
    completeness = 1 - (df.isna().sum().sum() / total_cells)
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0]) if df.shape[0] else 1
    overall = round(((completeness + uniqueness) / 2) * 100, 1)
    return {
        "completeness_pct": round(completeness * 100, 1),
        "uniqueness_pct": round(uniqueness * 100, 1),
        "overall_score": overall,
    }
