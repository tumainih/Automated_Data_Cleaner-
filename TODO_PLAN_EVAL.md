# Plan: Add Objectives + Research Questions + Full Evaluation (synthetic)

## Files to modify
- `pipeline.py`: add synthetic evaluation helpers
- `app.py`: run evaluations after imputation and outlier detection; pass results to report
- `report.py`: render Objectives (1.3/1.4/1.5) and evaluation metrics in HTML

## Implementation details
1. **Synthetic imputation evaluation**
   - Select continuous columns from `profile['Continuous']` and categorical from `profile['Categorical']`.
   - Create a copy of the cleaned (post-step-3) dataframe.
   - For each selected column, randomly mask a fraction (e.g. 10%) of *non-null* values.
   - Re-run imputation logic on the masked dataframe (we can call existing functions).
   - Metrics:
     - Continuous: MAE and RMSE against the original values at masked positions.
     - Categorical: accuracy (and optionally macro-F1) against original values at masked positions.

2. **Synthetic outlier detection evaluation**
   - Inject synthetic outliers into a numeric subset of continuous columns (e.g. add large noise or multiply by factor).
   - Re-run `detect_outliers_isolation_forest` on the injected dataset.
   - Metrics:
     - precision/recall/F1 against injected outlier rows (row-level labels).

3. **UI wiring**
   - In `app.py`, after Step 3 (imputation) compute imputation evaluation results.
   - After Step 4 (outliers) compute outlier evaluation results.
   - Pass both result dicts into `rp.build_html_report()`.

4. **HTML rendering**
   - Update `report.py`:
     - Add sections for 1.3 General Objective, 1.4 Specific Objectives, 1.5 Research Questions i–iv.
     - Add evaluation metrics section with computed values.
     - Include an explicit mapping: which pipeline functions answer each objective/RQ.

