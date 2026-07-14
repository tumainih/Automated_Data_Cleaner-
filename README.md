# Smart Data Cleaner — Group 13 (BDS III)

An automated, ML-powered data cleaning and EDA web application built with Streamlit,
implementing all functional requirements from Chapter 4 (System Development Life Cycle)
of the Group 13 project report.

## Files
- `app.py` — Streamlit UI: a 6-step guided pipeline (Upload & Profile → Missing Values &
  Duplicates → Standardize & Impute → Outlier Detection → EDA → Export).
- `pipeline.py` — Core engine: column profiling, missing-value reporting, duplicate
  resolution, standardization, Random Forest–based imputation (continuous & categorical),
  Isolation Forest outlier detection, and data quality scoring. Refactored from the
  original interactive scripts (Data_import.py, Data_consistency.py, Duplicates.py,
  FillContinous.py, FillCategorical.py, FillDate.py, FillMissingToid.py,
  OutlierDetection.py, Missing_summary.py, Sorting_missing.py) into pure functions with
  no `input()`/`print()` calls, so they run safely inside Streamlit.
- `report.py` — Export helpers: CSV/Excel export of the cleaned dataset, and a
  self-contained downloadable HTML data-quality & EDA report with embedded charts.
- `requirements.txt` — Python dependencies.

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes on design decisions
- All the original scripts used blocking `input()` prompts (e.g. asking whether to keep
  negative values, or to manually type in missing dates). These have been replaced with
  Streamlit widgets (checkboxes, sliders, selectboxes) so the app never blocks and behaves
  predictably for business users.
- Column-role detection (Identity / Date / Continuous / Categorical) can be manually
  overridden in Step 1 if the automatic heuristics misclassify a column.
- The Random Forest imputation step reuses the original feature-importance-based
  predictor selection logic (top 20–30% of features depending on dataset size).
- A single data-quality score (based on completeness + uniqueness) is tracked before and
  after cleaning so the improvement is visible at a glance and included in the exported
  report.
