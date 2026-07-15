# TODO — Smart Data Cleaner (Group 13)

- [x] Identify the actual runtime exception (not just Streamlit warnings) from the platform logs.
- [x] Apply the minimal code change(s) in `app.py`/other modules to resolve the exception.
- [ ] Remove deprecated `use_container_width` usages (optional, warnings only) by replacing with `width=`.
- [x] Run `python -m py_compile` / basic import checks locally.
- [ ] Run `streamlit run app.py` locally to verify the app loads.

## Full-evaluation requirement (Objectives/RQs + metrics in final HTML report)
- [ ] Add synthetic imputation evaluation helpers to `pipeline.py`.
- [ ] Add synthetic outlier evaluation helpers to `pipeline.py`.
- [ ] Wire evaluation execution in `app.py` after Step 3/Step 4 and pass results into `report.py`.
- [ ] Extend `report.py` HTML with:
  - 1.3 General Objective
  - 1.4 Specific Objectives
  - 1.5 Research Questions i–iv
  - evaluation metrics section (computed)
- [ ] Verify `report.py` can render without crashing for datasets with few columns.

