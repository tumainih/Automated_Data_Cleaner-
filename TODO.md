# TODO — Fix Streamlit app runtime error

- [x] Identify the actual runtime exception (not just Streamlit warnings) from the platform logs.
- [x] Apply the minimal code change(s) in `app.py`/other modules to resolve the exception.
- [ ] Remove deprecated `use_container_width` usages (optional, warnings only) by replacing with `width=`.
- [x] Run `python -m py_compile` / basic import checks locally.
- [ ] Run `streamlit run app.py` locally to verify the app loads.


