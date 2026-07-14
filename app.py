import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

import pipeline as pl
import report as rp

st.set_page_config(
    page_title="Smart Data Cleaner | Group 13",
    page_icon="🧹",
    layout="wide",
)

# -----------------------------------------------------------------
# Styling
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background: transparent;
    }
    .main {
        background: rgba(255,255,255,0.95);
        border-radius: 28px;
        padding: 2rem 2.5rem 2.5rem;
        box-shadow: 0 24px 80px rgba(15, 50, 96, 0.14);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.65);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        background: rgba(255,255,255,0.92);
        border-radius: 32px;
        box-shadow: 0 22px 66px rgba(15, 50, 96, 0.12);
    }
    .stMarkdown, .stText, .stExpander, .stDataFrame, .stCheckbox, .stRadio, .stSelectbox, .stSlider, .stButton, .stDownloadButton, .stFileUploader {
        background: rgba(255,255,255,0.96) !important;
        border-radius: 20px !important;
        padding: 0.9rem !important;
        box-shadow: 0 14px 34px rgba(15, 50, 96, 0.08) !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stText, .stExpanderHeader {
        color: #0b2447 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #0f4c81 0%, #2d7dd2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.2rem !important;
        box-shadow: 0 10px 24px rgba(15, 76, 129, 0.18) !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15, 76, 129, 0.28) !important;
    }
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {
        outline: 2px solid rgba(255,255,255,0.9) !important;
    }
    .hero {
        background: linear-gradient(135deg, rgba(0, 59, 114, 0.95), rgba(16, 126, 204, 0.85)),
                    url('https://images.unsplash.com/photo-1556155092-490a1ba16284?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
        margin-bottom: 1.8rem;
    }
    .hero h1 { font-size: 2.8rem; margin-bottom: 0.3rem; }
    .hero p { font-size: 1.1rem; opacity: 0.92; }
    .stage-row { display: flex; flex-wrap: wrap; gap: 0.65rem; margin-bottom: 1.25rem; }
    .stage-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334e68;
        background: rgba(227, 242, 255, 0.95);
        border: 1px solid rgba(46, 117, 182, 0.12);
    }
    .stage-pill.active { background: #1558d6; color: white; border-color: transparent; }
    .stage-pill.completed { background: #d6e8ff; color: #0f4c81; }
    .button-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem; }
    .button-row .stButton>button, .button-row .stDownloadButton>button {
        min-width: 180px;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #0f4c81 0%, #2d7dd2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.2rem !important;
        box-shadow: 0 10px 24px rgba(15, 76, 129, 0.18) !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15, 76, 129, 0.28) !important;
    }
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {
        outline: 2px solid rgba(255,255,255,0.9) !important;
    }
    .custom-info {
        background: rgba(15, 76, 129, 0.06);
        border-left: 4px solid #0f4c81;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        color: #0f3d6f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------
defaults = {
    "raw_df": None,
    "profile": None,
    "clean_df": None,
    "action_log": [],
    "duplicates_report": None,
    "outlier_report": None,
    "stage_reached": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_pipeline():
    for key, value in defaults.items():
        st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes, file_name):
    if file_bytes is None:
        return None
    name = file_name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes))
    st.error("Please upload a CSV or Excel file.")
    return None


def get_export_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = st.session_state.get("original_columns")
    if columns:
        return df.reindex(columns=columns)
    return df.copy()


# -----------------------------------------------------------------
# Main controls
# -----------------------------------------------------------------
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 0

project_name = st.text_input("Project / dataset name", value="Untitled Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None and (
    st.session_state.raw_df is None or uploaded_file.name != st.session_state.get("_last_file")
):
    df_loaded = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
    if df_loaded is not None:
        reset_pipeline()
        st.session_state.raw_df = df_loaded
        st.session_state.original_columns = df_loaded.columns.tolist()
        st.session_state["_last_file"] = uploaded_file.name
        st.session_state.current_stage = 0

steps = [
    "Upload & Profile",
    "Missing Values & Duplicates",
    "Standardize & Impute",
    "Outlier Detection",
    "EDA & Insights",
    "Export",
]
stage_idx = st.session_state.current_stage

stage_html = "<div class='stage-row'>"
for idx, label in enumerate(steps):
    css_class = "stage-pill"
    if idx == stage_idx:
        css_class += " active"
    elif idx < st.session_state.stage_reached:
        css_class += " completed"
    stage_html += f"<div class='{css_class}'>Step {idx + 1}: {label}</div>"
stage_html += "</div>"
st.markdown(stage_html, unsafe_allow_html=True)

st.markdown(
    "<div class='hero'>"
    "<h1>Smart Data Cleaner</h1>"
    "<p>Accelerated, guided data cleaning with beautiful visual styling — from upload through export.</p>"
    "</div>",
    unsafe_allow_html=True,
)

if st.button("🔄 Reset pipeline", use_container_width=False):
    reset_pipeline()
    st.session_state.current_stage = 0
    st.rerun()

st.markdown("---")

if st.session_state.raw_df is None:
    st.info("👋 Upload a CSV or Excel dataset above to begin the guided cleaning workflow.")
    st.stop()

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.markdown(
    f"""
    <div class='hero'>
        <h1 style='margin-bottom:0.2rem;'>Automated Data Cleaning &amp; EDA Platform</h1>
        <p style='margin-top:0.2rem; opacity:0.95;'>
        Machine-learning-based cleaning, imputation and exploratory analysis for structured datasets —
        <b>{project_name}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.raw_df is None:
    st.info("👋 Upload a CSV or Excel dataset above to begin the cleaning workflow.")
    st.stop()

raw_df = st.session_state.raw_df

# ===================================================================
# STEP 1 — Upload & Profile
# ===================================================================
if stage_idx == 0:
    st.subheader("Step 1 · Dataset Overview & Column Profiling")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", f"{raw_df.shape[1]:,}")
    c3.metric("Missing values", f"{raw_df.isna().sum().sum():,}")
    c4.metric("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}")

    st.markdown("#### Preview")
    st.dataframe(raw_df.head(100), use_container_width=True)

    st.markdown("#### Automatic column type detection")
    st.caption(
        "Columns are classified as Identity, Date, Continuous, or Categorical using pattern "
        "and statistical heuristics — this drives every downstream cleaning step."
    )

    if st.button("🔍 Run column profiling", type="primary"):
        with st.spinner("Profiling dataset..."):
            profile = pl.profile_dataset(raw_df)
        st.session_state.profile = profile
        st.session_state.stage_reached = max(st.session_state.stage_reached, 1)
        st.success("Profiling complete.")

    if st.session_state.profile:
        profile = st.session_state.profile
        colA, colB, colC, colD = st.columns(4)
        colA.markdown(f"**🪪 Identity ({len(profile['Identity'])})**\n\n" + "\n".join(f"- {c}" for c in profile["Identity"]) or "—")
        colB.markdown(f"**📅 Date ({len(profile['Date'])})**\n\n" + "\n".join(f"- {c}" for c in profile["Date"]) or "—")
        colC.markdown(f"**🔢 Continuous ({len(profile['Continuous'])})**\n\n" + "\n".join(f"- {c}" for c in profile["Continuous"]) or "—")
        colD.markdown(f"**🏷️ Categorical ({len(profile['Categorical'])})**\n\n" + "\n".join(f"- {c}" for c in profile["Categorical"]) or "—")

        st.markdown("Adjust detected roles if needed:")
        all_cols = list(raw_df.columns)
        with st.expander("✏️ Manually override column roles"):
            new_identity = st.multiselect("Identity columns", all_cols, default=profile["Identity"])
            new_date = st.multiselect("Date columns", all_cols, default=profile["Date"])
            new_continuous = st.multiselect("Continuous columns", all_cols, default=profile["Continuous"])
            new_categorical = st.multiselect("Categorical columns", all_cols, default=profile["Categorical"])
            if st.button("Apply overrides"):
                profile["Identity"] = new_identity
                profile["Date"] = new_date
                profile["Continuous"] = new_continuous
                profile["Categorical"] = new_categorical
                st.session_state.profile = profile
                st.success("Column roles updated.")

    if st.session_state.stage_reached >= 1:
        nav1, nav2 = st.columns([1, 1])
        if nav2.button("Continue to Step 2 →", key="next_stage_1"):
            st.session_state.current_stage = 1
            st.rerun()
    else:
        st.info("Complete profiling to unlock the next stage.")

# ===================================================================
# STEP 2 — Missing values & duplicates
# ===================================================================
elif stage_idx == 1:
    st.subheader("Step 2 · Missing Value Summary & Duplicate Review")

    if not st.session_state.profile:
        st.warning("Please complete Step 1 (column profiling) first.")
        st.stop()

    profile = st.session_state.profile
    working_df = profile["Data"]

    column_report, row_summary = pl.missing_value_report(working_df)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Missing values per column")
        st.dataframe(column_report, use_container_width=True, height=280)
    with col2:
        st.markdown("#### Rows by missing-value count")
        st.dataframe(row_summary, use_container_width=True, height=280)

    st.markdown("#### Duplicate detection")
    resolve_ids = st.checkbox(
        "Also resolve identity-column duplicates (keep the most complete row)", value=False
    )

    if st.button("🧹 Detect & remove duplicates", type="primary"):
        with st.spinner("Reviewing duplicates..."):
            deduped_df, dup_report = pl.review_duplicates(
                working_df, profile["Identity"], resolve_identity_duplicates=resolve_ids
            )
        st.session_state.clean_df = deduped_df
        st.session_state.duplicates_report = dup_report
        st.session_state.stage_reached = max(st.session_state.stage_reached, 2)
        st.success(f"Removed {dup_report['full_row_duplicates_removed']} full-row duplicates.")

    if st.session_state.duplicates_report:
        rep = st.session_state.duplicates_report
        st.markdown(f"- **Full-row duplicates removed:** {rep['full_row_duplicates_removed']}")
        if rep["identity_duplicates_found"]:
            st.markdown("- **Identity duplicates found:**")
            for col, vals in rep["identity_duplicates_found"].items():
                st.write(f"  · `{col}`: {vals[:10]}{' ...' if len(vals) > 10 else ''}")
        if rep["identity_duplicates_resolved"]:
            for col, removed in rep["identity_duplicates_resolved"].items():
                st.write(f"  · Resolved `{col}`: removed {removed} duplicate identity rows")

    if st.session_state.stage_reached >= 2:
        nav1, nav2 = st.columns([1, 1])
        if nav1.button("← Back to Step 1", key="back_stage_2"):
            st.session_state.current_stage = 0
            st.rerun()
        if nav2.button("Continue to Step 3 →", key="next_stage_2"):
            st.session_state.current_stage = 2
            st.rerun()
    else:
        st.info("Run duplicate detection to unlock the next stage.")

# ===================================================================
# STEP 3 — Standardization & ML imputation
# ===================================================================
elif stage_idx == 2:
    st.subheader("Step 3 · Data Standardization & ML-Based Imputation")

    if st.session_state.clean_df is None:
        st.warning("Please complete Step 2 (duplicate review) first.")
        st.stop()

    profile = st.session_state.profile
    df = st.session_state.clean_df

    st.markdown("#### Standardization options")
    c1, c2 = st.columns(2)
    with c1:
        clip_negative = st.checkbox(
            "Treat negative values in continuous columns as invalid (convert to NaN)", value=False
        )
    with c2:
        max_categories = st.slider("Max categories kept per categorical column", 5, 50, 20)

    st.markdown("#### Imputation options")
    c3, c4 = st.columns(2)
    with c3:
        min_training_ratio = st.slider(
            "Minimum training-data ratio required for ML imputation", 0.0, 0.95, 0.5, 0.05,
            help="Columns with less known data than this ratio fall back to median/mode imputation."
        )
    with c4:
        date_strategy = st.selectbox(
            "Missing date-fill strategy", ["unknown", "ffill", "bfill", "mode"], index=0
        )

    if st.button("⚙️ Run standardization + ML imputation pipeline", type="primary"):
        log = []
        with st.spinner("Standardizing data..."):
            df_std, std_log = pl.standardize_dataset(
                df, profile["Identity"], profile["Date"], profile["Continuous"], profile["Categorical"],
                clip_negative_continuous=clip_negative, max_categories=max_categories,
            )
        log += std_log

        with st.spinner("Imputing missing identity values..."):
            df_std, id_log = pl.fill_missing_identity(df_std, profile["Identity"])
        log += id_log

        with st.spinner("Imputing missing dates..."):
            df_std, date_log = pl.fill_missing_dates(df_std, profile["Date"], strategy=date_strategy)
        log += date_log

        with st.spinner("Running Random Forest imputation for continuous columns..."):
            df_std, cont_log = pl.fill_continuous_missing_rf(
                df_std, profile["Continuous"], profile["Identity"], profile["Date"], min_training_ratio
            )
        log += cont_log

        with st.spinner("Running Random Forest imputation for categorical columns..."):
            df_std, cat_log = pl.fill_categorical_missing_rf(
                df_std, profile["Categorical"], profile["Identity"], profile["Date"], min_training_ratio
            )
        log += cat_log

        st.session_state.clean_df = df_std
        st.session_state.action_log = log
        st.session_state.stage_reached = max(st.session_state.stage_reached, 3)
        st.success("Standardization and imputation complete.")

    if st.session_state.action_log:
        with st.expander("📋 View cleaning action log", expanded=True):
            for entry in st.session_state.action_log:
                st.write(f"- {entry}")

        st.markdown("#### Before vs. after")
        before_missing = int(raw_df.isna().sum().sum())
        after_missing = int(st.session_state.clean_df.isna().sum().sum())
        q_before = pl.data_quality_score(raw_df)
        q_after = pl.data_quality_score(st.session_state.clean_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing values before", f"{before_missing:,}")
        c2.metric("Missing values after", f"{after_missing:,}")
        c3.metric("Quality score before", f"{q_before['overall_score']}%")
        c4.metric("Quality score after", f"{q_after['overall_score']}%")

        st.dataframe(st.session_state.clean_df.head(100), use_container_width=True)

    if st.session_state.stage_reached >= 3:
        nav1, nav2 = st.columns([1, 1])
        if nav1.button("← Back to Step 2", key="back_stage_3"):
            st.session_state.current_stage = 1
            st.rerun()
        if nav2.button("Continue to Step 4 →", key="next_stage_3"):
            st.session_state.current_stage = 3
            st.rerun()
    else:
        st.info("Run standardization and imputation to unlock the next stage.")

# ===================================================================
# STEP 4 — Outlier detection
# ===================================================================
elif stage_idx == 3:
    st.subheader("Step 4 · Isolation Forest Outlier Detection")

    if st.session_state.stage_reached < 3:
        st.warning("Please complete Step 3 (standardization & imputation) first.")
        st.stop()

    profile = st.session_state.profile
    df = st.session_state.clean_df

    contamination = st.slider("Expected outlier proportion (contamination)", 0.01, 0.20, 0.05, 0.01)

    if st.button("🚨 Run outlier detection", type="primary"):
        with st.spinner("Scoring anomalies with Isolation Forest..."):
            flagged_df, outlier_results = pl.detect_outliers_isolation_forest(
                df, profile["Continuous"], profile["Categorical"], contamination=contamination
            )
        st.session_state.clean_df = flagged_df
        st.session_state.outlier_report = outlier_results
        st.session_state.stage_reached = max(st.session_state.stage_reached, 4)
        st.success("Outlier detection complete.")

    if st.session_state.outlier_report:
        rep = st.session_state.outlier_report
        c1, c2, c3 = st.columns(3)
        c1.metric("Continuous outliers", rep.get("continuous", 0))
        c2.metric("Categorical outliers", rep.get("categorical", 0))
        c3.metric("Overall flagged rows", rep.get("overall", 0))

        if "overall_outlier" in st.session_state.clean_df.columns:
            st.markdown("#### Flagged records")
            flagged = st.session_state.clean_df[st.session_state.clean_df["overall_outlier"]]
            st.dataframe(flagged.head(200), use_container_width=True)

            drop_outliers = st.checkbox("Remove flagged outlier rows from the cleaned dataset")
            if drop_outliers and st.button("Apply outlier removal"):
                st.session_state.clean_df = st.session_state.clean_df[
                    ~st.session_state.clean_df["overall_outlier"]
                ].reset_index(drop=True)
                st.success("Outlier rows removed.")

    if st.session_state.stage_reached >= 4:
        nav1, nav2 = st.columns([1, 1])
        if nav1.button("← Back to Step 3", key="back_stage_4"):
            st.session_state.current_stage = 2
            st.rerun()
        if nav2.button("Continue to Step 5 →", key="next_stage_4"):
            st.session_state.current_stage = 4
            st.rerun()
    else:
        if st.button("Skip outlier detection and continue", key="skip_outlier_stage"):
            st.session_state.current_stage = 4
            st.session_state.stage_reached = max(st.session_state.stage_reached, 4)
            st.rerun()
        st.info("Run outlier detection or skip it to continue to EDA.")

# ===================================================================
# STEP 5 — EDA
# ===================================================================
elif stage_idx == 4:
    st.subheader("Step 5 · Exploratory Data Analysis")

    if st.session_state.stage_reached < 3:
        st.warning("Please complete Step 3 (standardization & imputation) first.")
        st.stop()

    profile = st.session_state.profile
    df = st.session_state.clean_df

    st.markdown("#### Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    numeric_cols = profile["Continuous"]
    categorical_cols = profile["Categorical"]

    tab1, tab2, tab3 = st.tabs(["Numeric distributions", "Categorical frequencies", "Correlation"])

    with tab1:
        if numeric_cols:
            selected_num = st.selectbox("Choose a numeric column", numeric_cols)
            colh, colb = st.columns(2)
            with colh:
                st.markdown("**Histogram**")
                st.bar_chart(df[selected_num].dropna().value_counts(bins=20).sort_index())
            with colb:
                st.markdown("**Summary statistics**")
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.write("No continuous columns detected.")

    with tab2:
        if categorical_cols:
            selected_cat = st.selectbox("Choose a categorical column", categorical_cols)
            counts = df[selected_cat].value_counts().head(15)
            st.bar_chart(counts)
        else:
            st.write("No categorical columns detected.")

    with tab3:
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr(numeric_only=True)
            st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
        else:
            st.write("Not enough numeric columns for a correlation matrix.")

    st.session_state.stage_reached = max(st.session_state.stage_reached, 5)
    nav1, nav2 = st.columns([1, 1])
    if nav1.button("← Back to Step 4", key="back_stage_5"):
        st.session_state.current_stage = 3
        st.rerun()
    if nav2.button("Continue to Step 6 →", key="next_stage_5"):
        st.session_state.current_stage = 5
        st.rerun()


elif stage_idx == 5:
    st.subheader("Step 6 · Export Cleaned Data & Report")

    if st.session_state.stage_reached < 3:
        st.warning("Please complete Step 3 (standardization & imputation) first.")
        st.stop()

    profile = st.session_state.profile
    df = st.session_state.clean_df

    q_before = pl.data_quality_score(raw_df)
    q_after = pl.data_quality_score(df)

    c1, c2 = st.columns(2)
    c1.metric("Quality score before", f"{q_before['overall_score']}%")
    c2.metric("Quality score after", f"{q_after['overall_score']}%")

    st.markdown("#### Export cleaned dataset")
    export_cols = st.columns(2)
    export_df = get_export_df(df)
    export_df = get_export_df(df)
    with export_cols[0]:
        st.download_button(
            "⬇️ Download as CSV",
            data=rp.export_dataframe(export_df, "csv"),
            file_name=f"{project_name.replace(' ', '_')}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "⬇️ Download as Excel",
            data=rp.export_dataframe(export_df, "excel"),
            file_name=f"{project_name.replace(' ', '_')}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("#### Export data quality & EDA report")
    if st.button("📄 Generate HTML report", type="primary"):
        with st.spinner("Building report..."):
            html_report = rp.build_html_report(
                project_name, raw_df, df, profile, q_before, q_after, st.session_state.action_log
            )
        st.session_state["_html_report"] = html_report
        st.success("Report generated.")

    if st.button("← Back to Step 5", key="back_stage_6"):
        st.session_state.current_stage = 4
        st.rerun()

    if st.session_state.get("_html_report"):
        st.download_button(
            "⬇️ Download report (HTML)",
            data=st.session_state["_html_report"],
            file_name=f"{project_name.replace(' ', '_')}_report.html",
            mime="text/html",
            use_container_width=True,
        )

    st.balloons()
