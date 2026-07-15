"""
report.py
Export helpers: cleaned dataset export (CSV/Excel) and a self-contained
HTML EDA/quality report the user can download and share.
"""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def export_dataframe(df: pd.DataFrame, file_type: str) -> bytes:
    if file_type == "csv":
        return df.to_csv(index=False).encode("utf-8")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    return output.getvalue()


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_html_report(
    project_name: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column_roles: dict,
    quality_before: dict,
    quality_after: dict,
    action_log: list,
    evaluation_results: dict | None = None,
) -> str:

    numeric_cols = column_roles.get("Continuous", [])
    categorical_cols = column_roles.get("Categorical", [])

    charts_html = ""

    for col in numeric_cols[:6]:
        fig, ax = plt.subplots(figsize=(5, 3))
        df_after[col].dropna().hist(bins=25, ax=ax, color="#2d7dd2")
        ax.set_title(f"Distribution: {col}")
        img = _fig_to_base64(fig)
        charts_html += f'<div class="chart"><img src="data:image/png;base64,{img}"/></div>'

    for col in categorical_cols[:4]:
        counts = df_after[col].value_counts().head(10)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        counts.plot(kind="bar", ax=ax, color="#0f4c81")
        ax.set_title(f"Top categories: {col}")
        plt.xticks(rotation=45, ha="right")
        img = _fig_to_base64(fig)
        charts_html += f'<div class="chart"><img src="data:image/png;base64,{img}"/></div>'

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        corr = df_after[numeric_cols].corr(numeric_only=True)
        cax = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        fig.colorbar(cax)
        ax.set_title("Correlation matrix")
        img = _fig_to_base64(fig)
        charts_html += f'<div class="chart"><img src="data:image/png;base64,{img}"/></div>'

    log_items = "".join(f"<li>{entry}</li>" for entry in action_log) or "<li>No changes logged.</li>"

    roles_html = "".join(
        f"<tr><td>{role}</td><td>{', '.join(cols) if cols else '—'}</td></tr>"
        for role, cols in column_roles.items() if role != "Data"
    )

    html = f"""
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>{project_name} — Data Quality Report</title>
    <style>
        body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
        h1 {{ color: #0f4c81; }}
        h2 {{ color: #2d7dd2; border-bottom: 2px solid #eaf0f7; padding-bottom: 4px; margin-top: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #e0e6ee; padding: 8px 12px; text-align: left; font-size: 14px; }}
        th {{ background: #f4f8fc; }}
        .metrics {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
        .metric-box {{ background: #f4f8fc; border-radius: 10px; padding: 1rem 1.5rem; min-width: 160px; }}
        .metric-box .value {{ font-size: 1.6rem; font-weight: 700; color: #0f4c81; }}
        .metric-box .label {{ font-size: 0.85rem; color: #555; }}
        .chart {{ display: inline-block; margin: 0.5rem; }}
        .chart img {{ max-width: 100%; border-radius: 8px; }}
        footer {{ margin-top: 3rem; font-size: 0.8rem; color: #888; }}
    </style>
    </head>
    <body>
        <h1>{project_name}</h1>
        <p>Data Quality &amp; EDA Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>Dataset Overview</h2>
        <div class="metrics">
            <div class="metric-box"><div class="value">{df_before.shape[0]:,}</div><div class="label">Rows (original)</div></div>
            <div class="metric-box"><div class="value">{df_after.shape[0]:,}</div><div class="label">Rows (cleaned)</div></div>
            <div class="metric-box"><div class="value">{df_after.shape[1]:,}</div><div class="label">Columns</div></div>
        </div>

        <h2>Data Quality Score</h2>
        <div class="metrics">
            <div class="metric-box"><div class="value">{quality_before['overall_score']}%</div><div class="label">Before cleaning</div></div>
            <div class="metric-box"><div class="value">{quality_after['overall_score']}%</div><div class="label">After cleaning</div></div>
            <div class="metric-box"><div class="value">{quality_after['completeness_pct']}%</div><div class="label">Completeness (after)</div></div>
            <div class="metric-box"><div class="value">{quality_after['uniqueness_pct']}%</div><div class="label">Uniqueness (after)</div></div>
        </div>

        <h2>Detected Column Roles</h2>
        <table><tr><th>Role</th><th>Columns</th></tr>{roles_html}</table>

        <h2>Cleaning Actions Log</h2>
        <ul>{log_items}</ul>

        <h2>Exploratory Data Analysis</h2>
        {charts_html if charts_html else '<p>No numeric or categorical columns available for charting.</p>'}

        <h2>1.3 General Objective</h2>
        <p>To develop a machine learning-based web application for automated data cleaning and exploratory data analysis using structured datasets.</p>

        <h2>1.4 Specific Objectives</h2>
        <ol>
            <li>Design a system for detecting data quality issues such as missing values, duplicates, and outliers using statistical and machine learning-based methods.</li>
            <li>Develop a machine learning-based imputation engine that learns inter-variable relationships within datasets to estimate and fill missing values.</li>
            <li>Implement an exploratory data analysis module for generating statistical summaries and visualizations from cleaned datasets.</li>
            <li>Evaluate performance in terms of imputation accuracy, outlier detection accuracy, data quality improvement, processing efficiency, and usability using appropriate evaluation metrics.</li>
        </ol>

        <h2>1.5 Research Questions</h2>
        <ol>
            <li>How can a system for detecting data quality issues such as missing values, duplicates, and outliers be designed using statistical and machine learning-based methods?</li>
            <li>How can a machine learning-based imputation engine that learns inter-variable relationships within datasets be developed for missing value estimation?</li>
            <li>How can an exploratory data analysis module be implemented to generate statistical insights and visualizations from cleaned datasets?</li>
            <li>How can the performance of the system be evaluated using imputation accuracy, outlier detection accuracy, data quality improvement, processing efficiency, and usability metrics?</li>
        </ol>

        <h2>Performance Evaluation (Full Evaluation — Synthetic)</h2>
        {'' if not evaluation_results else evaluation_results.get('html_block', '<p>No evaluation results available.</p>')}

        <footer>Generated by Smart Data Cleaner — Group 13, BDS III, EASTC</footer>

    </body>
    </html>
    """
    return html
