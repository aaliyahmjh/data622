##########################################
# app.py - NYC Payroll Salary Predictor
# Run with: uv run streamlit run src/data622/app/app.py
##########################################

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data622.app.config import (
    APP_DISCLAIMER,
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    COL_AGENCY,
    COL_JOB_TITLE,
    COL_SALARY,
    YEAR_MAX,
)
from data622.app.loader import (
    load_model,
    load_reference_table,
    load_title_category_map,
    load_yoy_summary,
)
from data622.dataset import clean_text
from data622.features import bucket_tenure

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load resources (cached)
# ---------------------------------------------------------------------------
predictor = load_model()
reference_df = load_reference_table()
title_category_map = load_title_category_map()
yoy_df = load_yoy_summary()

# ---------------------------------------------------------------------------
# Build lookup structures from reference table
# ---------------------------------------------------------------------------
if reference_df is not None:
    all_titles: list[str] = sorted(
        t.title() for t in reference_df[COL_JOB_TITLE].dropna().unique()
    )
    # title-cased title -> list of title-cased agencies (display values; clean_text lowercases for lookup)
    title_to_agencies: dict[str, list[str]] = (
        reference_df.groupby(COL_JOB_TITLE)[COL_AGENCY]
        .apply(lambda s: sorted(a.title() for a in s.dropna().unique()))
        .to_dict()
    )
    # remap keys to title-case to match selectbox return values
    title_to_agencies = {k.title(): v for k, v in title_to_agencies.items()}
    # precompute title_frequency and agency_size — keyed by lowercase (matches clean_text output)
    title_frequency: dict[str, int] = (
        reference_df.groupby(COL_JOB_TITLE)["agency_title_count"].sum().to_dict()
    )
    agency_size: dict[str, int] = (
        reference_df.groupby(COL_AGENCY)["agency_title_count"].sum().to_dict()
    )
else:
    all_titles = []
    title_to_agencies = {}
    title_frequency = {}
    agency_size = {}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)


if reference_df is None:
    st.error(
        "Reference table not found at `data/processed/reference_table.csv`. "
        "Run `uv run python -m data622.train` to generate it."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input form — sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Enter Your Job Details")

    job_title = st.selectbox(
        "Job Title",
        options=all_titles,
        index=None,
        placeholder="Select job title…",
    )

    agency_options = title_to_agencies.get(job_title, []) if job_title else []
    agency = st.selectbox(
        "Agency",
        options=agency_options,
        index=None,
        placeholder="Select agency…",
        disabled=not bool(job_title),
    )

    tenure = st.number_input(
        "Years in Current Role",
        min_value=0,
        max_value=40,
        value=5,
        step=1,
    )

    predict_clicked = st.button(
        "Predict Salary", type="primary", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Prediction + output
# ---------------------------------------------------------------------------
if predict_clicked:
    if not job_title or not agency:
        st.warning("Please select both a Job Title and an Agency before predicting.")
        st.stop()

    # Standardize inputs to match reference table keys
    title_std = clean_text(pd.Series([job_title]))[0]
    agency_std = clean_text(pd.Series([agency]))[0]

    # Reference table lookup
    ref_row = reference_df[
        (reference_df[COL_JOB_TITLE] == title_std)
        & (reference_df[COL_AGENCY] == agency_std)
    ]

    if ref_row.empty:
        st.warning(
            f"No reference data found for **{job_title}** at **{agency}**. "
            "Try a different title/agency combination."
        )
        st.stop()

    ref = ref_row.iloc[0]

    # Build model input row
    tenure_df = pd.DataFrame({"tenure_years": [int(tenure)]})
    tenure_df = bucket_tenure(tenure_df)
    tenure_bucket_val = str(tenure_df["tenure_bucket"].iloc[0])

    title_freq = int(title_frequency.get(title_std, ref.get("agency_title_count", 1)))
    agency_sz = int(agency_size.get(agency_std, 1))
    title_cat = title_category_map.get(title_std, "other")
    agency_title_cnt = int(ref.get("agency_title_count", 1))
    title_grouped = title_std if agency_title_cnt >= 100 else "other_title"

    input_row = {
        "agency_std": agency_std,
        "title_std": title_std,
        "title_category": title_cat,
        "title_std_grouped": title_grouped,
        "tenure_bucket": tenure_bucket_val,
        "fiscal_year": YEAR_MAX,
        "tenure_years": int(tenure),
        "title_frequency": title_freq,
        "agency_size": agency_sz,
        "title_avg_salary": float(ref["median_salary_by_title"]),
        "median_salary_by_title": float(ref["median_salary_by_title"]),
        "median_salary_by_agency": float(ref["median_salary_by_agency"]),
        "count_of_job_titles": float(ref.get("count_of_job_titles", 1)),
        "current_year": float(ref.get("current_year", 2022)),
        "regular_hours": float(ref.get("regular_hours", 1820)),
    }
    input_df = pd.DataFrame([input_row])

    st.divider()

    tab1, tab2 = st.tabs(["Role & Trajectory", "Salary Distribution"])

    with tab1:
        # ── A. Role Summary with inline YoY deltas ───────────────────────────
        st.subheader(f"Role Summary — {job_title.title()}")

        # Compute YoY deltas to show as metric deltas
        title_growth: float | None = None
        agency_growth: float | None = None
        yoy_label = ""
        if yoy_df is not None:
            latest_year = int(yoy_df["fiscal_year"].max())
            prev_year = latest_year - 1
            yoy_label = f"{prev_year} → {latest_year}"

            def _get_yoy(
                df: pd.DataFrame, filter_col: str, filter_val: str, metric_col: str
            ) -> float | None:
                mask = df[filter_col] == filter_val
                latest_val = df[(df["fiscal_year"] == latest_year) & mask][metric_col]
                prev_val = df[(df["fiscal_year"] == prev_year) & mask][metric_col]
                if latest_val.empty or prev_val.empty:
                    return None
                prev_med = (
                    prev_val.iloc[0]
                    if metric_col in ("headcount", "regular_hours")
                    else prev_val.median()
                )
                latest_med = (
                    latest_val.iloc[0]
                    if metric_col in ("headcount", "regular_hours")
                    else latest_val.median()
                )
                if prev_med == 0:
                    return None
                return float((latest_med - prev_med) / prev_med * 100)

            agency_yoy_df = yoy_df[yoy_df[COL_JOB_TITLE] == title_std]
            title_growth = _get_yoy(yoy_df, COL_JOB_TITLE, title_std, "base_salary")
            agency_growth = _get_yoy(agency_yoy_df, COL_AGENCY, agency_std, "base_salary")
            headcount_growth = _get_yoy(agency_yoy_df, COL_AGENCY, agency_std, "headcount")
            hours_growth = _get_yoy(agency_yoy_df, COL_AGENCY, agency_std, "regular_hours")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "Median Salary (Title)",
            f"${float(ref['median_salary_by_title']):,.0f}",
            delta=f"{title_growth:+.1f}% ({yoy_label})"
            if title_growth is not None
            else None,
        )
        s2.metric(
            "Median Salary (Agency)",
            f"${float(ref['median_salary_by_agency']):,.0f}",
            delta=f"{agency_growth:+.1f}% ({yoy_label})"
            if agency_growth is not None
            else None,
        )
        s3.metric(
            "Employees in This Role",
            f"{int(ref['agency_title_count']):,}",
            delta=f"{headcount_growth:+.1f}% ({yoy_label})"
            if headcount_growth is not None
            else None,
        )
        s4.metric(
            "Typical Hours / Year",
            f"{int(ref.get('regular_hours', 1820)):,}",
            delta=f"{hours_growth:+.1f}% ({yoy_label})"
            if hours_growth is not None
            else None,
        )

        # ── B. Prediction ────────────────────────────────────────────────────
        if predictor is None:
            st.info("No trained model found. Showing reference-based estimate only.")
            predicted = float(ref["median_salary_by_title"])
        else:
            try:
                salary_preds, _ = predictor.predict(input_df)
                raw_prediction = float(salary_preds[0])
                hist_median = float(ref["median_salary_by_title"])
                if predictor.feature_stats is not None and title_std in predictor.feature_stats.get("title_avg_salary", {}):
                    hist_median = float(predictor.feature_stats["title_avg_salary"][title_std])
                predicted = 0.70 * raw_prediction + 0.30 * hist_median
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        median_title = float(ref["median_salary_by_title"])

        st.subheader("Salary Trajectory")
        st.caption(f"{agency.title()} | {tenure} year(s) in role")

        st.metric("Predicted Salary (2024)", f"${predicted:,.0f}")
        st.caption(
            ":information_source: The model was trained on years of service as a proxy for tenure. "
            "In the underlying data, longer tenure correlates with lower salaries in some roles — "
            "this is a known model limitation, not a prediction error."
        )

        # Build historical series for this title/agency from yoy_summary
        hist_rows = []
        hist_std = 0.0
        if yoy_df is not None:
            hist_slice = yoy_df[
                (yoy_df[COL_JOB_TITLE] == title_std) & (yoy_df[COL_AGENCY] == agency_std)
            ].sort_values("fiscal_year")
            hist_rows = (
                hist_slice[["fiscal_year", COL_SALARY]]
                .rename(columns={COL_SALARY: "salary"})
                .assign(kind="Historical Median")
                .to_dict("records")
            )
            hist_std = (
                float(hist_slice[COL_SALARY].std())
                if len(hist_slice) > 1
                else predicted * 0.10
            )

        # Compute average YoY growth rate from historical data
        if len(hist_rows) >= 2:
            salaries = [r["salary"] for r in hist_rows]
            years = [r["fiscal_year"] for r in hist_rows]
            n_periods = years[-1] - years[0]
            avg_growth = (
                (salaries[-1] / salaries[0]) ** (1 / n_periods) - 1
                if n_periods > 0
                else 0.02
            )
        else:
            avg_growth = 0.02
            hist_std = predicted * 0.10

        # Project 5 years forward; confidence band widens with time (uncertainty compounds)
        projection_rows = []
        band_rows = []
        base_year = YEAR_MAX
        base_salary = predicted
        for i in range(1, 6):
            proj_sal = base_salary * ((1 + avg_growth) ** i)
            spread = hist_std * (i**0.5)
            projection_rows.append(
                {"fiscal_year": base_year + i, "salary": proj_sal, "kind": "Projected"}
            )
            band_rows.append(
                {
                    "fiscal_year": base_year + i,
                    "upper": proj_sal + spread,
                    "lower": proj_sal - spread,
                }
            )

        anchor = [{"fiscal_year": YEAR_MAX, "salary": predicted, "kind": "Projected"}]
        anchor_band = [{"fiscal_year": YEAR_MAX, "upper": predicted, "lower": predicted}]

        chart_data = pd.DataFrame(hist_rows + anchor + projection_rows)
        band_data = pd.DataFrame(anchor_band + band_rows)

        overall_median_rows = []
        if yoy_df is not None:
            overall_median_rows = (
                yoy_df[yoy_df[COL_JOB_TITLE] == title_std]
                .groupby("fiscal_year")["base_salary"]
                .median()
                .reset_index()
                .rename(columns={"base_salary": "salary"})
                .assign(kind="Overall Title Median")
                .to_dict("records")
            )

        overall_df = pd.DataFrame(overall_median_rows)

        if not chart_data.empty:
            all_salaries = (
                list(chart_data["salary"])
                + list(band_data["lower"])
                + list(band_data["upper"])
            )
            if not overall_df.empty:
                all_salaries += list(overall_df["salary"])
            y_min = min(all_salaries) * 0.90
            y_max = max(all_salaries) * 1.10
            y_scale = alt.Scale(domain=[y_min, y_max], zero=False)
            y_enc = alt.Y(
                "salary:Q",
                title="Median Base Salary ($)",
                scale=y_scale,
                axis=alt.Axis(format="$,.0f"),
            )
            x_enc = alt.X("fiscal_year:O", title="Fiscal Year")

            hist_df = chart_data[chart_data["kind"] == "Historical Median"]
            proj_df = chart_data[chart_data["kind"] == "Projected"]

            overall_line = (
                alt.Chart(overall_df)
                .mark_line(color="#666688", strokeWidth=1.5, strokeDash=[2, 2], opacity=0.7)
                .encode(
                    x=x_enc,
                    y=alt.Y("salary:Q", scale=y_scale),
                    tooltip=[
                        alt.Tooltip("fiscal_year:O", title="Year"),
                        alt.Tooltip(
                            "salary:Q", title="Overall Title Median", format="$,.0f"
                        ),
                    ],
                )
            )

            hist_line = (
                alt.Chart(hist_df)
                .mark_line(point=True, color="#80cbc4", strokeWidth=2)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    tooltip=[
                        alt.Tooltip("fiscal_year:O", title="Year"),
                        alt.Tooltip("salary:Q", title="This Agency Median", format="$,.0f"),
                    ],
                )
            )

            ci_band = (
                alt.Chart(band_data)
                .mark_area(color="#e8a830", opacity=0.15)
                .encode(
                    x=alt.X("fiscal_year:O"),
                    y=alt.Y("upper:Q", scale=y_scale),
                    y2="lower:Q",
                )
            )

            proj_line = (
                alt.Chart(proj_df)
                .mark_line(point=True, color="#e8a830", strokeWidth=2, strokeDash=[5, 3])
                .encode(
                    x=x_enc,
                    y=alt.Y("salary:Q", scale=y_scale),
                    tooltip=[
                        alt.Tooltip("fiscal_year:O", title="Year"),
                        alt.Tooltip("salary:Q", title="Projected Salary", format="$,.0f"),
                    ],
                )
            )

            layers = (
                [overall_line, hist_line, ci_band, proj_line]
                if not overall_df.empty
                else [hist_line, ci_band, proj_line]
            )

            trajectory = (
                alt.layer(*layers).properties(height=320).configure_view(strokeOpacity=0)
            )
            st.altair_chart(trajectory, use_container_width=True)
            st.caption(
                f"Teal: this agency's median. "
                f"Gray dashed: overall median across all agencies for this title. "
                f"Amber: projected at {avg_growth * 100:.1f}% avg annual growth with uncertainty band."
            )

    with tab2:
        st.subheader("Salary Distribution — All Job Titles")
        st.caption(
            "Median salary per title/agency combination across all NYC payroll roles (4,108 data points)."
        )

        selected_salary = float(ref["median_salary_by_title"])
        hist_source = reference_df[["median_salary_by_title"]].copy()

        bars = (
            alt.Chart(hist_source)
            .mark_bar(color="#80cbc4", opacity=0.75)
            .encode(
                x=alt.X(
                    "median_salary_by_title:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Median Salary ($)",
                    axis=alt.Axis(format="$,.0f"),
                ),
                y=alt.Y("count()", title="Number of Roles"),
                tooltip=[
                    alt.Tooltip(
                        "median_salary_by_title:Q",
                        bin=True,
                        title="Salary Range",
                        format="$,.0f",
                    ),
                    alt.Tooltip("count()", title="Roles in Range"),
                ],
            )
        )

        rule = (
            alt.Chart(pd.DataFrame({"salary": [selected_salary]}))
            .mark_rule(color="#e8557a", strokeWidth=2.5)
            .encode(
                x=alt.X("salary:Q"),
                tooltip=[
                    alt.Tooltip(
                        "salary:Q",
                        title=f"{job_title.title()} Median",
                        format="$,.0f",
                    )
                ],
            )
        )

        dist_chart = (
            alt.layer(bars, rule)
            .properties(height=320)
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(dist_chart, use_container_width=True)
        st.caption(
            f"Pink line: {job_title.title()} median salary (${selected_salary:,.0f})."
        )


st.caption(APP_DISCLAIMER)
