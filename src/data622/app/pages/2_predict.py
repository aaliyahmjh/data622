##########################################
# pages/2_outlook.py
# Outlook — Salary Range Predictor
##########################################

import numpy as np
import pandas as pd
import streamlit as st

from data622.app.config import (
    BOROUGHS,
    COL_AGENCY,
    COL_BOROUGH,
    COL_JOB_TITLE,
    COL_YEARS_SERVICE,
    MODEL_FEATURE_COLS,
)
from data622.app.loader import load_model, load_payroll_data

st.title("Salary Range Predictor")
st.caption("Enter job details below to get a projected salary range based on historical NYC payroll data.")

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
model = load_model()
df = load_payroll_data()

# ---------------------------------------------------------------------------
# Build input option lists from actual data so partners get accurate dropdowns
# ---------------------------------------------------------------------------
job_titles = sorted(df[COL_JOB_TITLE].dropna().unique().tolist())
agencies = sorted(df[COL_AGENCY].dropna().unique().tolist()) if COL_AGENCY in df.columns else []

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Job Details")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Job Title", options=job_titles)
        borough = st.selectbox("Borough", options=BOROUGHS)

    with col2:
        agency = st.selectbox("Agency", options=agencies) if agencies else st.text_input("Agency")
        years_of_service = st.slider(
            "Years of Service",
            min_value=0,
            max_value=40,
            value=5,
            step=1,
        )

    submitted = st.form_submit_button("Predict Salary Range", width="stretch")

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if submitted:
    st.divider()

    if model is None:
        # ── No model loaded — fall back to historical percentiles ────────
        st.info(
            "No trained model found. Showing historical salary range from the data instead. "
            "Drop a trained `.pkl` model into the `models/` directory to enable predictions."
        )

        hist_filter = df[df[COL_JOB_TITLE] == job_title]
        if COL_BOROUGH in df.columns:
            hist_filter = hist_filter[hist_filter[COL_BOROUGH] == borough]

        if hist_filter.empty:
            st.warning("No historical data found for this job title and borough combination.")
        else:
            from data622.app.config import COL_SALARY

            p10 = hist_filter[COL_SALARY].quantile(0.10)
            p25 = hist_filter[COL_SALARY].quantile(0.25)
            p50 = hist_filter[COL_SALARY].quantile(0.50)
            p75 = hist_filter[COL_SALARY].quantile(0.75)
            p90 = hist_filter[COL_SALARY].quantile(0.90)

            st.subheader(f"Historical Salary Range — {job_title}")
            st.caption(f"Based on {len(hist_filter):,} records in {borough}")

            m1, m2, m3 = st.columns(3)
            m1.metric("25th Percentile", f"${p25:,.0f}")
            m2.metric("Median (50th)", f"${p50:,.0f}")
            m3.metric("75th Percentile", f"${p75:,.0f}")

            with st.expander("Full distribution"):
                st.write(
                    pd.DataFrame(
                        {
                            "Percentile": ["10th", "25th", "50th", "75th", "90th"],
                            "Salary": [f"${p:,.0f}" for p in [p10, p25, p50, p75, p90]],
                        }
                    ).set_index("Percentile")
                )

    else:
        # ── Model loaded — run inference ─────────────────────────────────
        input_data = {
            COL_JOB_TITLE: [job_title],
            COL_BOROUGH: [borough],
            COL_YEARS_SERVICE: [years_of_service],
        }
        if agencies:
            input_data[COL_AGENCY] = [agency]

        input_df = pd.DataFrame(input_data)

        # Keep only the columns the model was trained on, in order
        model_cols = [c for c in MODEL_FEATURE_COLS if c in input_df.columns]
        input_df = input_df[model_cols]

        try:
            prediction = float(model.predict(input_df)[0])

            # Derive a ±range from historical std dev for the selected title
            hist_std = df[df[COL_JOB_TITLE] == job_title][
                __import__("data622.app.config", fromlist=["COL_SALARY"]).COL_SALARY
            ].std()
            margin = hist_std * 0.5 if not np.isnan(hist_std) else prediction * 0.10

            low = max(0.0, prediction - margin)
            high = prediction + margin

            st.subheader(f"Predicted Salary Range — {job_title}")
            st.caption(f"{borough} | {years_of_service} year(s) of service")

            m1, m2, m3 = st.columns(3)
            m1.metric("Low Estimate", f"${low:,.0f}")
            m2.metric("Point Estimate", f"${prediction:,.0f}")
            m3.metric("High Estimate", f"${high:,.0f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Input passed to model:", input_df)

st.divider()
st.caption(
    "Salary range is estimated using a ±0.5 standard deviation band around the model point estimate, "
    "derived from historical data for the selected job title."
)
