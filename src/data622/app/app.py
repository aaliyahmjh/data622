# Entry point — Job Discovery
# Run with: uv run streamlit run src/data622/app/app.py

import altair as alt
import pydeck as pdk
import streamlit as st

from data622.app.config import (
    APP_DISCLAIMER,
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    BOROUGH_COLORS_HEX,
    BOROUGH_COLORS_RGBA,
    BOROUGHS,
    COL_BOROUGH,
    COL_FISCAL_YEAR,
    COL_JOB_TITLE,
    COL_SALARY,
    YEAR_MAX,
    YEAR_MIN,
)
from data622.app.loader import load_data_dictionary, load_payroll_data

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    page_name="Historical",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Override selected tab color to be readable against the dark theme
st.markdown(
    """
    <style>
    .stTabs [aria-selected="true"] {
        color: #80cbc4 !important;
        border-bottom-color: #80cbc4 !important;
    }
    .stDataFrame thead tr th {
        background-color: #2d3748 !important;
        color: #f0f0f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.markdown("### Research real job data by title, salary, or career path")
with col2:
    st.markdown("### Explore jobs and salaries specific to your borough")
with col3:
    st.markdown("### Project your career path and predicted salary range")

st.divider()

st.subheader("Job Discovery")
st.markdown(
    "Explore salaries for jobs with the City of New York. Use the sidebar to change jobs.\n\n Go to [Salary Predictor](./2_predict) for projections."
)

df = load_payroll_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    all_titles = sorted(df[COL_JOB_TITLE].dropna().unique().tolist())
    selected_title = st.selectbox("Job Title", options=all_titles, index=0)

    selected_boroughs = st.multiselect(
        "Borough",
        options=BOROUGHS,
        default=BOROUGHS,
    )

    year_range = st.slider(
        "Fiscal Year",
        min_value=YEAR_MIN,
        max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX),
    )

# Apply filters
mask = (
    (df[COL_JOB_TITLE] == selected_title)
    & (df[COL_BOROUGH].isin(selected_boroughs))
    & (df[COL_FISCAL_YEAR].between(*year_range))
)
filtered = df[mask].copy()

# Tabs
charts_tab, dict_tab = st.tabs(["Charts", "Data Dictionary"])

# Chart tab
with charts_tab:
    if filtered.empty:
        st.warning("No data matches the current filters.")
    else:
        # Salary Distribution
        st.subheader("Salary Distribution")

        hist = (
            alt.Chart(filtered)
            .mark_bar(opacity=0.8)
            .encode(
                alt.X(
                    f"{COL_SALARY}:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Base Salary ($)",
                    axis=alt.Axis(format="$,.0f"),
                ),
                alt.Y("count()", title="Number of Employees"),
                alt.Color(
                    f"{COL_BOROUGH}:N",
                    title="Borough",
                    scale=alt.Scale(
                        domain=list(BOROUGH_COLORS_HEX.keys()),
                        range=list(BOROUGH_COLORS_HEX.values()),
                    ),
                ),
                tooltip=[
                    alt.Tooltip(f"{COL_SALARY}:Q", title="Salary Bin", format="$,.0f"),
                    alt.Tooltip("count()", title="Count"),
                    alt.Tooltip(f"{COL_BOROUGH}:N", title="Borough"),
                ],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(hist, width="stretch")

        st.divider()

        # Salary Trend line chart
        st.subheader("Median Salary Trend by Year")

        trend_df = (
            filtered.groupby(COL_FISCAL_YEAR)[COL_SALARY]
            .median()
            .reset_index()
            .rename(columns={COL_SALARY: "Median Salary"})
        )

        trend = (
            alt.Chart(trend_df)
            .mark_line(point=True, color="#ff7f0e")
            .encode(
                alt.X(f"{COL_FISCAL_YEAR}:O", title="Fiscal Year"),
                alt.Y(
                    "Median Salary:Q",
                    title="Median Base Salary ($)",
                    axis=alt.Axis(format="$,.0f"),
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(f"{COL_FISCAL_YEAR}:O", title="Year"),
                    alt.Tooltip(
                        "Median Salary:Q", title="Median Salary", format="$,.0f"
                    ),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(trend, width="stretch")

        st.divider()

        # Choropleth map of boroughs
        st.subheader("Median Salary by Borough")

        BOROUGH_COORDS: dict[str, tuple[float, float]] = {
            "MANHATTAN": (40.7831, -73.9712),
            "BROOKLYN": (40.6782, -73.9442),
            "QUEENS": (40.7282, -73.7949),
            "BRONX": (40.8448, -73.8648),
            "STATEN ISLAND": (40.5795, -74.1502),
        }

        borough_agg = (
            filtered.groupby(COL_BOROUGH)[COL_SALARY]
            .median()
            .reset_index()
            .rename(columns={COL_SALARY: "median_salary"})
        )
        borough_agg["lat"] = borough_agg[COL_BOROUGH].map(
            lambda x: BOROUGH_COORDS.get(x, (40.7128, -74.0060))[0]
        )
        borough_agg["lon"] = borough_agg[COL_BOROUGH].map(
            lambda x: BOROUGH_COORDS.get(x, (40.7128, -74.0060))[1]
        )
        borough_agg["color"] = borough_agg[COL_BOROUGH].map(
            lambda x: BOROUGH_COLORS_RGBA.get(x, [128, 128, 128, 200])
        )

        sal_min = borough_agg["median_salary"].min()
        sal_max = borough_agg["median_salary"].max()
        scale_range = sal_max - sal_min if sal_max != sal_min else 1
        borough_agg["radius"] = (
            1000 + 7200 * (borough_agg["median_salary"] - sal_min) / scale_range
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=borough_agg,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=9,
            pitch=0,
        )

        tooltip = {
            "html": "<b>{"
            + COL_BOROUGH
            + "}</b><br/>Median Salary: ${median_salary:,.0f}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,  # type: ignore[arg-type]
                map_style="light",
            )
        )

        display_cols = {COL_BOROUGH: "Borough", "median_salary": "Median Salary ($)"}
        summary_tbl = borough_agg[[COL_BOROUGH, "median_salary"]].rename(
            columns=display_cols
        )
        summary_tbl["Median Salary ($)"] = summary_tbl["Median Salary ($)"].map(
            "${:,.0f}".format
        )
        st.dataframe(
            summary_tbl.style.set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#2d3748"),
                            ("color", "#f0f0f0"),
                        ],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )


# Data Dictionary tab
with dict_tab:
    st.subheader("Data Dictionary")

    data_dict = load_data_dictionary()

    if data_dict is None:
        st.info(
            "No data dictionary found. Add a CSV file at `references/data_dictionary.csv` "
            "with columns: `column_name`, `data_type`, `description`, `example`."
        )
    else:
        search = st.text_input("Search columns", placeholder="e.g. salary, borough...")
        if search:
            mask = data_dict.apply(
                lambda col: col.astype(str).str.contains(search, case=False)
            ).any(axis=1)
            data_dict = data_dict[mask]

        st.dataframe(data_dict, width="stretch", hide_index=True)

st.caption(APP_DISCLAIMER)
