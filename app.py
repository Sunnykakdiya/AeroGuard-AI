
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.ingestion import COLUMN_NAMES
from src.inference.predict import predict_latest_rul


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AeroGuard AI | Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
RUL_CAP = 125


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: #1b2230;
        border: 1px solid #303b4d;
        padding: 18px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_uploaded_data(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
            engine="python",
        )

    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        raise ValueError(
            "Please upload a .txt or .csv file."
        )

    return df


def get_status_counts(predictions):

    counts = (
        predictions["maintenance_status"]
        .value_counts()
    )

    return {
        "NORMAL": int(counts.get("NORMAL", 0)),
        "WARNING": int(counts.get("WARNING", 0)),
        "CRITICAL": int(counts.get("CRITICAL", 0)),
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("✈️ AeroGuard AI")

st.sidebar.caption(
    "Jet Engine Predictive Maintenance System"
)

st.sidebar.divider()

st.sidebar.subheader("Prediction Settings")

st.sidebar.info(
    """
    **Model:** Random Forest

    **Dataset:** NASA C-MAPSS FD001

    **Features:** 130

    **RUL prediction range:** 0–125 cycles
    """
)

st.sidebar.caption(
    "Maintenance thresholds are preliminary project settings."
)

st.sidebar.divider()

st.sidebar.markdown(
    "**Developed as an AI/ML predictive maintenance project.**"
)


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("✈️ AeroGuard AI")

st.markdown(
    "### Jet Engine Predictive Maintenance Dashboard"
)

st.write(
    "Monitor engine health and estimate Remaining Useful Life (RUL) "
    "using historical sensor readings."
)

st.divider()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.subheader("Upload Engine Sensor Data")

st.write(
    "Upload raw NASA C-MAPSS sensor readings in TXT or CSV format. "
    "The file should contain multiple chronological readings per engine."
)

uploaded_file = st.file_uploader(
    "Choose sensor data file",
    type=["txt", "csv"],
)

if uploaded_file is None:

    st.info(
        "Upload a sensor dataset to generate engine predictions. "
        "For your first test, use data/raw/test_FD001.txt."
    )

    st.stop()


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

if st.button(
    "Generate RUL Predictions",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Processing sensor readings and predicting engine health..."
        ):

            raw_data = load_uploaded_data(
                uploaded_file
            )

            predictions = predict_latest_rul(
                raw_data
            )

        st.session_state["predictions"] = predictions
        st.session_state["raw_data"] = raw_data

        st.success(
            "Predictions generated successfully!"
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "predictions" in st.session_state:

    predictions = st.session_state["predictions"]

    st.divider()

    st.subheader("Engine Health Overview")

    status_counts = get_status_counts(
        predictions
    )

    total_engines = len(predictions)

    # KPI METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Engines",
        total_engines,
    )

    col2.metric(
        "Normal",
        status_counts["NORMAL"],
    )

    col3.metric(
        "Warning",
        status_counts["WARNING"],
    )

    col4.metric(
        "Critical",
        status_counts["CRITICAL"],
    )

    st.divider()

    # STATUS DISTRIBUTION
    st.subheader("Maintenance Status Distribution")

    status_df = pd.DataFrame({
        "Status": list(status_counts.keys()),
        "Engines": list(status_counts.values()),
    }).set_index("Status")

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            status_df,
            y="Engines",
        )

    with col2:

        st.write("Status percentages")

        for status, count in status_counts.items():

            percentage = (
                count / total_engines * 100
                if total_engines else 0
            )

            st.write(
                f"**{status}:** {count} engines "
                f"({percentage:.1f}%)"
            )

            st.progress(
                percentage / 100
            )

    st.divider()

    # RUL DISTRIBUTION
    st.subheader("Engine Remaining Useful Life")

    chart_data = predictions.set_index(
        "engine_id"
    )[["predicted_RUL"]]

    st.bar_chart(
    predictions,
    x="engine_id",
    y="predicted_RUL",
    )
    

    st.caption(
        "Predicted RUL is expressed in operating cycles. "
        "Predictions are capped at 125 cycles."
    )

    st.divider()

    # ENGINE FILTER
    st.subheader("Individual Engine Analysis")

    engine_ids = sorted(
        predictions["engine_id"].unique()
    )

    selected_engine = st.selectbox(
        "Select Engine ID",
        engine_ids,
    )

    engine_result = predictions[
        predictions["engine_id"] == selected_engine
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Engine ID",
        int(engine_result["engine_id"]),
    )

    col2.metric(
        "Latest Cycle",
        int(engine_result["latest_cycle"]),
    )

    col3.metric(
        "Predicted RUL",
        f"{engine_result['predicted_RUL']:.2f} cycles",
    )

    status = engine_result["maintenance_status"]

    if status == "CRITICAL":

        st.error(
            "CRITICAL: This engine has a low predicted remaining useful life."
        )

    elif status == "WARNING":

        st.warning(
            "WARNING: This engine should receive closer monitoring."
        )

    else:

        st.success(
            "NORMAL: The predicted remaining useful life is above the warning threshold."
        )

    st.divider()

    # COMPLETE RESULTS TABLE
    st.subheader("All Engine Predictions")

    st.dataframe(
        predictions.style.format({
            "predicted_RUL": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # DOWNLOAD RESULTS
    csv_data = predictions.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="aeroguard_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "AeroGuard AI | Predictive Maintenance | NASA C-MAPSS FD001"
    )
