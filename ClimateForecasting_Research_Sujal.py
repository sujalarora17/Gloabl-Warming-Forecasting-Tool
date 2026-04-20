
# python -m streamlit run ClimateForecasting_Research_Sujal.py

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO

#  Prophet import
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

APP_TITLE = "🌍 Climate Trend Forecasting System"
APP_ICON = "🌍"
DATA_FILE = "climate_trend_dataset.csv"  # <-- renamed dataset file



# Page theme

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly"   # "plotly" or "ggplot2"


# CSS (Heading fix + spacing)

st.markdown("""
<style>

/* Push whole page content lower so title isn't cut */
.block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 2rem !important;
}

/* Sidebar width */
section[data-testid="stSidebar"] { width: 320px !important; }

/* Hero block */
.hero-wrap { margin-top: 18px !important; }
.hero-title {
    font-size: 2.55rem !important;
    font-weight: 900 !important;
    line-height: 1.25 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.hero-sub {
    opacity: 0.9;
    font-size: 1.05rem;
    margin-top: 8px !important;
    line-height: 1.4 !important;
}

/* Cards */
.card {
  background: rgba(0,0,0,0.04);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 18px;
  padding: 16px 18px;
  margin: 14px 0;
  box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 14px !important;
  padding: 0.65rem 1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

</style>
""", unsafe_allow_html=True)


# Data Loading

@st.cache_data
def load_data(file_path: str = DATA_FILE) -> pd.DataFrame:
    climate_df = pd.read_csv(file_path)
    if "Year" in climate_df.columns:
        climate_df["Year"] = pd.to_numeric(climate_df["Year"], errors="coerce")
    return climate_df


# Helper Functions

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_test_split_time_series(series: pd.Series, test_size: float = 0.2):
    series = series.dropna().astype(float)
    n = len(series)
    test_n = max(1, int(n * test_size))
    train = series.iloc[:-test_n]
    test = series.iloc[-test_n:]
    return train, test


def coerce_numeric_df(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Convert as many columns as possible to numeric.
    Non-convertible values become NaN.
    Prevents crashes like: 'could not convert string to float: Afghanistan'
    """
    temp = df_.copy()
    for c in temp.columns:
        if temp[c].dtype == "object":
            temp[c] = temp[c].astype(str).str.replace(",", "", regex=False).str.strip()
        temp[c] = pd.to_numeric(temp[c], errors="coerce")
    return temp.select_dtypes(include=["number"]).copy()


def year_to_datetime(df_: pd.DataFrame, year_col: str = "Year") -> pd.Series:
    years = pd.to_numeric(df_[year_col], errors="coerce")
    return pd.to_datetime(years.astype("Int64").astype(str) + "-12-31", errors="coerce")


def simulate_climate_scenario(
    climate_df: pd.DataFrame,
    co2_change: float,
    ch4_change: float,
    n2o_change: float
) -> pd.DataFrame:
    """
    Simple regression-based scenario simulation.
    Works ONLY if the dataset includes the required gas columns.
    """
    simulated_df = climate_df.copy()

    needed_cols = ["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb", "Temperature_Anomaly_C"]
    missing = [c for c in needed_cols if c not in simulated_df.columns]
    if missing:
        raise ValueError(f"Missing columns for scenario model: {missing}")

    simulated_df["CO2_Concentration_ppm"] = pd.to_numeric(simulated_df["CO2_Concentration_ppm"], errors="coerce") + co2_change
    simulated_df["CH4_Concentration_ppb"] = pd.to_numeric(simulated_df["CH4_Concentration_ppb"], errors="coerce") + ch4_change
    simulated_df["N2O_Concentration_ppb"] = pd.to_numeric(simulated_df["N2O_Concentration_ppb"], errors="coerce") + n2o_change
    simulated_df["Temperature_Anomaly_C"] = pd.to_numeric(simulated_df["Temperature_Anomaly_C"], errors="coerce")

    X = simulated_df[["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb"]].dropna()
    y = simulated_df.loc[X.index, "Temperature_Anomaly_C"].dropna()
    X = X.loc[y.index]

    model = LinearRegression()
    model.fit(X, y)

    simulated_df["Predicted_Temperature_Anomaly_C"] = np.nan
    simulated_df.loc[X.index, "Predicted_Temperature_Anomaly_C"] = model.predict(X)
    return simulated_df


# Sidebar Navigation

st.sidebar.markdown("## 🌍 Climate Trend Dashboard")
menu_choice = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Scenario Analysis",
        "📈 Visualizations",
        "🔬 Forecast + Evaluation (MAE/RMSE)",
        "📥 Upload & Analyze (Any CSV)",
        "📋 Reports",
        "ℹ️ About",
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Research mode: MAE/RMSE evaluation + model comparison")


# Load Dataset

try:
    climate_df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Could not load `{DATA_FILE}`. Keep it in the same folder as app.py.\n\nError: {e}")
    st.stop()

for req in ["Year", "Temperature_Anomaly_C"]:
    if req not in climate_df.columns:
        st.error(f"Dataset missing required column: `{req}`")
        st.stop()


# HOME

if menu_choice == "🏠 Home":
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">🌍 Climate Change Trend Forecasting & Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Trend analysis, scenario simulation, forecasting + MAE/RMSE evaluation.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("This research evaluates statistical forecasting techniques including ARIMA and Prophet to model long-term climate anomaly trends.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", f"{int(np.nanmin(climate_df['Year']))} – {int(np.nanmax(climate_df['Year']))}")
    c2.metric("Records", f"{len(climate_df):,}")
    c3.metric("Latest Anomaly (°C)", f"{pd.to_numeric(climate_df['Temperature_Anomaly_C'], errors='coerce').dropna().iloc[-1]:.3f}")
    c4.metric("Prophet Available", "Yes ✅" if PROPHET_OK else "No ❌")

    tab1, tab2, tab3 = st.tabs(["📌 Overview", "📈 Trend", "🔎 Data Preview"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Research Objectives")
        st.write(
            "- Compare ARIMA vs Prophet for temperature anomaly forecasting\n"
            "- Evaluate models using MAE and RMSE\n"
            "- Simulate greenhouse gas scenarios (if gas columns exist)\n"
            "- Analyze correlations between variables\n"
            "- Support any CSV upload for robust analytics"
        )
        st.info("SDG Alignment: **SDG 13 – Climate Action**")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.line(climate_df, x="Year", y="Temperature_Anomaly_C", title="Temperature Anomaly Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(climate_df.head(30), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

# SCENARIO ANALYSIS

elif menu_choice == "📊 Scenario Analysis":
    st.header("📊 Scenario Analysis")
    st.caption("Adjust CO₂ / CH₄ / N₂O and see predicted anomaly changes (requires gas columns).")

    needed = {"Year", "CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb", "Temperature_Anomaly_C"}
    if not needed.issubset(climate_df.columns):
        st.warning("Scenario simulation is disabled because the base dataset does not contain the required gas columns.")
        st.stop()

    co2_change = st.slider("CO₂ change (ppm)", -50.0, 50.0, 0.0, step=1.0)
    ch4_change = st.slider("CH₄ change (ppb)", -200.0, 200.0, 0.0, step=10.0)
    n2o_change = st.slider("N₂O change (ppb)", -20.0, 20.0, 0.0, step=1.0)

    simulated_df = simulate_climate_scenario(climate_df, co2_change, ch4_change, n2o_change)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = px.line(
        simulated_df,
        x="Year",
        y=["Temperature_Anomaly_C", "Predicted_Temperature_Anomaly_C"],
        title="Actual vs Scenario Predicted Temperature Anomaly",
        labels={"value": "Temperature Anomaly (°C)", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)



# VISUALIZATIONS

elif menu_choice == "📈 Visualizations":
    st.header("📈 Visualizations")

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Temperature Anomaly Distribution")
        series = pd.to_numeric(climate_df["Temperature_Anomaly_C"], errors="coerce")
        fig = px.histogram(series.dropna(), nbins=25, title="Histogram of Temperature Anomaly")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Correlation Heatmap (numeric only)")
        num_df = coerce_numeric_df(climate_df)
        if num_df.shape[1] < 2:
            st.warning("Not enough numeric columns.")
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)



# FORECAST + EVALUATION

elif menu_choice == "🔬 Forecast + Evaluation (MAE/RMSE)":
    st.header("🔬 Forecast + Evaluation (MAE & RMSE)")
    st.caption("Train/Test split on time series, evaluate predictions, then forecast future values.")

    horizon = st.slider("Future forecast horizon (years)", 5, 80, 20, step=5)
    test_ratio = st.slider("Test ratio", 0.1, 0.4, 0.2, step=0.05)

    series = pd.to_numeric(climate_df["Temperature_Anomaly_C"], errors="coerce").dropna().astype(float)
    years = pd.to_numeric(climate_df["Year"], errors="coerce").dropna().astype(int)

    if len(series) < 12:
        st.warning("Not enough data points for forecasting.")
        st.stop()

    train_y, test_y = train_test_split_time_series(series, test_ratio)
    train_years = years.iloc[:len(train_y)].tolist()
    test_years = years.iloc[-len(test_y):].tolist()

    # ---- ARIMA ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ARIMA Evaluation + Forecast")

    arima_order = st.selectbox("ARIMA order (p,d,q)", [(1, 1, 1), (2, 1, 2), (3, 1, 3)], index=1)

    arima_model = ARIMA(train_y, order=arima_order)
    arima_fit = arima_model.fit()
    arima_test_pred = arima_fit.forecast(steps=len(test_y))
    arima_future = arima_fit.forecast(steps=horizon)

    arima_mae = mean_absolute_error(test_y, arima_test_pred)
    arima_rmse = rmse(test_y, arima_test_pred)

    m1, m2 = st.columns(2)
    m1.metric("ARIMA MAE", f"{arima_mae:.4f}")
    m2.metric("ARIMA RMSE", f"{arima_rmse:.4f}")

    last_year = int(years.max())
    future_years = list(range(last_year + 1, last_year + 1 + horizon))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_years, y=train_y, mode="lines", name="Train"))
    fig.add_trace(go.Scatter(x=test_years, y=test_y, mode="lines", name="Test"))
    fig.add_trace(go.Scatter(x=test_years, y=arima_test_pred, mode="lines", name="ARIMA Test Pred"))
    fig.add_trace(go.Scatter(x=future_years, y=arima_future, mode="lines", name="ARIMA Future"))
    fig.update_layout(title="ARIMA: Evaluation + Future Forecast", xaxis_title="Year", yaxis_title="Anomaly (°C)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prophet 
    prophet_rmse_val = None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prophet Evaluation + Forecast")

    if not PROPHET_OK:
        st.warning("Prophet not installed. ARIMA evaluation is still valid for research demo.")
    else:
        prophet_df = pd.DataFrame({
            "ds": year_to_datetime(climate_df, "Year"),
            "y": pd.to_numeric(climate_df["Temperature_Anomaly_C"], errors="coerce"),
        }).dropna().sort_values("ds").reset_index(drop=True)

        n = len(prophet_df)
        test_n = max(1, int(n * test_ratio))
        train_p = prophet_df.iloc[:-test_n]
        test_p = prophet_df.iloc[-test_n:]

        p_model = Prophet()
        p_model.fit(train_p)

        future_test = p_model.make_future_dataframe(periods=test_n, freq="Y")
        fc_test = p_model.predict(future_test)

        yhat_test = fc_test["yhat"].tail(test_n).values
        y_true = test_p["y"].values

        prophet_mae = mean_absolute_error(y_true, yhat_test)
        prophet_rmse_val = rmse(y_true, yhat_test)

        p1, p2 = st.columns(2)
        p1.metric("Prophet MAE", f"{prophet_mae:.4f}")
        p2.metric("Prophet RMSE", f"{prophet_rmse_val:.4f}")

        final_p = Prophet()
        final_p.fit(prophet_df)
        future = final_p.make_future_dataframe(periods=horizon, freq="Y")
        fc = final_p.predict(future)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Actual"))
        fig2.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Prophet Forecast"))
        fig2.update_layout(title="Prophet: Future Forecast", xaxis_title="Year", yaxis_title="Anomaly (°C)")
        st.plotly_chart(fig2, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    #  Comparison 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏆 Model Comparison")
    if prophet_rmse_val is not None:
        best = "Prophet" if prophet_rmse_val < arima_rmse else "ARIMA"
        st.success(f"Based on RMSE, better model on this split: **{best}**")
    else:
        st.info("Only ARIMA evaluated (Prophet not available).")
    st.markdown("</div>", unsafe_allow_html=True)


# UPLOAD & ANALYZE (ANY CSV)

elif menu_choice == "📥 Upload & Analyze (Any CSV)":
    st.header("📥 Upload & Analyze Any CSV")
    st.caption("Numeric conversion is automatic; text columns won’t crash correlation.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV to start.")
        st.stop()

    user_df = pd.read_csv(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview")
    st.dataframe(user_df.head(30), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    numeric_df = coerce_numeric_df(user_df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Summary Statistics (Numeric Only)")
    if numeric_df.empty:
        st.warning("No numeric columns found or could be converted.")
    else:
        st.dataframe(numeric_df.describe().T, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Explore")
    option = st.selectbox("Choose analysis", ["Correlation Heatmap", "Histogram", "Scatter Plot", "Top Categories (Text)"])

    if option == "Correlation Heatmap":
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif option == "Histogram":
        if numeric_df.empty:
            st.warning("No numeric columns available.")
        else:
            col = st.selectbox("Column", numeric_df.columns.tolist())
            bins = st.slider("Bins", 5, 60, 20)
            fig, ax = plt.subplots()
            ax.hist(numeric_df[col].dropna(), bins=bins)
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)

    elif option == "Scatter Plot":
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            cols = numeric_df.columns.tolist()
            x_col = st.selectbox("X-axis", cols, index=0)
            y_col = st.selectbox("Y-axis", cols, index=1)
            fig = px.scatter(numeric_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

    else:
        text_cols = [c for c in user_df.columns if user_df[c].dtype == "object"]
        if not text_cols:
            st.info("No text columns found.")
        else:
            col = st.selectbox("Text column", text_cols)
            top_n = st.slider("Top N", 5, 30, 10)
            vc = user_df[col].astype(str).value_counts().head(top_n).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=f"Top {top_n} values in {col}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# REPORTS

elif menu_choice == "📋 Reports":
    st.header("📋 Reports")
    st.caption("Download the base dataset as CSV or Excel.")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    buffer_csv = BytesIO()
    climate_df.to_csv(buffer_csv, index=False)
    buffer_csv.seek(0)

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        climate_df.to_excel(writer, index=False, sheet_name="ClimateData")
    buffer_excel.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV", data=buffer_csv, file_name="climate_trend_data.csv", mime="text/csv")
    with col2:
        st.download_button(
            "⬇️ Download Excel",
            data=buffer_excel,
            file_name="climate_trend_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.info("PDF export + advanced deep learning models can be added later for the Major Project.")
    st.markdown("</div>", unsafe_allow_html=True)


# ABOUT

elif menu_choice == "ℹ️ About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("ℹ️ About (Research-Oriented)")

    st.write("""
**Developed By:** Sujal Arora  
**Department:** CSE Core  
**Institute:** SRM Institute of Science & Technology  
**Project Type:** Research-Oriented  

This system was developed to evaluate statistical time series models for long-term climate anomaly prediction.

**Research Contribution**
- Compare ARIMA vs Prophet forecasting  
- Evaluate using **MAE** and **RMSE**  
- Provide analytics dashboard + safe CSV exploration

**Major Project Ideas**
- LSTM / GRU deep learning forecasting  
- Multivariate forecasting with exogenous variables  
- Rolling-window time series cross-validation  
- More metrics: MAPE, R², AIC/BIC  
- Live data APIs + automated PDF reporting
""")

    # Honest attribution line (recommended)
    st.caption("Note: If you used any external code/repo as a reference, mention it in your report as 'inspired by/extended from' and cite it. That's the correct academic approach.")
    st.markdown("</div>", unsafe_allow_html=True)