import pandas as pd
import plotly.express as px
import streamlit as st

from services.database import get_alerts, get_executions


st.set_page_config(
    page_title="Dashboard AML",
    page_icon="📊",
    layout="wide",
)

st.title("Dashboard AML")


@st.cache_data(ttl=30)
def load_executions() -> pd.DataFrame:
    return get_executions()


@st.cache_data(ttl=30)
def load_alerts() -> pd.DataFrame:
    return get_alerts()


executions = load_executions()
alerts = load_alerts()

if executions.empty:
    st.warning(
        "No existen ejecuciones registradas. Ejecuta primero:\n\n"
        "`python -m services.anomaly_detection`"
    )
    st.stop()

last_execution = executions.iloc[0]

total_transactions = int(last_execution["total_transactions"])
alerts_detected = int(last_execution["alerts_detected"])
alert_rate = float(last_execution["alert_rate"])
contamination = float(last_execution["contamination"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transacciones procesadas", f"{total_transactions:,}")
col2.metric("Alertas detectadas", f"{alerts_detected:,}")
col3.metric("Tasa de alertas", f"{alert_rate:.2f}%")
col4.metric("Contamination", f"{contamination:.3f}")

st.divider()

if alerts.empty:
    st.info("No existen alertas registradas.")
    st.stop()

chart_col1, chart_col2 = st.columns(2)

risk_distribution = (
    alerts["model_risk_level"]
    .value_counts()
    .reset_index()
)

risk_distribution.columns = ["Nivel de riesgo", "Cantidad"]

risk_figure = px.bar(
    risk_distribution,
    x="Nivel de riesgo",
    y="Cantidad",
    title="Distribución de alertas por riesgo",
    text_auto=True,
)

chart_col1.plotly_chart(
    risk_figure,
    use_container_width=True,
)

channel_distribution = (
    alerts["channel"]
    .value_counts()
    .reset_index()
)

channel_distribution.columns = ["Canal", "Cantidad"]

channel_figure = px.pie(
    channel_distribution,
    names="Canal",
    values="Cantidad",
    title="Alertas por canal",
    hole=0.45,
)

chart_col2.plotly_chart(
    channel_figure,
    use_container_width=True,
)

chart_col3, chart_col4 = st.columns(2)

country_distribution = (
    alerts["country"]
    .value_counts()
    .head(10)
    .reset_index()
)

country_distribution.columns = ["País", "Cantidad"]

country_figure = px.bar(
    country_distribution,
    x="País",
    y="Cantidad",
    title="Principales países asociados a alertas",
    text_auto=True,
)

chart_col3.plotly_chart(
    country_figure,
    use_container_width=True,
)

score_figure = px.histogram(
    alerts,
    x="anomaly_score",
    nbins=30,
    title="Distribución del puntaje de anomalía",
)

chart_col4.plotly_chart(
    score_figure,
    use_container_width=True,
)

st.subheader("Alertas de mayor prioridad")

columns_to_show = [
    "transaction_id",
    "customer_id",
    "amount",
    "country",
    "channel",
    "anomaly_score",
    "model_risk_level",
    "model_reason",
]

st.dataframe(
    alerts[columns_to_show]
    .sort_values(
        by="anomaly_score",
        ascending=False,
    )
    .head(20),
    use_container_width=True,
    hide_index=True,
)