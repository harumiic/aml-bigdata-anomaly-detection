import pandas as pd
import streamlit as st

from services.database import get_alerts


st.set_page_config(
    page_title="Alertas AML",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Alertas sospechosas")


@st.cache_data(ttl=30)
def load_alerts() -> pd.DataFrame:
    return get_alerts()


alerts = load_alerts()

if alerts.empty:
    st.info("No existen alertas registradas.")
    st.stop()

risk_options = sorted(
    alerts["model_risk_level"]
    .dropna()
    .unique()
    .tolist()
)

selected_risks = st.multiselect(
    "Filtrar por nivel de riesgo",
    options=risk_options,
    default=risk_options,
)

country_options = [
    "Todos",
    *sorted(
        alerts["country"]
        .dropna()
        .unique()
        .tolist()
    ),
]

selected_country = st.selectbox(
    "Filtrar por país",
    options=country_options,
)

channel_options = [
    "Todos",
    *sorted(
        alerts["channel"]
        .dropna()
        .unique()
        .tolist()
    ),
]

selected_channel = st.selectbox(
    "Filtrar por canal",
    options=channel_options,
)

filtered_alerts = alerts.copy()

if selected_risks:
    filtered_alerts = filtered_alerts[
        filtered_alerts["model_risk_level"].isin(
            selected_risks
        )
    ]

if selected_country != "Todos":
    filtered_alerts = filtered_alerts[
        filtered_alerts["country"] == selected_country
    ]

if selected_channel != "Todos":
    filtered_alerts = filtered_alerts[
        filtered_alerts["channel"] == selected_channel
    ]

st.metric(
    "Alertas mostradas",
    f"{len(filtered_alerts):,}",
)

display_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "country",
    "channel",
    "device",
    "anomaly_score",
    "model_risk_level",
    "model_reason",
    "status",
]

st.dataframe(
    filtered_alerts[display_columns],
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_alerts.to_csv(
    index=False,
).encode("utf-8-sig")

st.download_button(
    label="⬇️ Descargar alertas filtradas",
    data=csv_data,
    file_name="alertas_filtradas.csv",
    mime="text/csv",
)