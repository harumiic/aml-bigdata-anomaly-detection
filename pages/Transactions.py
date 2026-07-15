from pathlib import Path

import pandas as pd
import streamlit as st

from config import OUTPUT_DATA_DIR


st.set_page_config(
    page_title="Transacciones",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Transacciones procesadas")


@st.cache_data(ttl=30)
def load_transactions() -> pd.DataFrame:
    file_path = (
        OUTPUT_DATA_DIR
        / "transactions_scored.csv"
    )

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


transactions = load_transactions()

if transactions.empty:
    st.warning(
        "No se encontró el archivo de transacciones procesadas."
    )
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total de transacciones",
    f"{len(transactions):,}",
)

col2.metric(
    "Operaciones normales",
    f"{(transactions['model_prediction'] == 0).sum():,}",
)

col3.metric(
    "Operaciones anómalas",
    f"{(transactions['model_prediction'] == 1).sum():,}",
)

search_text = st.text_input(
    "Buscar por ID de transacción o cliente"
)

filtered_transactions = transactions.copy()

if search_text:
    search_mask = (
        filtered_transactions["transaction_id"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False,
        )
        |
        filtered_transactions["customer_id"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False,
        )
    )

    filtered_transactions = filtered_transactions[
        search_mask
    ]

columns_to_show = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "country",
    "channel",
    "device",
    "transaction_type",
    "anomaly_score",
    "model_status",
    "model_risk_level",
]

st.dataframe(
    filtered_transactions[columns_to_show],
    use_container_width=True,
    hide_index=True,
)