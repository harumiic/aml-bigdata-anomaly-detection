import pandas as pd
import streamlit as st

from services.database import (
    get_alerts,
    get_executions,
)


st.set_page_config(
    page_title="Reportes AML",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Reportes")


executions = get_executions()
alerts = get_alerts()

st.subheader("Historial de ejecuciones")

if executions.empty:
    st.info("No existen ejecuciones registradas.")
else:
    history_columns = [
        "execution_id",
        "executed_at",
        "source_name",
        "total_transactions",
        "alerts_detected",
        "alert_rate",
        "contamination",
        "model_name",
    ]

    st.dataframe(
        executions[history_columns],
        use_container_width=True,
        hide_index=True,
    )

    execution_csv = executions.to_csv(
        index=False,
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Descargar historial",
        data=execution_csv,
        file_name="historial_ejecuciones.csv",
        mime="text/csv",
    )

st.divider()

st.subheader("Reporte general de alertas")

if alerts.empty:
    st.info("No existen alertas registradas.")
else:
    alerts_csv = alerts.to_csv(
        index=False,
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Descargar todas las alertas",
        data=alerts_csv,
        file_name="reporte_alertas_aml.csv",
        mime="text/csv",
    )

st.warning(
    "Las alertas generadas representan señales para revisión "
    "humana y no constituyen confirmaciones de lavado de activos."
)