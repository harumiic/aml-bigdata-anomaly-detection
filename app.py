"""
Main application

AML Detection System
"""

"""
Dashboard principal del sistema AML Anomaly Monitor.
"""

import streamlit as st

st.set_page_config(
    page_title="AML Anomaly Monitor",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AML Anomaly Monitor")

st.markdown(
    """
    Sistema académico para la detección y priorización de operaciones
    potencialmente sospechosas mediante **Isolation Forest**.
    """
)

st.info(
    "Usa el menú lateral para navegar entre Dashboard, Alertas, "
    "Transacciones y Reportes."
)

st.subheader("Flujo del sistema")

st.markdown(
    """
    1. Generación de datos sintéticos  
    2. Procesamiento ETL  
    3. Detección de anomalías  
    4. Registro en SQLite  
    5. Visualización y exportación
    """
)