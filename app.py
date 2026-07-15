"""
Main application

AML Detection System
"""

"""
Dashboard principal del sistema AML Anomaly Monitor.
"""

import streamlit as st

st.set_page_config(
    page_title="AML Detection System",
    page_icon="🔎",
    layout="wide",
)

st.title("Sistema Inteligente para la Detección de Operaciones Sospechosas")

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

import streamlit as st

st.title("Acerca del Proyecto")

st.markdown("""
## Sistema de Big Data para la Detección de Operaciones Sospechosas de Lavado de Activos mediante Modelos de Anomalías

### Objetivo

Detectar operaciones financieras potencialmente sospechosas utilizando técnicas de Machine Learning no supervisado.

### Tecnologías

- Python
- Streamlit
- Pandas
- Scikit-Learn
- SQLite
- Plotly

### Modelo utilizado

Isolation Forest

### Dataset

100000 transacciones sintéticas

### Autor

Harumi Contreras
""")


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