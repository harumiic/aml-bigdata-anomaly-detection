"""
ETL Pipeline
"""

"""
Pipeline ETL para la preparación de transacciones financieras.

Este módulo valida, limpia y transforma el dataset sintético
antes de utilizarlo en el modelo de detección de anomalías.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_id",
    "timestamp",
    "amount",
    "country",
    "channel",
    "device",
    "transaction_type",
    "risk_level",
]


def load_raw_data(
    file_path: Path | None = None,
) -> pd.DataFrame:
    """
    Carga el archivo CSV de transacciones.

    Parameters
    ----------
    file_path:
        Ruta opcional del archivo. Si no se especifica, se utiliza
        data/raw/transactions.csv.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """
    input_file = file_path or RAW_DATA_DIR / "transactions.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {input_file}"
        )

    dataframe = pd.read_csv(input_file)

    return dataframe


def validate_schema(dataframe: pd.DataFrame) -> None:
    """
    Verifica que el dataset contenga las columnas obligatorias.

    Raises
    ------
    ValueError
        Si falta una o más columnas requeridas.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "El dataset no contiene las columnas obligatorias: "
            + ", ".join(missing_columns)
        )


def clean_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia duplicados, tipos inválidos y valores faltantes.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Dataset limpio y resumen del proceso.
    """
    cleaned = dataframe.copy()

    initial_rows = len(cleaned)

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="coerce",
    )

    cleaned["amount"] = pd.to_numeric(
        cleaned["amount"],
        errors="coerce",
    )

    duplicated_rows = int(
        cleaned.duplicated(
            subset=["transaction_id"],
        ).sum()
    )

    cleaned = cleaned.drop_duplicates(
        subset=["transaction_id"],
        keep="first",
    )

    invalid_timestamp_rows = int(
        cleaned["timestamp"].isna().sum()
    )

    invalid_amount_rows = int(
        cleaned["amount"].isna().sum()
    )

    cleaned = cleaned.dropna(
        subset=[
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
        ]
    )

    cleaned = cleaned[
        cleaned["amount"] > 0
    ].copy()

    categorical_columns = [
        "country",
        "channel",
        "device",
        "transaction_type",
        "risk_level",
    ]

    for column in categorical_columns:
        cleaned[column] = (
            cleaned[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    cleaned = cleaned.reset_index(drop=True)

    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "removed_rows": initial_rows - len(cleaned),
        "duplicated_rows": duplicated_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "invalid_amount_rows": invalid_amount_rows,
    }

    return cleaned, summary


def add_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables temporales derivadas.
    """
    enriched = dataframe.copy()

    enriched["transaction_hour"] = enriched["timestamp"].dt.hour
    enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
    enriched["day_name"] = enriched["timestamp"].dt.day_name()
    enriched["month"] = enriched["timestamp"].dt.month

    enriched["is_weekend"] = (
        enriched["day_of_week"] >= 5
    ).astype(int)

    enriched["is_night"] = (
        enriched["transaction_hour"] <= 4
    ).astype(int)

    return enriched


def add_amount_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables relacionadas con el monto de la operación.
    """
    enriched = dataframe.copy()

    enriched["amount_log"] = np.log1p(
        enriched["amount"]
    )

    customer_statistics = (
        enriched.groupby("customer_id")["amount"]
        .agg(
            customer_amount_mean="mean",
            customer_amount_median="median",
            customer_amount_std="std",
        )
        .reset_index()
    )

    enriched = enriched.merge(
        customer_statistics,
        on="customer_id",
        how="left",
    )

    enriched["customer_amount_std"] = (
        enriched["customer_amount_std"]
        .fillna(0)
    )

    enriched["amount_to_customer_mean"] = (
        enriched["amount"]
        / enriched["customer_amount_mean"].replace(0, np.nan)
    )

    enriched["amount_to_customer_median"] = (
        enriched["amount"]
        / enriched["customer_amount_median"].replace(0, np.nan)
    )

    enriched[
        [
            "amount_to_customer_mean",
            "amount_to_customer_median",
        ]
    ] = enriched[
        [
            "amount_to_customer_mean",
            "amount_to_customer_median",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    return enriched


def add_frequency_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la frecuencia diaria de operaciones por cliente.
    """
    enriched = dataframe.copy()

    enriched["transaction_date"] = (
        enriched["timestamp"].dt.date
    )

    daily_frequency = (
        enriched.groupby(
            ["customer_id", "transaction_date"]
        )["transaction_id"]
        .transform("count")
    )

    enriched["customer_daily_frequency"] = daily_frequency

    return enriched


def add_risk_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables binarias de riesgo.
    """
    enriched = dataframe.copy()

    high_risk_countries = {
        "Panamá",
        "Belice",
        "Islas Caimán",
        "Venezuela",
    }

    enriched["is_high_risk_country"] = (
        enriched["country"]
        .isin(high_risk_countries)
        .astype(int)
    )

    enriched["is_crypto_channel"] = (
        enriched["channel"]
        .eq("Criptomonedas")
        .astype(int)
    )

    risk_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    enriched["customer_risk_numeric"] = (
        enriched["risk_level"]
        .map(risk_mapping)
        .fillna(0)
        .astype(int)
    )

    return enriched


def preprocess_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline completo de preparación de datos.
    """
    validate_schema(dataframe)

    processed, summary = clean_data(dataframe)

    processed = add_time_features(processed)
    processed = add_amount_features(processed)
    processed = add_frequency_features(processed)
    processed = add_risk_features(processed)

    processed = processed.sort_values(
        by="timestamp",
        ascending=True,
    ).reset_index(drop=True)

    return processed, summary


def save_processed_data(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Guarda el dataset procesado en formato CSV.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DATA_DIR
        / "transactions_processed.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    return output_file


def main() -> None:
    """
    Ejecuta el pipeline ETL completo.
    """
    raw_dataframe = load_raw_data()

    processed_dataframe, summary = preprocess_data(
        raw_dataframe
    )

    output_file = save_processed_data(
        processed_dataframe
    )

    print("=" * 60)
    print("PIPELINE ETL EJECUTADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Registros iniciales: {summary['initial_rows']:,}")
    print(f"Registros finales: {summary['final_rows']:,}")
    print(f"Duplicados eliminados: {summary['duplicated_rows']:,}")
    print(
        "Fechas inválidas eliminadas: "
        f"{summary['invalid_timestamp_rows']:,}"
    )
    print(
        "Montos inválidos eliminados: "
        f"{summary['invalid_amount_rows']:,}"
    )
    print(f"Archivo generado: {output_file}")
    print("=" * 60)

    selected_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "transaction_hour",
        "is_weekend",
        "amount_log",
        "amount_to_customer_mean",
        "customer_daily_frequency",
        "is_high_risk_country",
        "customer_risk_numeric",
    ]

    print(
        processed_dataframe[
            selected_columns
        ].head()
    )


if __name__ == "__main__":
    main()
