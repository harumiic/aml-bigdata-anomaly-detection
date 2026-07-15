"""
Generador de datos sintéticos para el sistema AML.

Este módulo genera transacciones financieras simuladas con perfiles
de clientes, montos habituales y escenarios potencialmente sospechosos.
"""

from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from config import (
    ANOMALY_PERCENTAGE,
    DATASET_SIZE,
    RANDOM_STATE,
    RAW_DATA_DIR,
)


# =========================================================
# CONFIGURACIÓN DE REPRODUCIBILIDAD
# =========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =========================================================
# LISTAS BASE
# =========================================================

COUNTRIES = [
    "Perú",
    "Chile",
    "Colombia",
    "Brasil",
    "México",
    "Estados Unidos",
    "España",
    "Panamá",
    "Belice",
    "Islas Caimán",
    "Venezuela",
]

CHANNELS = [
    "Transferencia",
    "Depósito",
    "Retiro",
    "Pago Online",
    "POS",
    "Criptomonedas",
]

DEVICES = [
    "Mobile",
    "Desktop",
    "ATM",
]

TRANSACTION_TYPES = [
    "Transferencia",
    "Pago",
    "Compra",
    "Retiro",
]

RISK_LEVELS = [
    "Low",
    "Medium",
    "High",
]


# =========================================================
# GENERACIÓN DE PERFILES DE CLIENTES
# =========================================================

CUSTOMERS = {}

for i in range(1, 5001):
    customer_id = f"CUS{i:05d}"

    CUSTOMERS[customer_id] = {
        "average_amount": random.randint(100, 3000),
        "risk_level": random.choices(
            population=RISK_LEVELS,
            weights=[0.70, 0.22, 0.08],
            k=1,
        )[0],
    }


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def generate_transaction_id(index: int) -> str:
    """Genera un identificador único para una transacción."""
    return f"TX{index:08d}"


def generate_customer_id() -> str:
    """Selecciona aleatoriamente un cliente existente."""
    return random.choice(list(CUSTOMERS.keys()))


def generate_account(customer_id: str) -> str:
    """Genera una cuenta asociada de forma estable al cliente."""
    numeric_customer_id = int(customer_id.replace("CUS", ""))
    return f"ACC{numeric_customer_id:08d}"


def generate_amount(customer_id: str) -> float:
    """
    Genera un monto cercano al comportamiento habitual del cliente.

    Se usa una distribución normal centrada en el monto promedio
    asignado al perfil del cliente.
    """
    average_amount = CUSTOMERS[customer_id]["average_amount"]

    amount = np.random.normal(
        loc=average_amount,
        scale=average_amount * 0.25,
    )

    return round(max(float(amount), 10.0), 2)


def generate_timestamp() -> datetime:
    """Genera una fecha y hora aleatoria dentro del año 2025."""
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31, 23, 59, 59)

    total_seconds = int((end - start).total_seconds())
    random_seconds = random.randint(0, total_seconds)

    return start + timedelta(seconds=random_seconds)


# =========================================================
# INYECCIÓN DE ESCENARIOS AML
# =========================================================

def inject_anomaly(transaction: dict) -> dict:
    """
    Incorpora señales sintéticas de riesgo en una transacción.

    La etiqueta is_suspicious se utiliza únicamente como referencia
    para evaluar posteriormente el desempeño del modelo no supervisado.
    """
    risk_score = 0
    reasons = []

    # Inyección controlada de montos muy superiores al patrón normal.
    if random.random() < ANOMALY_PERCENTAGE:
        transaction["amount"] = round(
            random.uniform(15000, 50000),
            2,
        )

        risk_score += 40
        reasons.append("Monto inusualmente elevado")

    # Operación vinculada con un país definido como de mayor riesgo.
    if transaction["country"] in {
        "Panamá",
        "Belice",
        "Islas Caimán",
    }:
        risk_score += 20
        reasons.append("País de mayor riesgo")

    # Uso de un canal asociado con mayor dificultad de trazabilidad.
    if transaction["channel"] == "Criptomonedas":
        risk_score += 15
        reasons.append("Operación mediante criptomonedas")

    # Operación realizada en horario poco habitual.
    transaction_hour = transaction["timestamp"].hour

    if transaction_hour <= 4:
        risk_score += 15
        reasons.append("Operación en horario nocturno")

    # Riesgo previo asignado al perfil del cliente.
    if transaction["risk_level"] == "High":
        risk_score += 15
        reasons.append("Cliente con perfil de riesgo alto")

    elif transaction["risk_level"] == "Medium":
        risk_score += 5
        reasons.append("Cliente con perfil de riesgo medio")

    transaction["risk_score"] = min(risk_score, 100)
    transaction["reason"] = (
        "; ".join(reasons)
        if reasons
        else "Sin señales sintéticas relevantes"
    )

    transaction["is_suspicious"] = int(risk_score >= 40)

    return transaction


# =========================================================
# GENERACIÓN DE TRANSACCIONES
# =========================================================

def generate_transaction(index: int) -> dict:
    """Genera una transacción financiera sintética."""
    customer_id = generate_customer_id()

    transaction = {
        "transaction_id": generate_transaction_id(index),
        "customer_id": customer_id,
        "account_id": generate_account(customer_id),
        "timestamp": generate_timestamp(),
        "amount": generate_amount(customer_id),
        "country": random.choice(COUNTRIES),
        "channel": random.choice(CHANNELS),
        "device": random.choice(DEVICES),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "risk_level": CUSTOMERS[customer_id]["risk_level"],
    }

    return inject_anomaly(transaction)


def create_dataset() -> pd.DataFrame:
    """Genera el conjunto completo de transacciones."""
    transactions = [
        generate_transaction(index)
        for index in range(1, DATASET_SIZE + 1)
    ]

    dataframe = pd.DataFrame(transactions)

    dataframe = dataframe.sort_values(
        by="timestamp",
        ascending=True,
    ).reset_index(drop=True)

    return dataframe


# =========================================================
# ALMACENAMIENTO
# =========================================================

def save_dataset(dataframe: pd.DataFrame) -> None:
    """Guarda el dataset generado en formato CSV."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RAW_DATA_DIR / "transactions.csv"

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    suspicious_count = int(dataframe["is_suspicious"].sum())
    suspicious_percentage = (
        suspicious_count / len(dataframe) * 100
        if len(dataframe) > 0
        else 0
    )

    print("=" * 60)
    print("DATASET GENERADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Archivo: {output_file}")
    print(f"Total de transacciones: {len(dataframe):,}")
    print(f"Operaciones sospechosas: {suspicious_count:,}")
    print(f"Porcentaje sospechoso: {suspicious_percentage:.2f}%")
    print("=" * 60)
    print(dataframe.head())


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================

def main() -> None:
    """Ejecuta la generación y almacenamiento del dataset."""
    dataset = create_dataset()
    save_dataset(dataset)


if __name__ == "__main__":
    main()