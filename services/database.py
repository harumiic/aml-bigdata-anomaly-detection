"""
Persistencia SQLite para ejecuciones y alertas AML.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """
    Crea una conexión con la base de datos SQLite.
    """
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """
    Crea las tablas necesarias si todavía no existen.
    """
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                executed_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                total_transactions INTEGER NOT NULL,
                alerts_detected INTEGER NOT NULL,
                alert_rate REAL NOT NULL,
                contamination REAL NOT NULL,
                model_name TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                timestamp TEXT,
                amount REAL,
                country TEXT,
                channel TEXT,
                device TEXT,
                transaction_type TEXT,
                customer_risk TEXT,
                anomaly_score REAL,
                model_risk_level TEXT,
                model_reason TEXT,
                status TEXT DEFAULT 'Pendiente',
                FOREIGN KEY (execution_id)
                    REFERENCES executions(execution_id)
            )
            """
        )

        connection.commit()


def save_execution(
    source_name: str,
    total_transactions: int,
    alerts_detected: int,
    alert_rate: float,
    contamination: float,
    model_name: str = "Isolation Forest",
) -> int:
    """
    Registra una ejecución y devuelve su identificador.
    """
    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO executions (
                executed_at,
                source_name,
                total_transactions,
                alerts_detected,
                alert_rate,
                contamination,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(
                    sep=" ",
                    timespec="seconds",
                ),
                source_name,
                total_transactions,
                alerts_detected,
                alert_rate,
                contamination,
                model_name,
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)


def save_alerts(
    alerts: pd.DataFrame,
    execution_id: int,
) -> int:
    """
    Guarda las alertas correspondientes a una ejecución.
    """
    initialize_database()

    if alerts.empty:
        return 0

    records = []

    for _, row in alerts.iterrows():
        records.append(
            (
                execution_id,
                str(row.get("transaction_id", "")),
                str(row.get("customer_id", "")),
                str(row.get("timestamp", "")),
                float(row.get("amount", 0)),
                str(row.get("country", "")),
                str(row.get("channel", "")),
                str(row.get("device", "")),
                str(row.get("transaction_type", "")),
                str(row.get("risk_level", "")),
                float(row.get("anomaly_score", 0)),
                str(row.get("model_risk_level", "")),
                str(row.get("model_reason", "")),
                "Pendiente",
            )
        )

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO alerts (
                execution_id,
                transaction_id,
                customer_id,
                timestamp,
                amount,
                country,
                channel,
                device,
                transaction_type,
                customer_risk,
                anomaly_score,
                model_risk_level,
                model_reason,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        connection.commit()

    return len(records)


def get_executions() -> pd.DataFrame:
    """
    Obtiene el historial de ejecuciones.
    """
    initialize_database()

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM executions
            ORDER BY execution_id DESC
            """,
            connection,
        )


def get_alerts(
    execution_id: int | None = None,
) -> pd.DataFrame:
    """
    Obtiene las alertas almacenadas.
    """
    initialize_database()

    with get_connection() as connection:
        if execution_id is None:
            query = """
                SELECT *
                FROM alerts
                ORDER BY anomaly_score DESC
            """

            return pd.read_sql_query(
                query,
                connection,
            )

        query = """
            SELECT *
            FROM alerts
            WHERE execution_id = ?
            ORDER BY anomaly_score DESC
        """

        return pd.read_sql_query(
            query,
            connection,
            params=(execution_id,),
        )


def clear_database() -> None:
    """
    Elimina los registros de prueba.
    """
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM alerts")
        connection.execute("DELETE FROM executions")
        connection.commit()


if __name__ == "__main__":
    initialize_database()
    print(f"Base de datos inicializada: {DATABASE_PATH}")