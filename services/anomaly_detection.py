"""
Isolation Forest Model
"""

"""
Modelo de detección de anomalías para transacciones financieras.

Este módulo prepara las variables, entrena un modelo Isolation Forest,
genera puntajes de anomalía, clasifica las operaciones y guarda tanto
el modelo entrenado como los resultados.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from services.database import save_alerts, save_execution
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CONTAMINATION,
    MAX_SAMPLES,
    MODEL_DIR,
    MODEL_PATH,
    N_ESTIMATORS,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


class AMLAnomalyDetector:
    """
    Encapsula el preprocesamiento, entrenamiento y evaluación
    del modelo Isolation Forest.
    """

    def __init__(
        self,
        contamination: float = CONTAMINATION,
        n_estimators: int = N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.numeric_features = [
            "amount",
            "transaction_hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_night",
            "amount_log",
            "customer_amount_mean",
            "customer_amount_median",
            "customer_amount_std",
            "amount_to_customer_mean",
            "amount_to_customer_median",
            "customer_daily_frequency",
            "is_high_risk_country",
            "is_crypto_channel",
            "customer_risk_numeric",
        ]

        self.categorical_features = [
            "country",
            "channel",
            "device",
            "transaction_type",
            "risk_level",
        ]

        self.pipeline: Pipeline | None = None
        self.results: pd.DataFrame | None = None

    def load_data(
        self,
        file_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Carga el dataset procesado.
        """
        input_file = (
            file_path
            or PROCESSED_DATA_DIR / "transactions_processed.csv"
        )

        if not input_file.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset procesado: {input_file}"
            )

        dataframe = pd.read_csv(
            input_file,
            parse_dates=["timestamp"],
        )

        return dataframe

    def validate_features(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Verifica que todas las variables requeridas existan.
        """
        required_features = (
            self.numeric_features
            + self.categorical_features
        )

        missing_features = [
            feature
            for feature in required_features
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Faltan variables requeridas para el modelo: "
                + ", ".join(missing_features)
            )

    def build_pipeline(self) -> Pipeline:
        """
        Construye el pipeline de transformación e Isolation Forest.
        """
        numeric_transformer = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                )
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_transformer,
                    self.numeric_features,
                ),
                (
                    "categorical",
                    categorical_transformer,
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=MAX_SAMPLES,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        return self.pipeline

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Selecciona las variables utilizadas por el modelo.
        """
        self.validate_features(dataframe)

        selected_features = (
            self.numeric_features
            + self.categorical_features
        )

        features = dataframe[selected_features].copy()

        return features

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> Pipeline:
        """
        Entrena el modelo Isolation Forest.
        """
        features = self.prepare_features(dataframe)

        if self.pipeline is None:
            self.build_pipeline()

        if self.pipeline is None:
            raise RuntimeError(
                "No fue posible construir el pipeline."
            )

        self.pipeline.fit(features)

        return self.pipeline

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Genera predicciones y puntajes de anomalía.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "El modelo debe entrenarse o cargarse antes de predecir."
            )

        features = self.prepare_features(dataframe)

        raw_predictions = self.pipeline.predict(features)

        raw_decision_scores = self.pipeline.decision_function(
            features
        )

        anomaly_scores = -raw_decision_scores

        results = dataframe.copy()

        results["model_prediction"] = (
            raw_predictions == -1
        ).astype(int)

        results["anomaly_score"] = anomaly_scores

        results["model_status"] = np.where(
            results["model_prediction"] == 1,
            "Anómala",
            "Normal",
        )

        results["model_risk_level"] = self.assign_risk_level(
            results
        )

        results["model_reason"] = results.apply(
            self.generate_model_reason,
            axis=1,
        )

        self.results = results

        return results

    def assign_risk_level(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Asigna riesgo bajo, medio o alto a las alertas detectadas.
        """
        risk_levels = pd.Series(
            "Normal",
            index=dataframe.index,
            dtype="object",
        )

        anomaly_mask = (
            dataframe["model_prediction"] == 1
        )

        anomaly_scores = dataframe.loc[
            anomaly_mask,
            "anomaly_score",
        ]

        if anomaly_scores.empty:
            return risk_levels

        medium_threshold = anomaly_scores.quantile(0.60)
        high_threshold = anomaly_scores.quantile(0.85)

        risk_levels.loc[
            anomaly_mask
            & (
                dataframe["anomaly_score"]
                < medium_threshold
            )
        ] = "Bajo"

        risk_levels.loc[
            anomaly_mask
            & (
                dataframe["anomaly_score"]
                >= medium_threshold
            )
            & (
                dataframe["anomaly_score"]
                < high_threshold
            )
        ] = "Medio"

        risk_levels.loc[
            anomaly_mask
            & (
                dataframe["anomaly_score"]
                >= high_threshold
            )
        ] = "Alto"

        return risk_levels

    def generate_model_reason(
        self,
        row: pd.Series,
    ) -> str:
        """
        Genera una explicación basada en variables observables.

        Esta explicación no representa una interpretación interna exacta
        de Isolation Forest, sino una descripción de señales presentes
        en la transacción detectada.
        """
        if row["model_prediction"] == 0:
            return "Operación dentro del comportamiento esperado"

        reasons = []

        if row["amount_to_customer_mean"] >= 3:
            reasons.append(
                "Monto muy superior al promedio del cliente"
            )

        if row["is_high_risk_country"] == 1:
            reasons.append(
                "Operación vinculada con país de mayor riesgo"
            )

        if row["is_crypto_channel"] == 1:
            reasons.append(
                "Operación realizada mediante criptomonedas"
            )

        if row["is_night"] == 1:
            reasons.append(
                "Operación realizada en horario nocturno"
            )

        if row["customer_daily_frequency"] >= 4:
            reasons.append(
                "Frecuencia diaria de operaciones elevada"
            )

        if row["customer_risk_numeric"] == 2:
            reasons.append(
                "Cliente con perfil de riesgo alto"
            )

        if not reasons:
            reasons.append(
                "Combinación atípica de características transaccionales"
            )

        return "; ".join(reasons)

    def evaluate(
        self,
        results: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Evalúa las predicciones usando la etiqueta sintética.

        La columna is_suspicious solo se emplea como referencia
        experimental, no como variable de entrenamiento.
        """
        if "is_suspicious" not in results.columns:
            raise ValueError(
                "El dataset no contiene la etiqueta sintética "
                "'is_suspicious'."
            )

        y_true = results["is_suspicious"].astype(int)
        y_pred = results["model_prediction"].astype(int)

        metrics = {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }

        return metrics

    def save_model(
        self,
        file_path: Path = MODEL_PATH,
    ) -> Path:
        """
        Guarda el pipeline entrenado.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "No existe un modelo entrenado para guardar."
            )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = {
            "pipeline": self.pipeline,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }

        joblib.dump(
            artifact,
            file_path,
        )

        return file_path

    def load_model(
        self,
        file_path: Path = MODEL_PATH,
    ) -> Pipeline:
        """
        Carga un modelo previamente entrenado.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo: {file_path}"
            )

        artifact = joblib.load(file_path)

        self.pipeline = artifact["pipeline"]
        self.numeric_features = artifact[
            "numeric_features"
        ]
        self.categorical_features = artifact[
            "categorical_features"
        ]
        self.contamination = artifact[
            "contamination"
        ]
        self.n_estimators = artifact[
            "n_estimators"
        ]
        self.random_state = artifact[
            "random_state"
        ]

        return self.pipeline

    def save_results(
        self,
        results: pd.DataFrame,
    ) -> tuple[Path, Path]:
        """
        Guarda todas las predicciones y las alertas detectadas.
        """
        OUTPUT_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions_file = (
            OUTPUT_DATA_DIR
            / "transactions_scored.csv"
        )

        alerts_file = (
            OUTPUT_DATA_DIR
            / "alerts_detected.csv"
        )

        results.to_csv(
            predictions_file,
            index=False,
            encoding="utf-8-sig",
        )

        alerts = results[
            results["model_prediction"] == 1
        ].copy()

        alerts = alerts.sort_values(
            by="anomaly_score",
            ascending=False,
        )

        alerts.to_csv(
            alerts_file,
            index=False,
            encoding="utf-8-sig",
        )

        return predictions_file, alerts_file


def main() -> None:
    """
    Entrena, evalúa y guarda el modelo.
    """
    detector = AMLAnomalyDetector()

    dataframe = detector.load_data()

    detector.train(dataframe)

    results = detector.predict(dataframe)

    metrics = detector.evaluate(results)

    model_file = detector.save_model()

    predictions_file, alerts_file = detector.save_results(results)
        
    detected_alerts = int(
        results["model_prediction"].sum()
    )

    alert_rate = (
        detected_alerts / len(results) * 100
        if len(results) > 0
        else 0
    )

    alerts_dataframe = results[
        results["model_prediction"] == 1
    ].copy()

    execution_id = save_execution(
        source_name="transactions_processed.csv",
        total_transactions=len(results),
        alerts_detected=detected_alerts,
        alert_rate=alert_rate,
        contamination=detector.contamination,
        model_name="Isolation Forest",
    )

    saved_alerts = save_alerts(
        alerts_dataframe,
        execution_id,
    )
            

    print("=" * 70)
    print("MODELO ISOLATION FOREST ENTRENADO CORRECTAMENTE")
    print("=" * 70)
    print(f"Total de transacciones: {len(results):,}")
    print(f"Alertas detectadas: {detected_alerts:,}")
    print(f"Tasa de alertas: {alert_rate:.2f}%")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print()
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print()
    print("Reporte de clasificación:")
    print(metrics["classification_report"])
    print(f"Modelo guardado en: {model_file}")
    print(f"Predicciones guardadas en: {predictions_file}")
    print(f"Alertas guardadas en: {alerts_file}")
    print("=" * 70)
    print(f"Ejecución registrada con ID: {execution_id}")
    print(f"Alertas guardadas en SQLite: {saved_alerts:,}")
    
    
    preview_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "country",
        "channel",
        "anomaly_score",
        "model_status",
        "model_risk_level",
        "model_reason",
    ]

    print(
        results[
            results["model_prediction"] == 1
        ][preview_columns]
        .sort_values(
            by="anomaly_score",
            ascending=False,
        )
        .head(10)
    )


if __name__ == "__main__":
    main()