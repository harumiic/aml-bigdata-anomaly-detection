from pathlib import Path

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DATA_DIR = DATA_DIR / "outputs"

MODEL_DIR = BASE_DIR / "models"

DATABASE_DIR = BASE_DIR / "database"

DATABASE_PATH = DATABASE_DIR / "aml.db"

MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

# =====================================
# DATASET
# =====================================

DATASET_SIZE = 100000

RANDOM_STATE = 42

ANOMALY_PERCENTAGE = 0.03

# =====================================
# MODEL
# =====================================

CONTAMINATION = 0.065

N_ESTIMATORS = 250

MAX_SAMPLES = "auto"

# =====================================
# AML CONFIGURATION
# =====================================

HIGH_RISK_COUNTRIES = [
    "Panamá",
    "Islas Caimán",
    "Belice",
    "Venezuela"
]

HIGH_RISK_CHANNELS = [
    "Criptomonedas",
    "Transferencia"
]

MAX_NORMAL_AMOUNT = 5000

MAX_SUSPICIOUS_AMOUNT = 50000