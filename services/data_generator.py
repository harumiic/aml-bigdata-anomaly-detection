"""
Synthetic Dataset Generator
"""
from faker import Faker
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta

from config import RAW_DATA_DIR, DATASET_SIZE, RANDOM_STATE, ANOMALY_PERCENTAGE

# configuracion

fake = Faker("es_ES")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# listas base

COUNTRIES = [
    "Perú",
    "Chile",
    "Colombia",
    "Brasil",
    "México",
    "Estados Unidos",
    "España",
    "Panamá"
]

CHANNELS = [
    "Transferencia",
    "Depósito",
    "Retiro",
    "Pago Online",
    "POS",
    "Criptomonedas"
]

DEVICES = [
    "Mobile",
    "Desktop",
    "ATM"
]

TRANSACTION_TYPES = [
    "Transferencia",
    "Pago",
    "Compra",
    "Retiro"
]

RISK_LEVELS = [
    "Low",
    "Medium",
    "High"
]


def generate_transaction_id(index):
    return f"TX{index:08d}"


def generate_customer_id():
    return f"CUS{random.randint(1000,9999)}"


def generate_account():
    return f"ACC{random.randint(10000000,99999999)}"


def generate_amount():
    return round(np.random.gamma(2.5, 250), 2)


def generate_timestamp():
    start = datetime(2025,1,1)
    end = datetime(2025,12,31)

    delta = end - start

    random_days = random.randint(0, delta.days)

    random_seconds = random.randint(0,86399)

    return start + timedelta(days=random_days,seconds=random_seconds)


def generate_transaction(index):

    transaction = {

        "transaction_id": generate_transaction_id(index),

        "customer_id": generate_customer_id(),

        "account_id": generate_account(),

        "timestamp": generate_timestamp(),

        "amount": generate_amount(),

        "country": random.choice(COUNTRIES),

        "channel": random.choice(CHANNELS),

        "device": random.choice(DEVICES),

        "transaction_type": random.choice(TRANSACTION_TYPES),

        "risk_level": random.choice(RISK_LEVELS)

    }

    return transaction


def create_dataset():

    transactions = []

    for i in range(DATASET_SIZE):

        transactions.append(generate_transaction(i+1))

    df = pd.DataFrame(transactions)

    return df


def save_dataset(df):

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RAW_DATA_DIR / "transactions.csv"

    df.to_csv(output_file,index=False)

    print("===================================")
    print("Dataset generado correctamente")
    print(output_file)
    print(df.head())
    
    
    
if __name__ == "__main__":

    dataset = create_dataset()

    save_dataset(dataset)