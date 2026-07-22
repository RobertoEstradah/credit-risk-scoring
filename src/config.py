"""Configuración central del pipeline de scoring crediticio.

Dataset objetivo: Home Credit Default Risk (tabla principal application_train.csv).
Si el CSV no existe, el pipeline puede generar datos sintéticos con el mismo
esquema para desarrollo y tests (ver src/data.py).
"""
from pathlib import Path

# ---------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "application_train.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------- datos
TARGET = "TARGET"          # 1 = default, 0 = pagó
ID_COL = "SK_ID_CURR"

# Subconjunto curado de columnas de application_train (suficiente para un
# modelo sólido; se puede ampliar en F2 con agregaciones de tablas secundarias).
NUMERIC_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "REGION_POPULATION_RELATIVE",
]
# Home Credit usa 365243 ("~1000 años empleado") como sentinel de nulo en
# DAYS_EMPLOYED - típicamente pensionados/desempleados. Se limpia en data.py.
DAYS_EMPLOYED_SENTINEL = 365243

# Outlier real: AMT_INCOME_TOTAL llega a 117,000,000 (percentil 99.9 real es
# ~900,000). Cap fijo (no calculado del split train/test, para no filtrar
# estadísticos de test) por encima del cual casi seguro es error de captura,
# no un ingreso alto legítimo. Ver data.py::_cap_income_outliers.
AMT_INCOME_CAP = 1_000_000

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    # Alta cardinalidad (58 categorías reales); el OneHotEncoder de train.py
    # ya usa min_frequency=0.01 para agrupar categorías raras, así que no
    # requiere manejo especial. "XNA" (~18% de las filas) es el mismo grupo
    # de pensionados/desempleados que el sentinel DAYS_EMPLOYED==365243.
    "ORGANIZATION_TYPE",
]

# ---------------------------------------------------------------- modelado
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5

# ------------------------------------------------------- matriz de costos
# Unidades monetarias arbitrarias, ajustables al negocio:
#   FN (aprobar a quien hace default)  -> pérdida del principal esperado
#   FP (rechazar a buen cliente)       -> margen de interés no ganado
COST_FN = 1.0   # costo relativo de un default no detectado
COST_FP = 0.15  # costo relativo de rechazar a un buen cliente
