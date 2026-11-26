"""Configuración central del proyecto: un solo lugar para cambiar parámetros."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "dataset_completo.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
DOCS = ROOT / "docs"

# --- Adquisición (Myo Armband, dongle BLED112) ---
FS = 200
CHANNELS = [f"CH{i}" for i in range(1, 9)]
CLASSES = ["reposo", "mano_abierta", "puño_cerrado", "pinza", "agarre_vaso"]

# --- Filtrado (idéntico al pipeline original) ---
BAND_LOW = 20.0
BAND_HIGH = 99.0     # Nyquist = 100 Hz: el "450 Hz" del script original se recorta a 99 Hz
NOTCH_Q = 35.0
NOTCHES = (50.0, 60.0)
FILTER_ORDER = 4

# --- Ventaneo y características ---
WINDOW = 150         # 0.75 s a 200 Hz
HOP = 50             # 0.25 s (el script original en tiempo real usa HOP=20)
TRIM = 0             # 0 -> 2.139 ventanas (documento); 50 -> 2.125 (script original)
ZC_THRESHOLD = 0.0
TRAIN_FRACTION = 0.70
SEED = 13

# --- Modelo neuro-difuso ---
N_CLUSTERS = 5
FUZZIFIER = 2.0
MAX_ITER = 200
TOL = 1e-5
RIDGE = 1e-6

# --- Tiempo real / servo (adapatado de realtime_servos.py) ---
SMOOTH_K = 7
MOVE_COOLDOWN = 0.7
RAMP_STEP = 8
RAMP_DT = 0.005
SERVO_CHANNELS = [0, 1, 2, 3, 4]
PULSE_MIN_US = 500
PULSE_MAX_US = 2500
POSES = {
    "reposo":       [50, 50, 50, 50, 50],
    "mano_abierta": [10, 10, 10, 10, 10],
    "puño_cerrado": [150, 150, 150, 150, 150],
    "pinza":        [10, 10, 10, 150, 150],
    "agarre_vaso":  [120, 120, 120, 120, 130],
}

FEATURE_NAMES = [
    f"{ch}_{feat}"
    for ch in CHANNELS
    for feat in ("RMS", "MAV", "WL", "ZC")
]
