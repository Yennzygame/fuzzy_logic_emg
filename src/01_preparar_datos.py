"""Fase 1: de las señales crudas a una tabla de 32 características por ventana."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    CHANNELS, CLASSES, DATA_PROCESSED, DATA_RAW, FEATURE_NAMES, FS, HOP,
    SEED, TRAIN_FRACTION, TRIM, WINDOW, ZC_THRESHOLD,
)
from signal_features import (
    build_filters, filter_emg, hudgins_features, normalize,
    segment_blocks, stratified_split,
)

ALIASES = {
    "puno_cerrado": "puño_cerrado", "puño cerrado": "puño_cerrado",
    "mano abierta": "mano_abierta", "agarre vaso": "agarre_vaso",
    "agarre de vaso": "agarre_vaso",
}


def main():
    print("[1/5] Leyendo dataset crudo:", DATA_RAW)
    df = pd.read_csv(DATA_RAW)
    missing = [c for c in CHANNELS + ["G"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    labels = df["G"].astype(str).str.strip().str.lower().replace(ALIASES)
    unknown = set(labels.unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Etiquetas desconocidas: {unknown}")
    raw_counts = Counter(labels)

    print("[2/5] Filtrando (pasabanda 20-99 Hz + notch 50/60 Hz)…")
    Xf = filter_emg(df[CHANNELS].to_numpy(dtype=np.float64), build_filters(FS))

    print(f"[3/5] Ventaneando (W={WINDOW}, HOP={HOP}, TRIM={TRIM})…")
    Xw, y_str, groups = segment_blocks(Xf, labels.to_numpy(), WINDOW, HOP, TRIM)
    print("      ventanas por gesto:", dict(Counter(y_str)))

    print("[4/5] Hudgins + z-score y split estratificado 70/30…")
    X = hudgins_features(Xw, zc_threshold=ZC_THRESHOLD)
    y = np.asarray([CLASSES.index(v) for v in y_str], dtype=np.int64)
    Y = np.eye(len(CLASSES), dtype=np.float32)[y]
    tr, te = stratified_split(y, TRAIN_FRACTION, SEED)
    (Xtr, Xte), mu, sigma = normalize(X[tr], X[te])

    print("[5/5] Guardando artefactos…")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PROCESSED / "ventanas_features.npz",
        X_train=Xtr, X_test=Xte, y_train=y[tr], y_test=y[te], Y_train=Y[tr],
        groups_train=groups[tr], groups_test=groups[te],
        mu=mu, sigma=sigma, feature_names=np.array(FEATURE_NAMES),
        classes=np.array(CLASSES),
    )

    split = np.array(["test"] * len(y), dtype=object)
    split[tr] = "train"
    tabla = pd.DataFrame(X, columns=FEATURE_NAMES)
    tabla.insert(0, "grupo_bloque", groups)
    tabla.insert(0, "split", split)
    tabla.insert(0, "gesto", np.array(CLASSES)[y])
    tabla.to_csv(DATA_PROCESSED / "ventanas_features.csv", index=False)

    resumen = {
        "dataset_crudo": {
            "archivo": DATA_RAW.name, "muestras": int(len(df)),
            "canales": CHANNELS, "frecuencia_hz": FS,
            "muestras_por_gesto": {k: int(v) for k, v in raw_counts.items()},
        },
        "ventaneo": {
            "window": WINDOW, "hop": HOP, "trim": TRIM,
            "total_ventanas": int(len(y)),
            "ventanas_por_gesto": {k: int(v) for k, v in Counter(y_str).items()},
            "num_bloques": int(len(np.unique(groups))),
            "caracteristicas": len(FEATURE_NAMES), "zc_threshold": ZC_THRESHOLD,
        },
        "split": {
            "estrategia": "estratificado aleatorio por ventanas (70/30)",
            "train": int(len(tr)), "test": int(len(te)), "semilla": SEED,
        },
        "advertencia": (
            "El split aleatorio por ventanas mezcla ventanas solapadas del mismo "
            "bloque entre train y test (fuga de información). Reproduce el "
            "protocolo reportado, pero no sirve para reportar desempeño clínico."
        ),
    }
    (DATA_PROCESSED / "resumen_dataset.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Listo ->", DATA_PROCESSED)


if __name__ == "__main__":
    main()
