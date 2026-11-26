"""Barrido del exponente de difusividad m: PC, PE y exactitud.

Sirve para entender por qué los valores PC=0.3905 / PE=1.1849 del contexto
no coinciden con los de esta reproducción (dependen del espacio de
características y de m).
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, FIGURES, FUZZIFIER, METRICS
from fuzzy_model import (
    FuzzyCMeans, TSKClassifier, partition_coefficient, partition_entropy,
)


def main():
    d = np.load(DATA_PROCESSED / "ventanas_features.npz", allow_pickle=True)
    Xtr, Xte, ytr, yte = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    Ytr = d["Y_train"]
    prototipos = np.stack([Xtr[ytr == j].mean(axis=0) for j in range(5)])
    filas = []
    for m in [1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]:
        fcm = FuzzyCMeans(m=m).fit(Xtr, init_centers=prototipos)
        Utr = fcm.membership(Xtr)
        tsk = TSKClassifier().fit(Xtr, Ytr, Utr, m=m)
        acc = float(np.mean(tsk.predict(Xte, fcm.membership(Xte)) == yte))
        filas.append({"m": m, "PC": partition_coefficient(Utr),
                      "PE": partition_entropy(Utr), "exactitud_test": acc,
                      "iteraciones": fcm.n_iter})
        print(f"m={m:>4}  PC={filas[-1]['PC']:.4f}  PE={filas[-1]['PE']:.4f}  acc={acc:.2%}")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(METRICS / "barrido_m.csv", index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(tabla["m"], tabla["PC"], "o-"); axes[0].axhline(0.3905, ls="--", c="r", label="PC context")
    axes[0].set_title("Coeficiente de partición"); axes[0].legend()
    axes[1].plot(tabla["m"], tabla["PE"], "o-"); axes[1].axhline(1.1849, ls="--", c="r", label="PE contexto")
    axes[1].set_title("Entropía de partición"); axes[1].legend()
    axes[2].plot(tabla["m"], tabla["exactitud_test"] * 100, "o-")
    axes[2].set_title("Exactitud en prueba")
    for ax in axes:
        ax.set_xlabel("exponente m"); ax.grid(alpha=0.3)
    fig.tight_layout()
    temporal = FIGURES / f".tmp_{time.time_ns()}_08_barrido_m.png"
    for i in range(6):
        try:
            fig.savefig(temporal, dpi=130)
            temporal.replace(FIGURES / "08_barrido_m.png")
            break
        except OSError:
            time.sleep(0.3 * (i + 1))
    print("Guardado:", METRICS / "barrido_m.csv")


if __name__ == "__main__":
    main()
