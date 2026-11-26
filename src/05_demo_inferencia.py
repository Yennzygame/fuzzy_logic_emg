"""Demo didáctica: muestra paso a paso una inferencia TSK para una ventana."""
from __future__ import annotations

import json

import numpy as np

from config import CLASSES, DATA_PROCESSED, FEATURE_NAMES, MODELS
from fuzzy_model import partition_coefficient, partition_entropy


def main():
    d = np.load(DATA_PROCESSED / "ventanas_features.npz", allow_pickle=True)
    m = json.loads((MODELS / "modelo_fcm_tsk.json").read_text(encoding="utf-8"))
    X, y = d["X_test"], d["y_test"]
    centros, P, q = map(lambda k: np.asarray(m[k], float), ("centros", "P", "q"))
    mm = float(m["m"])

    i = 0
    x = X[i]
    print(f"Ventana de prueba #{i} | gesto real: {CLASSES[y[i]]}")
    print("\n1) Vector normalizado de 32 características (primeras 8):")
    for nombre, valor in list(zip(FEATURE_NAMES, x))[:8]:
        print(f"   {nombre:<10} {valor:+.3f}")

    dist = np.sum((x[None, :] - centros) ** 2, axis=1)
    w = 1 / dist ** (1 / (mm - 1))
    w = w / w.sum()
    print("\n2) Grados de pertenencia w_i (SUMAN 1):")
    for k, wi in enumerate(w):
        print(f"   regla R{k}: {wi:.4f}")

    print("\n3) Consecuente local f_i(x) = P_i·x + q_i (puntaje por gesto):")
    locales = []
    for k in range(len(P)):
        f = P[k] @ x + q[k]
        locales.append(f)
        print(f"   R{k}: " + "  ".join(f"{CLASSES[j]}={f[j]:+.3f}" for j in range(5)))

    aportes = w[:, None] * np.asarray(locales)
    final = aportes.sum(axis=0) / w.sum()
    print("\n4) Defuzzificación: suma w_i·f_i / suma w_i")
    for j, c in enumerate(CLASSES):
        print(f"   {c:<14} {final[j]:+.4f}")
    print(f"\n=> Gesto predicho: {CLASSES[int(np.argmax(final))]}")

    print(f"\nPC de esta ventana: {partition_coefficient(w):.4f} | PE: {partition_entropy(w):.4f}")


if __name__ == "__main__":
    main()
