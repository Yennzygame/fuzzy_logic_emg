"""Fases 2 y 3: antecedentes por Fuzzy C-Means y consecuentes TSK por WLS."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from config import DATA_PROCESSED, FEATURE_NAMES, FUZZIFIER, MODELS, N_CLUSTERS, OUTPUTS
from fuzzy_model import FuzzyCMeans, TSKClassifier, partition_coefficient, partition_entropy


def main():
    d = np.load(DATA_PROCESSED / "ventanas_features.npz", allow_pickle=True)
    Xtr, Xte, Ytr = d["X_train"], d["X_test"], d["Y_train"]
    ytr, yte = d["y_train"], d["y_test"]
    classes = [str(c) for c in d["classes"]]
    mu, sigma = d["mu"], d["sigma"]

    print("[1/4] Ajustando Fuzzy C-Means (c=5, m=2, prototipos por clase)…")
    prototipos = np.stack([Xtr[ytr == j].mean(axis=0) for j in range(N_CLUSTERS)])
    fcm = FuzzyCMeans(c=N_CLUSTERS, m=FUZZIFIER).fit(Xtr, init_centers=prototipos)
    Utr = fcm.membership(Xtr)
    pc, pe = partition_coefficient(Utr), partition_entropy(Utr)
    print(f"      iteraciones={fcm.n_iter}  PC={pc:.4f}  PE={pe:.4f}")

    print("[2/4] Estimando consecuentes por mínimos cuadrados ponderados…")
    tsk = TSKClassifier().fit(Xtr, Ytr, Utr, m=FUZZIFIER)
    acc_tr = float(np.mean(tsk.predict(Xtr, fcm.membership(Xtr)) == ytr))
    acc_te = float(np.mean(tsk.predict(Xte, fcm.membership(Xte)) == yte))
    print(f"      exactitud train={acc_tr:.2%}  test={acc_te:.2%}")

    print("[3/4] Guardando modelo y reglas auditables…")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    modelo = {
        "tipo": "FCM + TSK orden 1 (5 reglas, salida one-hot de 5 gestos)",
        "clases": classes, "c": N_CLUSTERS, "m": FUZZIFIER,
        "inicializacion": "medias por clase (una regla por gesto base)",
        "centros": fcm.centers.tolist(), "P": tsk.P.tolist(), "q": tsk.q.tolist(),
        "mu_zscore": mu.tolist(), "sigma_zscore": sigma.tolist(),
        "caracteristicas": [str(f) for f in FEATURE_NAMES],
        "validacion_fcm": {
            "PC": pc, "PE": pe, "iteraciones": fcm.n_iter,
            "funcion_objetivo": fcm.objective,
        },
        "exactitud_train": acc_tr, "exactitud_test": acc_te,
    }
    (MODELS / "modelo_fcm_tsk.json").write_text(
        json.dumps(modelo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    reglas = []
    for i in range(N_CLUSTERS):
        dominante = int(np.argmax(np.abs(tsk.q[i])))
        aportes = np.mean(np.abs(tsk.P[i]), axis=0)
        top = np.argsort(aportes)[::-1][:5]
        reglas.append({
            "regla": i, "gesto_inicial": classes[i],
            "gesto_dominante_q": classes[dominante],
            "intercepto_q": [round(float(v), 4) for v in tsk.q[i]],
            "top_5_caracteristicas": [
                {"caracteristica": FEATURE_NAMES[j], "peso_abs": round(float(aportes[j]), 4)}
                for j in top
            ],
            "peso_ponderado": round(float(np.sum(fcm.U[:, i] ** FUZZIFIER)), 2),
        })
    (MODELS / "reglas_resumen.json").write_text(
        json.dumps(reglas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    importancia = np.mean(np.abs(tsk.P), axis=(0, 1))
    with open(MODELS / "importancia_canales.csv", "w", encoding="utf-8") as f:
        f.write("caracteristica,importancia_media_abs\n")
        for j in np.argsort(importancia)[::-1]:
            f.write(f"{FEATURE_NAMES[j]},{importancia[j]:.6f}\n")

    print("[4/4] Modelo guardado en", MODELS)
    print("     Cluster dominante por muestra:",
          dict(Counter(np.asarray(classes)[np.argmax(Utr, axis=1)])))


if __name__ == "__main__":
    main()
