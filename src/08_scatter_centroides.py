"""Proyecciones 2D de las ventanas y de los centroides FCM.

Los centroides tienen 32 dimensiones, por lo que no pueden dibujarse
directamente. Se proyectan con la misma transformación que las muestras:

  * PCA: no supervisada, coherente con el criterio usado por FCM.
  * LDA: supervisada, incluida solo como referencia visual de separación.

Genera:
  outputs/figures/10_scatter_centroides.png
  outputs/metrics/proyeccion_centroides.csv
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import CLASSES, DATA_PROCESSED, FIGURES, METRICS, MODELS


def guardar(fig, nombre, intentos=6):
    temporal = FIGURES / f".tmp_{time.time_ns()}_{nombre}"
    for i in range(intentos):
        try:
            fig.savefig(temporal, dpi=150)
            plt.close(fig)
            temporal.replace(FIGURES / nombre)
            return
        except OSError:
            time.sleep(0.3 * (i + 1))
    raise RuntimeError(f"No se pudo guardar {nombre}")


def centros_unicos(centros, tol=1.0):
    """Agrupa centroides casi coincidentes para contar posiciones reales."""
    representantes = []
    for centro in centros:
        if not any(np.linalg.norm(centro - r) < tol for r in representantes):
            representantes.append(centro)
    return representantes


def main():
    d = np.load(DATA_PROCESSED / "ventanas_features.npz", allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    modelo = json.loads((MODELS / "modelo_fcm_tsk.json").read_text(encoding="utf-8"))
    centros = np.asarray(modelo["centros"], dtype=float)

    n_unicos = len(centros_unicos(centros))

    pca = PCA(n_components=2).fit(Xtr)
    Z_pca, C_pca = pca.transform(Xtr), pca.transform(centros)

    lda = LDA(n_components=2).fit(Xtr, ytr)
    Z_lda, C_lda = lda.transform(Xtr), lda.transform(centros)

    colores = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    vistas = (
        (axes[0], Z_pca, C_pca,
         f"PCA (no supervisada)\n{100 * pca.explained_variance_ratio_.sum():.1f} % de varianza explicada",
         "CP1", "CP2"),
        (axes[1], Z_lda, C_lda,
         "LDA (supervisada, referencia visual)",
         "LD1", "LD2"),
    )

    for ax, Z, C, titulo, ex, ey in vistas:
        for k, gesto in enumerate(CLASSES):
            mask = ytr == k
            ax.scatter(Z[mask, 0], Z[mask, 1], s=9, alpha=0.30,
                       color=colores[k], edgecolors="none", label=gesto)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=230, c="black",
                   edgecolors="white", linewidths=1.2, zorder=5,
                   label="Centroides FCM")
        for i, (cx, cy) in enumerate(C):
            ax.annotate(f"R{i}", (cx, cy), xytext=(7, 7),
                        textcoords="offset points", fontsize=9,
                        fontweight="bold", color="black", zorder=6)
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel(ex)
        ax.set_ylabel(ey)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle(
        f"Proyección de {len(Xtr):,} ventanas de entrenamiento y {len(centros)} "
        f"centroides FCM ({n_unicos} posiciones distintas)",
        fontsize=13,
    )
    fig.tight_layout()
    guardar(fig, "10_scatter_centroides.png")

    filas = []
    for i in range(len(centros)):
        filas.append({
            "regla": f"R{i}",
            "pca_1": C_pca[i, 0], "pca_2": C_pca[i, 1],
            "lda_1": C_lda[i, 0], "lda_2": C_lda[i, 1],
            "coincide_con": (
                next((f"R{j}" for j in range(i) if np.linalg.norm(centros[i] - centros[j]) < 1.0), "")
            ),
        })
    pd.DataFrame(filas).to_csv(
        METRICS / "proyeccion_centroides.csv", index=False, float_format="%.6f"
    )

    print("Centroides proyectados (PCA):")
    for r in filas:
        print(f"  {r['regla']}: ({r['pca_1']:+.2f}, {r['pca_2']:+.2f})"
              + (f"  = {r['coincide_con']}" if r["coincide_con"] else ""))
    print("Guardado:", FIGURES / "10_scatter_centroides.png")
    print("Guardado:", METRICS / "proyeccion_centroides.csv")


if __name__ == "__main__":
    main()
