"""Fase 4: evaluación, comparación, interpretabilidad y generación de informe."""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

from config import CLASSES, DATA_PROCESSED, DOCS, FIGURES, FEATURE_NAMES, METRICS, MODELS
from fuzzy_model import TSKClassifier


def guardar(fig, nombre, intentos=6):
    """Guarda de forma atómica (evita bloqueos temporales de archivos en Windows)."""
    temporal = FIGURES / f".tmp_{time.time_ns()}_{nombre}"
    for i in range(intentos):
        try:
            fig.savefig(temporal, dpi=130)
            plt.close(fig)
            temporal.replace(FIGURES / nombre)
            return
        except OSError:
            time.sleep(0.3 * (i + 1))
    raise RuntimeError(f"No se pudo guardar {nombre}")


def load_model():
    return json.loads((MODELS / "modelo_fcm_tsk.json").read_text(encoding="utf-8"))


def pertenencia_fcm(X, modelo):
    centros = np.asarray(modelo["centros"], dtype=np.float64)
    m = float(modelo["m"])
    d = np.maximum(np.sum((X[:, None, :] - centros) ** 2, axis=2), 1e-12)
    u = 1.0 / d ** (1.0 / (m - 1.0))
    return u / u.sum(axis=1, keepdims=True)


def pertenencia_gaussiana(X, U, m):
    w = U**m
    tot = np.maximum(w.sum(axis=0)[:, None], 1e-12)
    means = (w.T @ X) / tot
    var = np.maximum((w.T @ (X**2)) / tot - means**2, 1e-6)
    g = np.exp(-0.5 * np.sum((X[:, None, :] - means) ** 2 / var, axis=2))
    return g / g.sum(axis=1, keepdims=True)


def scores_tsk(X, W, modelo):
    P, q = np.asarray(modelo["P"]), np.asarray(modelo["q"])
    num = np.zeros((len(X), P.shape[1]))
    for i in range(len(P)):
        num += W[:, i, None] * (X @ P[i].T + q[i])
    return num / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)


def medir_latencia(func, X, repeticiones=200):
    """Latencia por muestra (ms) en Python/NumPy; no comparable con C embebido."""
    for x in X[:10]:
        func(np.atleast_2d(x))
    tiempos = []
    for _ in range(repeticiones):
        i = np.random.randint(len(X))
        x = np.atleast_2d(X[i])
        t0 = time.perf_counter_ns()
        func(x)
        tiempos.append((time.perf_counter_ns() - t0) / 1e6)
    return float(np.median(tiempos))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    np.random.seed(13)
    d = np.load(DATA_PROCESSED / "ventanas_features.npz", allow_pickle=True)
    Xtr, Xte, ytr, yte = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    resumen = json.loads((DATA_PROCESSED / "resumen_dataset.json").read_text(encoding="utf-8"))
    modelo = load_model()

    print("[1/6] Evaluando FCM + TSK…")
    Wtr, Wte = pertenencia_fcm(Xtr, modelo), pertenencia_fcm(Xte, modelo)
    Wtr_g = pertenencia_gaussiana(Xtr, Wtr, modelo["m"])
    Wte_g = pertenencia_gaussiana(Xte, pertenencia_fcm(Xte, modelo), modelo["m"])
    pred_tsk = np.argmax(scores_tsk(Xte, Wte, modelo), axis=1)
    pred_g = np.argmax(scores_tsk(Xte, Wte_g, modelo), axis=1)

    # Ablación: FCM converge a 3 centros útiles -> 5 gaussianas ancladas a cada gesto
    centros_fcm = np.asarray(modelo["centros"], dtype=float)
    representantes = []
    for centro in centros_fcm:
        if not any(np.linalg.norm(centro - r) < 1.0 for r in representantes):
            representantes.append(centro)
    n_unicos = len(representantes)
    prototipos = np.stack([Xtr[ytr == j].mean(axis=0) for j in range(len(CLASSES))])
    varianzas = np.maximum(
        np.stack([Xtr[ytr == j].var(axis=0) for j in range(len(CLASSES))]), 0.2)

    def gaussianas_por_gesto(X):
        z = -0.5 * np.sum((X[:, None, :] - prototipos) ** 2 / varianzas, axis=2)
        z -= z.max(axis=1, keepdims=True)
        g = np.exp(z)
        return g / g.sum(axis=1, keepdims=True)

    Utr_gc, Ute_gc = gaussianas_por_gesto(Xtr), gaussianas_por_gesto(Xte)
    tsk_gc = TSKClassifier().fit(Xtr, d["Y_train"], Utr_gc, m=1.0)
    pred_gc = tsk_gc.predict(Xte, Ute_gc)
    (MODELS / "modelo_gaussiano_clases.json").write_text(json.dumps({
        "tipo": "5 reglas gaussianas ancladas a cada gesto + TSK orden 1",
        "clases": CLASSES, "c": 5, "m": 1.0,
        "centros": prototipos.tolist(), "P": tsk_gc.P.tolist(), "q": tsk_gc.q.tolist(),
        "mu_zscore": d["mu"].tolist(), "sigma_zscore": d["sigma"].tolist(),
        "caracteristicas": [str(f) for f in FEATURE_NAMES],
        "centros_fcm_unicos": n_unicos,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    fcm_tsk = lambda x: np.argmax(scores_tsk(x, pertenencia_fcm(x, modelo), modelo), axis=1)[0]
    fcm_g = lambda x: np.argmax(
        scores_tsk(x, pertenencia_gaussiana(x, pertenencia_fcm(x, modelo), modelo["m"]), modelo), axis=1)[0]

    print("[2/6] Entrenando líneas base (k-means rígido, KNN, Random Forest)…")
    km = KMeans(n_clusters=5, n_init=10, random_state=13).fit(Xtr)
    centroides = km.cluster_centers_
    etiquetas_cluster = np.asarray([
        np.bincount(ytr[km.labels_ == i], minlength=5).argmax() for i in range(5)
    ])
    pred_km = etiquetas_cluster[np.argmin(
        np.sum((Xte[:, None, :] - centroides) ** 2, axis=2), axis=1)]
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    pred_knn = knn.predict(Xte)
    rf = RandomForestClassifier(n_estimators=200, random_state=13, n_jobs=-1).fit(Xtr, ytr)
    pred_rf = rf.predict(Xte)

    modelos = {
        "FCM + TSK (pertenencia FCM)": (pred_tsk, fcm_tsk),
        "FCM + TSK (pertenencia Gaussiana)": (pred_g, fcm_g),
        "5 gaussianas por gesto + TSK": (pred_gc, lambda x: np.argmax(
            tsk_gc.scores(x, gaussianas_por_gesto(x)), axis=1)[0]),
        "K-means rígido + regla de mayoría": (pred_km, lambda x: etiquetas_cluster[
            np.argmin(np.sum((x - centroides) ** 2, axis=1))]),
        "KNN (k=5)": (pred_knn, lambda x: knn.predict(x)[0]),
        "Random Forest (200 árboles)": (pred_rf, lambda x: rf.predict(x)[0]),
    }

    print("[3/6] Calculando métricas y latencias…")
    filas = []
    for nombre, (pred, fn) in modelos.items():
        filas.append({
            "modelo": nombre,
            "exactitud": accuracy_score(yte, pred),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
            "latencia_ms": medir_latencia(fn, Xte),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(METRICS / "comparacion_modelos.csv", index=False, float_format="%.6f")

    reporte = classification_report(
        yte, pred_tsk, target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(reporte).T.to_csv(METRICS / "reporte_clasificacion.csv")
    cm = confusion_matrix(yte, pred_tsk)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(METRICS / "matriz_confusion.csv")

    print("[4/6] Generando figuras…")
    raw = resumen["dataset_crudo"]["muestras_por_gesto"]
    win = resumen["ventaneo"]["ventanas_por_gesto"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(CLASSES))
    ax.bar(x - 0.2, [raw.get(c, 0) for c in CLASSES], 0.4, label="muestras crudas")
    ax.bar(x + 0.2, [win.get(c, 0) for c in CLASSES], 0.4, label="ventanas (W=150, HOP=50)")
    ax.set_xticks(x, CLASSES, rotation=20, ha="right")
    ax.set_ylabel("cantidad"); ax.legend(); ax.set_title("Distribución de clases")
    fig.tight_layout(); guardar(fig, "01_distribucion_clases.png");

    from config import DATA_RAW
    crudo = pd.read_csv(DATA_RAW, nrows=1200)
    from signal_features import build_filters, filter_emg
    filtrado = filter_emg(crudo[[f"CH{i}" for i in range(1, 9)]].to_numpy(float), build_filters())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(crudo["CH1"].values, alpha=0.6, label="CH1 crudo")
    ax.plot(filtrado[:, 0], label="CH1 filtrado (20-99 Hz + notch)")
    ax.set_xlabel("muestras (5 s a 200 Hz)"); ax.legend(); ax.set_title("Preprocesamiento de la señal")
    fig.tight_layout(); guardar(fig, "02_senal_filtrada.png");

    matriz_pertenencia = np.zeros((5, 5))
    for i, c in enumerate(CLASSES):
        matriz_pertenencia[i] = Wte[yte == i].mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz_pertenencia, cmap="viridis")
    ax.set_xticks(range(5), [f"C{i}" for i in range(5)])
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("cluster difuso"); ax.set_title("Pertenencia media por gesto real")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{matriz_pertenencia[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    guardar(fig, "03_pertenencias_fcm.png");

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.set_title("Matriz de confusión — FCM + TSK")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    guardar(fig, "04_matriz_confusion.png");

    imp = pd.read_csv(MODELS / "importancia_canales.csv")
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["caracteristica"], top["importancia_media_abs"])
    ax.set_xlabel("media de |P| en las 5 reglas"); ax.set_title("Interpretabilidad: características más influyentes")
    fig.tight_layout(); guardar(fig, "05_importancia_canales.png");

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in tabla.iterrows():
        ax.scatter(r["latencia_ms"], r["exactitud"] * 100, s=110)
        ax.annotate(r["modelo"], (r["latencia_ms"], r["exactitud"] * 100),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("latencia por inferencia (ms, escala log)")
    ax.set_ylabel("exactitud (%)"); ax.set_title("Frontera exactitud / latencia")
    ax.grid(alpha=0.3); fig.tight_layout()
    guardar(fig, "06_pareto.png");

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(modelo["validacion_fcm"]["funcion_objetivo"])
    ax.set_xlabel("iteración"); ax.set_ylabel("función objetivo FCM")
    ax.set_title("Convergencia del agrupamiento difuso")
    ax.grid(alpha=0.3); fig.tight_layout()
    guardar(fig, "07_convergencia_fcm.png");

    print("[5/6] Escribiendo métricas agregadas…")
    res = {
        "validacion_fcm": modelo["validacion_fcm"],
        "centros_fcm_unicos": n_unicos,
        "exactitud_train": modelo["exactitud_train"],
        "comparacion": filas,
        "matriz_confusion": cm.tolist(),
        "clases": CLASSES,
        "reporte_por_clase": {
            c: {k: float(v) for k, v in reporte[c].items()} for c in CLASSES
        },
        "archivos": {
            "figuras": [p.name for p in sorted(FIGURES.glob("*.png"))],
            "modelo": MODELS.name + "/modelo_fcm_tsk.json",
        },
    }
    (METRICS / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[6/6] Generando documento docs/03_resultados.md…")
    principal = tabla.iloc[0]
    filas_md = "\n".join(
        f"| {r['modelo']} | {r['exactitud']*100:.2f}% | {r['f1_macro']:.3f} | {r['latencia_ms']:.4f} |"
        for _, r in tabla.iterrows()
    )
    filas_clase = "\n".join(
        f"| {c} | {reporte[c]['precision']:.3f} | {reporte[c]['recall']:.3f} | {reporte[c]['f1-score']:.3f} |"
        for c in CLASSES
    )
    reglas = json.loads((MODELS / "reglas_resumen.json").read_text(encoding="utf-8"))
    filas_reglas = "\n".join(
        f"| R{r['regla']} | {r['gesto_inicial']} | " +
        ", ".join(t["caracteristica"] for t in r["top_5_caracteristicas"]) + " |"
        for r in reglas
    )
    md = f"""# Resultados experimentales (generados por `src/03_evaluar.py`)

## 1. Partición difusa (FCM)

- Clústeres: **{modelo['c']}** (uno por gesto base).
- Exponente de difusividad: **m = {modelo['m']}**.
- Iteraciones hasta convergencia: **{modelo['validacion_fcm']['iteraciones']}**.
- Coeficiente de partición (PC): **{modelo['validacion_fcm']['PC']:.4f}**.
- Entropía de partición (PE): **{modelo['validacion_fcm']['PE']:.4f}**.

PC y PE se calcularon sobre la matriz de pertenencia del conjunto de entrenamiento.
Valores intermedios (PC lejos de 1, PE lejos de 0) indican solapamiento difuso
entre gestos, coherente con la co-contracción muscular.

**Hallazgo:** los 5 centros de FCM colapsan en solo **{n_unicos} centros distintos**
al converger. Dos pares de gestos comparten estadísticas de Hudgins y no son
separables por prototipos, aunque sí por hiperplanos TSK. Para conservar 5 reglas
semánticas se evaluó una variante con gaussianas ancladas a cada gesto, incluida
en la tabla siguiente y exportada a `outputs/models/modelo_gaussiano_clases.json`.

## 2. Comparación de modelos (mismo split 70/30)

| Modelo | Exactitud | F1-macro | Latencia (ms/muestra) |
| :--- | ---: | ---: | ---: |
{filas_md}

> La latencia se mide en Python/NumPy sobre esta máquina. El valor de 0.003 ms
> citado en el contexto corresponde a una implementación optimizada en C para
> hardware embebido; no es directamente comparable con esta medición.

## 3. Desempeño por clase del modelo propuesto

| Gesto | Precisión | Sensibilidad | F1 |
| :--- | ---: | ---: | ---: |
{filas_clase}

Exactitud global del modelo propuesto: **{principal['exactitud']*100:.2f}%**.

## 4. Reglas TSK auditables

| Regla | Gesto dominante | 5 características de mayor peso |
| :--- | :--- | :--- |
{filas_reglas}

Las ecuaciones completas están en `outputs/models/modelo_fcm_tsk.json`
(`centros`, `P` de forma 5×5×32 y `q` de forma 5×5).

## 5. Figuras

![Distribución](../outputs/figures/01_distribucion_clases.png)
![Señal filtrada](../outputs/figures/02_senal_filtrada.png)
![Pertenencias](../outputs/figures/03_pertenencias_fcm.png)
![Matriz de confusión](../outputs/figures/04_matriz_confusion.png)
![Importancia](../outputs/figures/05_importancia_canales.png)
![Pareto](../outputs/figures/06_pareto.png)
![Convergencia](../outputs/figures/07_convergencia_fcm.png)
![Centroides proyectados](../outputs/figures/10_scatter_centroides.png)

La proyección de los centroides a 2D se genera con
`src/08_scatter_centroides.py`; las coordenadas están en
`outputs/metrics/proyeccion_centroides.csv`.

## 6. Advertencias metodológicas

1. El split es aleatorio por ventanas: hay solapamiento entre train y test
   (fuga de información), por lo que la exactitud reportada es optimista.
2. `TRIM=0` reproduce las 2.139 ventanas del contexto; el script original usa
   `TRIM=50` (2.125 ventanas) para descartar transiciones.
3. La normalización z-score se ajustó solo con entrenamiento.

## 7. Análisis comparativo

Este informe describe los resultados medidos. La discusión comparativa frente a
CNN-BiLSTM, TCN int8, Random Forest, Mamdani y ANFIS (complejidad, huella de
memoria, interpretabilidad, fisiología y Pareto) está en
[`05_analisis_comparativo.md`](05_analisis_comparativo.md), que se regenera con
`src/07_analisis_comparativo.py`.
"""
    (DOCS / "03_resultados.md").write_text(md, encoding="utf-8")
    print("Listo. Exactitud FCM+TSK:", f"{principal['exactitud']*100:.2f}%")


if __name__ == "__main__":
    main()
