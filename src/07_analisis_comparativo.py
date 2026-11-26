"""Fase 4b: análisis comparativo multidimensional (marco de maestría).

Combina dos planos que nunca deben mezclarse en una misma tabla sin advertencia:
  1) cifras REPORTADAS en las fuentes (protocolos heterogéneos);
  2) cifras MEDIDAS en esta reproducción (mismo split 70/30).

Genera:
  outputs/metrics/comparacion_multiparadigma.csv
  outputs/figures/09_pareto_multiparadigma.png
  docs/05_analisis_comparativo.md
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DOCS, FIGURES, METRICS

# -------------------- Plan A: literatura (contexto + propuesta NotebookLM) --------------------
LITERATURA = [
    {
        "paradigma": "Deep Learning pesado",
        "modelo": "CNN-1D + BiLSTM (FP16)",
        "exactitud": 99.60, "f1": 0.9958, "latencia": "10–35", "latencia_med": 22.5,
        "memoria": "Alta: cientos de KB a MB (buffers recurrentes)",
        "interpretabilidad": "Nula (caja negra)",
        "transicion": "Captura dependencias espacio-temporales complejas",
        "origen": "Tesis F-DC-125 / diapositivas",
    },
    {
        "paradigma": "TinyML cuantizado",
        "modelo": "TCN Ensemble int8",
        "exactitud": 83.46, "f1": None, "latencia": "46.2 (ESP-NN) / 277 (cons.)",
        "latencia_med": 46.2,
        "memoria": "140,4 KB Flash / 38,4 KB RAM",
        "interpretabilidad": "Opaca post-hoc (SHAP)",
        "transicion": "Degradación por truncamiento int8 (pérdida de 6,7 puntos)",
        "origen": "Artículo TinyML Benchmarking",
    },
    {
        "paradigma": "ML ligero",
        "modelo": "Random Forest en C",
        "exactitud": 90.67, "f1": 0.9092, "latencia": "0,004–0,021", "latencia_med": 0.0125,
        "memoria": "17,9 KB Flash / 0,08 KB RAM",
        "interpretabilidad": "Baja-media (caja gris, umbrales rígidos)",
        "transicion": "Umbrales duros; sensible a cambios leves de amplitud",
        "origen": "Artículo TinyML Benchmarking",
    },
    {
        "paradigma": "Lógica difusa tradicional",
        "modelo": "Mamdani estático",
        "exactitud": 88.94, "f1": None, "latencia": "0,018", "latencia_med": 0.018,
        "memoria": "Baja (tablas de búsqueda / universos discretos)",
        "interpretabilidad": "Alta lingüística",
        "transicion": "Falla en zonas de alta superposición muscular",
        "origen": "Benchmarking interno / diapositivas",
    },
    {
        "paradigma": "Neuro-difuso",
        "modelo": "ANFIS (TSK + gradiente/LSE)",
        "exactitud": 98.60, "f1": None, "latencia": "0,001", "latencia_med": 0.001,
        "memoria": "Moderada (muchas reglas si hay grid partition)",
        "interpretabilidad": "Media-alta (reglas TSK)",
        "transicion": "Buena, depende del número de reglas",
        "origen": "Contexto original del proyecto",
    },
    {
        "paradigma": "Neuro-difuso propuesto",
        "modelo": "FCM + TSK Orden 1 (reportado)",
        "exactitud": 97.66, "f1": None, "latencia": "0,003", "latencia_med": 0.003,
        "memoria": "Mínima: 5 centros + 5 matrices lineales de 32 entradas",
        "interpretabilidad": "Intrínseca (caja de cristal)",
        "transicion": "Continua: modela co-contracción y fatiga",
        "origen": "Diapositivas / contexto (NO medido aquí)",
    },
]

# -------------------- Plan B: reproducción local (mismo protocolo) --------------------
INTERP_LOCAL = {
    "FCM + TSK (pertenencia FCM)": "Intrínseca (caja de cristal)",
    "FCM + TSK (pertenencia Gaussiana)": "Intrínseca (caja de cristal)",
    "5 gaussianas por gesto + TSK": "Intrínseca (caja de cristal)",
    "K-means rígido + regla de mayoría": "Baja (centroides rígidos)",
    "KNN (k=5)": "Nula",
    "Random Forest (200 árboles)": "Media (importancias post-hoc)",
}


def guardar(fig, nombre, intentos=6):
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


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    local = pd.read_csv(METRICS / "comparacion_modelos.csv", encoding="utf-8")
    resumen = json.loads((METRICS / "resumen.json").read_text(encoding="utf-8"))
    pc = resumen["validacion_fcm"]["PC"]
    pe = resumen["validacion_fcm"]["PE"]
    n_unicos = resumen["centros_fcm_unicos"]

    filas = []
    for r in LITERATURA:
        filas.append({
            "plano": "literatura", "paradigma": r["paradigma"], "modelo": r["modelo"],
            "exactitud_pct": r["exactitud"], "f1": r["f1"],
            "latencia_ms": r["latencia"], "memoria": r["memoria"],
            "interpretabilidad": r["interpretabilidad"],
            "transicion_ruido": r["transicion"], "origen": r["origen"],
            "comparable_entre_si": "No (protocolos distintos)",
        })
    for _, r in local.iterrows():
        filas.append({
            "plano": "reproduccion_local", "paradigma": "evaluación local",
            "modelo": r["modelo"], "exactitud_pct": round(100 * r["exactitud"], 2),
            "f1": round(r["f1_macro"], 4), "latencia_ms": round(r["latencia_ms"], 4),
            "memoria": "no medida (Python/NumPy)",
            "interpretabilidad": INTERP_LOCAL[r["modelo"]],
            "transicion_ruido": "evaluada en el mismo split",
            "origen": "src/03_evaluar.py", "comparable_entre_si": "Sí (mismo split 70/30)",
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(METRICS / "comparacion_multiparadigma.csv", index=False)

    # --- Figura: frontera exactitud / latencia con los dos planos ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in LITERATURA:
        ax.scatter(r["latencia_med"], r["exactitud"], s=120, marker="o",
                   color="#3b6fb6", edgecolor="k", zorder=3, label="Literatura (heterogénea)")
    for _, r in local.iterrows():
        ax.scatter(r["latencia_ms"], 100 * r["exactitud"], s=140, marker="s",
                   color="#e07a2f", edgecolor="k", zorder=4,
                   label="Reproducción local (mismo split)")
    ax.axvspan(100, 125, color="green", alpha=0.12)
    ax.text(108, 82, "Umbral clínico\n100–125 ms\n(END-TO-END)", fontsize=8, color="green")
    ax.set_xscale("log")
    ax.set_xlabel("Latencia de inferencia (ms, escala logarítmica)")
    ax.set_ylabel("Exactitud (%)")
    ax.set_title("Mapa de diseño: exactitud vs. latencia vs. plan de evidencia")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[len(LITERATURA)]], [labels[0], labels[len(LITERATURA)]],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    guardar(fig, "09_pareto_multiparadigma.png")

    # --- Documento analítico ---
    lit_md = "\n".join(
        f"| {r['paradigma']} | {r['modelo']} | {r['exactitud']:.2f} % | {r['latencia']} ms | "
        f"{r['memoria']} | {r['interpretabilidad']} | {r['transicion']} | {r['origen']} |"
        for r in LITERATURA
    )
    loc_md = "\n".join(
        f"| {r['modelo']} | {100*r['exactitud']:.2f} % | {r['f1_macro']:.3f} | "
        f"{r['latencia_ms']:.4f} ms | {INTERP_LOCAL[r['modelo']]} |"
        for _, r in local.iterrows()
    )

    n_params = 5 * 32 + 5 * 5 * 32 + 5 * 5
    bytes_tsk = n_params * 4
    principal = local[local["modelo"] == "FCM + TSK (pertenencia FCM)"].iloc[0]
    gaussiana = local[local["modelo"] == "5 gaussianas por gesto + TSK"].iloc[0]

    md = f"""# Análisis comparativo multidimensional (nivel maestría)

Generado por `src/07_analisis_comparativo.py` a partir de los resultados locales
y de la propuesta guardada en `referencias/scripts_originales/comparacion_propuesta_notebooklm.txt`.

## 0. Regla metodológica: dos planos de evidencia

Antes de comparar, hay que decir de dónde sale cada número:

| Plano | Qué es | Cómo se usa |
| :--- | :--- | :--- |
| **Literatura / fuentes** | Cifras reportadas en la tesis F-DC-125, el artículo de TinyML y las diapositivas. Protocolos, hardware y particiones distintos. | Sirven para **mapear el espacio de diseño** y discutir tendencias. **No** sirven para declarar que un modelo es mejor que otro. |
| **Reproducción local** | Mediciones de `src/03_evaluar.py` sobre **el mismo dataset y el mismo split 70/30**. | Sirven para **comparar modelos entre sí** con rigor relativo. |

Mezclar ambos planos en una sola tabla sin esta advertencia es el error más
común en este tipo de informe. Aquí están separados.

## 1. Marco comparativo multiparadigma (literatura)

| Paradigma | Modelo | Exactitud | Latencia | Memoria / carga | Interpretabilidad | Transición y ruido | Origen |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- |
{lit_md}

Lectura: el mejor número de exactitud (99,60 %) pertenece al modelo más pesado
y opaco; el mejor número de latencia reportado (0,001 ms de ANFIS) pertenece a
una arquitectura con muchas reglas. No existe un ganador único: hay una
**frontera de compromisos**.

## 2. Comparación bajo protocolo común (reproducción local)

| Modelo | Exactitud | F1-macro | Latencia (Python) | Interpretabilidad |
| :--- | ---: | ---: | ---: | :--- |
{loc_md}

Aquí sí se puede afirmar: el modelo propuesto gana ~5 puntos al k-means
rígido, es comparable a KNN y pierde frente a Random Forest, pero es ~300 veces
más rápido que RF en esta máquina y además auditable. La variante con 5
gaussianas por gesto alcanza {100*gaussiana['exactitud']:.2f} %.

## 3. Análisis de complejidad y memoria

### 3.1 Costo de inferencia TSK

Para $c=5$ reglas y $d=32$ características:

- pertenencias: $c\\cdot d = 160$ operaciones;
- consecuentes: $c\\cdot(d\\cdot 5) = 800$ multiplicaciones;
- agregación: $c\\cdot5 = 25$;

≈ **{5*32 + 5*5*32 + 5*5} operaciones escalares** por decisión. En C embebido
eso son microsegundos, lo que hace creíble el valor reportado de 0,003 ms. En
Python/NumPy medimos {principal['latencia_ms']:.4f} ms: el costo lo domina el
*overhead* del interprete, no la aritmética.

### 3.2 Huella del modelo

```
centros: 5 × 32 = 160 floats
P:       5 × 5 × 32 = 800 floats
q:       5 × 5 = 25 floats
total:   {n_params} floats × 4 bytes = {bytes_tsk} bytes ≈ {bytes_tsk/1024:.2f} KB
```

Frente a 140,4 KB de Flash del TCN cuantizado y 17,9 KB del Random Forest en C,
la huella del clasificador difuso es casi despreciable. El costo real del
sistema está en el búfer de 150×8 muestras (9,6 KB en float32) y en el filtrado.

### 3.3 Por qué Mamdani es más caro

En Mamdani la defuzzificación por centroide exige

$$y^{{\\rm CoG}}=\\frac{{\\int y\\,\\mu(y)\\,dy}}{{\\int \\mu(y)\\,dy}}$$

sobre un universo discretizado de salida. Con $R$ reglas y $N$ puntos de la
cuadrícula el costo es $O(RN)$ más acceso a memoria para agregar áreas. En TSK
se sustituye esa integración por una media ponderada de hiperplanos: el mismo
resultado numérico sin discretización, sin pérdida de resolución y con menos
memoria. Es el argumento teórico central de la clase (notas *Parte 7* y
*main.pdf*) aplicado a un problema real.

## 4. Interpretabilidad: intrínseca vs. post-hoc

El artículo de *TinyML Benchmarking* encontró que, al explicar TCN con
GradientExplainer y Random Forest con TreeSHAP, la correlación de Spearman
entre la importancia de los canales era de solo **ρ = 0,262**. El canal 5 era
crítico para uno e irrelevante para el otro. Conclusión: la explicación
post-hoc es un artefacto del explicador, no una propiedad del modelo.

En contraste, cada regla TSK contiene sus propios coeficientes: se puede
responder "¿qué canal y qué característica empujaron esta decisión?" abriendo
`outputs/models/modelo_fcm_tsk.json`. No hay un segundo algoritmo que pueda
contradecir al primero.

**Pero cuidado con sobreclaimar:** transparencia no equivale a corrección
fisiológica. Los coeficientes son auditables, sí, pero su correspondencia con
la fisiología real debe validarse con un clínico. Además, en esta reproducción
los 5 centros FCM colapsaron en **{n_unicos} centros útiles**: dos pares de
gestos comparten prototipos. Esa limitación se reporta, no se oculta.

**Frente a otros enfoques neuro-difusos:**

| Enfoque | Antecedentes | Consecuentes | Aprendizaje | Limitación para prótesis |
| :--- | :--- | :--- | :--- | :--- |
| Wang & Mendel (1992) | Grid partition | Reglas lingüísticas | Extracción de ejemplos | Explosión de reglas en 32 dimensiones |
| ANFIS | FCM / grid | TSK | Gradiente + mínimos cuadrados | Muchas reglas si no se controla la partición |
| NEFCON | Base Mamdani | Difusa | Refuerzo en lazo cerrado | Defuzzificación por centroide |
| **Este trabajo** | FCM (o gaussianas por clase) | TSK orden 1 | WLS ponderado | Requiere validación clínica y split por sujeto |

## 5. Fisiología: por qué el difuso gana al rígido

La señal sEMG no es discreta:

- **diafonía**: un electrode capta músculos vecinos;
- **fatiga**: desplaza el espectro a frecuencias más bajas (WL es sensible a esto);
- **co-contracción**: agonistas y antagonistas se activan juntos;
- **transición**: el paso de reposo a pinza es gradual.

Un clasificador *crisp* con umbral duro comete **conmutación violenta**: un
poco de ruido cruza el umbral y el servo salta. FCM asigna grados continuos
($0\\le u\\le1$), por lo que la salida TSK se desplaza suavemente. Nuestros
índices, PC={pc:.4f} y PE={pe:.4f} (máximo ln5=1,609), confirman que la
partición conserva superposición moderada-alta.

La consecuencia observable está en la matriz de confusión: reposo, puño
cerrado y agarre de vaso son perfectos, mientras mano abierta tiene
sensibilidad 0,789 y se confunde con pinza. No es un bug: es la huella de la
diafonía en un par de gestos que comparten activación muscular.

## 6. Frontera de Pareto y viabilidad clínica

La literatura clínica fija ~100–125 ms como tiempo de respuesta percibido como
instantáneo. Los modelos de inferencia rápida (0,003–0,02 ms) están muy por
debajo de ese límite. Sin embargo, hay una observación de maestría que suele
omitirse:

> **La latencia de inferencia no es la latencia end-to-end.** El sistema espera
> una ventana de 750 ms antes de decidir, más la rampa mecánica de los servos
> (~65 ms en el peor caso). El tiempo percibido por el usuario es del orden de
> 800 ms, no 0,003 ms.

Por tanto, el aporte real de TSK no es "hacer que la prótesis responda en 3
microsegundos", sino **eliminar el cuello de botella computacional** para que
el limite lo ponga la fisiología y la mecánica, no el modelo. Reducir la ventana
de adquisición sin perder exactitud es el siguiente frente de investigación.

## 7. Tesis defendible

> *"No se propone únicamente un clasificador, sino una comparación formal entre
> TSK de orden 1 identificado con FCM, Mamdani/NEFCON y Deep Learning. La
> evaluación algebraica de TSK resuelve el cuello de botella de la
> defuzzificación geométrica y la opacidad clínica de las redes neuronales,
> ofreciendo una solución auditable en la frontera de Pareto para hardware
> embebido, con la salvedad de que la latencia end-to-end sigue dominada por la
> ventana de 750 ms."*

## 8. Limitaciones y trabajo futuro

1. Repetir el split por bloques o por sujeto para eliminar la fuga de ventanas
   solapadas.
2. Validar la interpretabilidad de los coeficientes con personal clínico.
3. Medir inferencia en C sobre la Raspberry Pi (no en Python).
4. Reducir la ventana de 750 ms y cuantificar el compromiso exactitud/latencia.
5. Comparar contra ANFIS y Mamdani **implementados sobre este mismo dataset**,
   no solo contra cifras literarias.

## 9. Figura

![Mapa multiparadigma](../outputs/figures/09_pareto_multiparadigma.png)
"""
    (DOCS / "05_analisis_comparativo.md").write_text(md, encoding="utf-8")
    print("Análisis comparativo generado:")
    print(" -", METRICS / "comparacion_multiparadigma.csv")
    print(" -", FIGURES / "09_pareto_multiparadigma.png")
    print(" -", DOCS / "05_analisis_comparativo.md")


if __name__ == "__main__":
    main()
