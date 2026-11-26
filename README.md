# Clasificación interpretable de sEMG con Fuzzy C-Means + TSK

Repositorio con el pipeline completo para clasificar gestos a partir de señales
electromiográficas de la Myo Armband (8 canales, 200 Hz). Cada ventana se resume
en 32 características de Hudgins y se clasifica en 5 gestos mediante un sistema
neuro-difuso **FCM + Takagi-Sugeno-Kang de orden 1**.

La motivación es el "trilema" del control protésico: exactitud alta, latencia
ultra-baja e **interpretabilidad clínica** (caja de cristal), frente a las redes
CNN/BiLSTM que son caja negra.

## Resultados reproducidos en esta carpeta

| Métrica | Valor obtenido | Valor citado en el contexto |
| :--- | ---: | ---: |
| Ventanas analizadas | 2.139 | 2.139 |
| Partición train / test | 1.497 / 642 (70/30) | 1.497 / 642 |
| Exactitud FCM + TSK | **97,19 %** | 97,66 % |
| PC (FCM) | **0,3454** | 0,3905 |
| PE (FCM) | **1,2856** | 1,1849 |
| Exactitud 5 gaussianas por gesto + TSK | **98,28 %** | — |
| Latencia (Python/NumPy, por inferencia) | ~0,18 ms | 0,003 ms (C embebido) |

> Los números de esta tabla son **mediciones reales** del pipeline incluido.
> Las diferencias con el contexto se explican en
> [`docs/03_resultados.md`](docs/03_resultados.md) y `docs/01_metodologia.md`.

## Análisis comparativo

El repositorio incluye una comparación **multidimensional** de seis paradigmas
(CNN-BiLSTM, TCN int8, Random Forest, Mamdani, ANFIS y FCM+TSK) que cubre
complejidad, huella de memoria, interpretabilidad intrínseca vs. post-hoc,
fisiología de la señal y frontera de Pareto. El principio metodológico central es separar dos planos de
evidencia: **literatura** (protocolos heterogéneos, solo para mapear el espacio
de diseño) y **reproducción local** (mismo split, para comparar modelos).

👉 [`docs/05_analisis_comparativo.md`](docs/05_analisis_comparativo.md) ·
[`outputs/metrics/comparacion_multiparadigma.csv`](outputs/metrics/comparacion_multiparadigma.csv) ·
[`outputs/figures/09_pareto_multiparadigma.png`](outputs/figures/09_pareto_multiparadigma.png)

Los criterios de comparación están documentados en
`referencias/scripts_originales/comparacion_referencia.txt`.

## Estructura

```
proyecto_fuzzy_emg/
├── README.md                    ← este archivo
├── requirements.txt
├── data/
│   ├── raw/dataset_completo.csv ← dataset original (copia íntegra)
│   └── processed/               ← tabla de 32 características + npz + resumen
├── src/
│   ├── config.py                ← parámetros (FS, W, HOP, c, m, semilla…)
│   ├── signal_features.py       ← filtrado, ventaneo y Hudgins
│   ├── fuzzy_model.py           ← Fuzzy C-Means y TSK + PC/PE
│   ├── 01_preparar_datos.py     ← fase 1
│   ├── 02_entrenar_fcm_tsk.py   ← fases 2 y 3
│   ├── 03_evaluar.py            ← fase 4 (métricas, figuras e informe)
│   ├── 04_inferencia_tiempo_real.py ← fase 5 (Myo/CSV + PCA9685)
│   ├── 05_demo_inferencia.py    ← una inferencia paso a paso
│   ├── 06_barrido_m.py          ← sensibilidad del exponente m
│   ├── 07_analisis_comparativo.py ← comparación multiparadigma
│   └── run_all.py               ← ejecuta 01 → 02 → 03 → 07
├── outputs/
│   ├── figures/                 ← 9 figuras PNG
│   ├── metrics/                 ← CSV/JSON con métricas y matriz de confusión
│   └── models/                  ← modelo FCM+TSK, reglas y pesos
├── docs/
│   ├── 01_metodologia.md        ← fundamentos del modelo
│   ├── 02_guia_ejecucion.md     ← comandos y solución de problemas
│   ├── 03_resultados.md         ← informe generado automáticamente
│   ├── 04_preguntas_frecuentes.md ← cuestionario técnico
│   └── 05_analisis_comparativo.md ← análisis comparativo (generado)
└── referencias/
    ├── scripts_originales/      ← 4 scripts originales + contexto + criterios de comparación
    └── material_clase/          ← PDFs y tesis (solo local, no publicado)
```

## Inicio rápido

```bash
cd proyecto_fuzzy_emg
python -m pip install -r requirements.txt
python src/run_all.py
python src/06_barrido_m.py
python src/07_analisis_comparativo.py
python src/05_demo_inferencia.py
python src/04_inferencia_tiempo_real.py --max-muestras 4000
```

Todo se ejecuta sin hardware. Para usar la pulsera real, consulta
[`docs/02_guia_ejecucion.md`](docs/02_guia_ejecucion.md).

## Control de versiones

El repositorio está inicializado en `main` y tiene configurado `origin`. Para
actualizarlo tras regenerar resultados:

```bash
cd proyecto_fuzzy_emg
git add -A
git commit -m "Actualizar resultados del pipeline FCM + TSK"
git push origin main
```

La carpeta `referencias/material_clase/` (PDFs y tesis) y los textos de trabajo
internos están excluidos mediante `.gitignore`: se conservan en local pero
**no se publican**. Sí se publican los scripts originales, los datos, los
resultados y la documentación técnica.

Antes de subir, revisa `data/raw/dataset_completo.csv`: contiene señales de un
participante. Si el protocolo exige anonimato, publica solo
`data/processed/ventanas_features.csv` (no incluye identificadores personales).

## Material de clase relacionado

Los siguientes PDFs se usaron como referencia teórica y **están solo en la copia
local** (`referencias/material_clase/`, excluida de GitHub):

- `Parte 2`, `Parte 3`, `Parte 6`, `parte_7_sistemas_fuzzy.pdf`, `main.pdf`:
  conjuntos fuzzy, operaciones, inferencia y TSK.
- `Fuzzy_C_Means_Thayza_final.pdf`, `Extração de Regras - slides.pdf`,
  `Generating Fuzzy Rules by Learning from Examples.pdf`: antecedentes y
  extracción de reglas.
- `F-DC-125 …docx`: tesis de grado antecedente.

Sí están publicados los scripts que se usaron como base:
`referencias/scripts_originales/realtime_servos.py` (versión TFLite + PCA9685 en
la que se basa la fase 5), `recorder_dongle.py`, `train_model_dongle.py` y
`realtime_dongle.py`.
