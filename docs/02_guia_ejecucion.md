# Guía de ejecución

## Requisitos

Python 3.10+ con:

```bash
python -m pip install -r requirements.txt
```

No se necesita TensorFlow para este pipeline: el modelo neuro-difuso se entrena
con NumPy/scikit-learn. Los scripts TFLite originales están en `referencias/`
solo como antecedente.

## Pipeline completo

Desde la raíz de `proyecto_fuzzy_emg`:

```bash
python src/run_all.py
```

Ejecuta, en orden:

1. `01_preparar_datos.py` → `data/processed/ventanas_features.npz` y `.csv`
2. `02_entrenar_fcm_tsk.py` → `outputs/models/modelo_fcm_tsk.json` + reglas
3. `03_evaluar.py` → `outputs/metrics/*`, `outputs/figures/*`, `docs/03_resultados.md`
4. `07_analisis_comparativo.py` → `comparacion_multiparadigma.csv`,
   `09_pareto_multiparadigma.png` y `docs/05_analisis_comparativo.md`
5. `08_scatter_centroides.py` → `10_scatter_centroides.png` y
   `proyeccion_centroides.csv`

Salida esperada (resumen):

```
PC=0.3454  PE=1.2856
exactitud train=97.86%  test=97.19%
Listo. Exactitud FCM+TSK: 97.19%
```

## Comandos adicionales

```bash
# Ver una inferencia desglosada
python src/05_demo_inferencia.py

# ¿Cómo cambia el solapamiento si cambio m?
python src/06_barrido_m.py

# Análisis comparativo multidimensional (requiere haber corrido 03_evaluar.py)
python src/07_analisis_comparativo.py

# Scatter de centroides en 2D (requiere 01_preparar_datos.py y 02_entrenar_fcm_tsk.py)
python src/08_scatter_centroides.py

# Inferencia en streaming desde el CSV, sin hardware (dry run)
python src/04_inferencia_tiempo_real.py --max-muestras 4000

# Reproducir el CSV a 200 Hz reales
python src/04_inferencia_tiempo_real.py --tiempo-real --max-muestras 10000
```

### Con hardware real (Raspberry Pi)

```bash
# 1) Grabar nuevos datos
python referencias/scripts_originales/recorder_dongle.py

# 2) Inferencia con Myo + dongle BLED112 (sin mover motores)
python src/04_inferencia_tiempo_real.py --fuente myo

# 3) Inferencia + servos PCA9685
python src/04_inferencia_tiempo_real.py --fuente myo --servos
```

Dependencias extra en la Raspberry:

```bash
python -m pip install pyomyo pyserial adafruit-circuitpython-servokit
```

Alimenta los servos con fuente externa de 5 V y **une el GND** con la
Raspberry/PCA9685. Ajusta `POSES` en `src/config.py` según tu mano real.

## Cambiar parámetros

Todo está en `src/config.py`:

| Parámetro | Valor | Efecto |
| :--- | ---: | :--- |
| `WINDOW` | 150 | ventana de 0,75 s |
| `HOP` | 50 | decisión cada 0,25 s |
| `TRIM` | 0 | 0 → 2.139 ventanas; 50 → 2.125 |
| `TRAIN_FRACTION` | 0,70 | 70/30 train/test |
| `N_CLUSTERS` | 5 | una regla por gesto |
| `FUZZIFIER` | 2,0 | difusividad $m$ |
| `SEED` | 13 | reproducibilidad |

Tras cambiar algo, vuelve a correr `python src/run_all.py`.

## Problemas frecuentes

* **Acentos mojibake en la terminal de Windows**: usa `chcp 65001` o
  PowerShell con UTF-8. Los archivos `.md` y `.json` ya están en UTF-8.
* **`ValueError: Invalid zi shape`**: significa que se mezclaron los filtros;
  `ProcesadorEMG` aplana `sos_bp + notches` antes de crear los estados.
* **No aparece `docs/03_resultados.md`**: ejecuta `03_evaluar.py`; lo genera.
* **`pyomyo` / `adafruit_servokit` no instalados**: solo se importan si usas
  `--fuente myo` o `--servos`. El modo `replay` funciona igual.
* **Figuras no se sobrescriben**: `guardar()` en `03_evaluar.py`,
  `07_analisis_comparativo.py` y `08_scatter_centroides.py` usa nombres
  temporales y reintenta para evitar bloqueos de archivos en Windows.
* **`07_analisis_comparativo.py` falla con `FileNotFoundError`**: ejecuta antes
  `src/03_evaluar.py`, porque lee `outputs/metrics/comparacion_modelos.csv` y
  `outputs/metrics/resumen.json`.
