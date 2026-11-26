# Datos

## `raw/dataset_completo.csv`

Copia exacta del dataset original (107.820 filas; columnas `CH1`…`CH8` y `G`).
Frecuencia: 200 Hz. Gestos: `reposo`, `mano_abierta`, `puño_cerrado`, `pinza`,
`agarre_vaso`.

Distribución de muestras crudas:

| Gesto | Muestras |
| :--- | ---: |
| pinza | 35.940 |
| agarre_vaso | 23.962 |
| reposo | 23.960 |
| puño_cerrado | 11.980 |
| mano_abierta | 11.978 |

El archivo está en bloques continuos: 7 bloques, con `pinza` y `agarre_vaso`
repetidos al final.

## `processed/`

Generado por `src/01_preparar_datos.py`:

* `ventanas_features.npz`: arreglos `X_train`, `X_test`, `y_train`, `y_test`,
  `Y_train` (one-hot), grupos de bloque, `mu`, `sigma` y nombres de variables.
* `ventanas_features.csv`: **2.139 filas**, una por ventana, con 32
  características Hudgins, etiqueta, split y bloque de origen. Es el archivo
  ideal para abrir en Excel/Python al presentar.
* `resumen_dataset.json`: conteos, parámetros y la advertencia sobre fuga de
  información en el split aleatorio.

Orden de columnas: `CH1_RMS, CH1_MAV, CH1_WL, CH1_ZC, CH2_RMS, … , CH8_ZC`.
