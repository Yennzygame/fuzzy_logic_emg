# Artefactos generados

Todos los archivos de esta carpeta se regeneran con `python src/run_all.py`
(más `src/06_barrido_m.py` y `src/07_analisis_comparativo.py`). No los edites
a mano.

## `models/`

| Archivo | Contenido |
| :--- | :--- |
| `modelo_fcm_tsk.json` | 5 centros, `P` (5×5×32), `q` (5×5), $\mu/\sigma$ de normalización, PC/PE e histórico de la función objetivo. |
| `modelo_gaussiano_clases.json` | Variante con 5 gaussianas ancladas a cada gesto (98,28 % de exactitud). |
| `reglas_resumen.json` | Por regla: gesto inicial, dominante según `q`, 5 características de mayor peso y peso ponderado. |
| `importancia_canales.csv` | $|P|$ promediado sobre reglas y clases, ordenado de mayor a menor. |

## `metrics/`

* `comparacion_modelos.csv`: exactitud, F1 y latencia de los 6 modelos
  (mismo split; plano de evidencia local).
* `comparacion_multiparadigma.csv`: 12 filas con los dos planos de evidencia —
  6 modelos de literatura (no comparables entre sí) y 6 locales (comparables).
* `reporte_clasificacion.csv`: precisión/sensibilidad/F1 por gesto.
* `matriz_confusion.csv`: 5×5 con etiquetas reales y predichas.
* `resumen.json`: métricas agregadas en un solo archivo.
* `barrido_m.csv`: PC, PE y exactitud para $m\in[1{,}2,10]$.

## `figures/`

1. Distribución de clases (crudas y ventanas)
2. Señal cruda vs. filtrada
3. Pertenencia media por gesto y cluster
4. Matriz de confusión
5. Importancia de características
6. Frontera exactitud / latencia
7. Convergencia de FCM
8. Barrido del exponente $m$
9. Mapa multiparadigma: exactitud vs. latencia, separando literatura de
   reproducción local y marcando el umbral clínico de 100–125 ms
