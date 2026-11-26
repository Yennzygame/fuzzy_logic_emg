# Archivos de referencia

Los scripts originales se publican para que el pipeline sea reproducible. No se
modificaron. Los materiales de clase y los textos de trabajo internos se
conservan solo en local y están excluidos mediante `.gitignore`.

## `scripts_originales/` (publicado)

| Archivo | Papel en el proyecto |
| :--- | :--- |
| `recorder_dongle.py` | Captura sEMG del Myo vía dongle BLED112 a 200 Hz y escribe `dataset_completo.csv`. |
| `train_model_dongle.py` | Pipeline CNN-1D + BiLSTM con filtrado, ventaneo (W=150, HOP=50, TRIM=50) y exportación TFLite FP16. Referencia de preprocesamiento y de la línea base "caja negra". |
| `realtime_dongle.py` | Inferencia TFLite en tiempo real con búfer circular, z-score y suavizado por votación. |
| `realtime_servos.py` | Inferencia TFLite + control de 5 servos por PCA9685 (poses, rampa y cooldown). Base de `src/04_inferencia_tiempo_real.py`. |

## Contenido solo local (no publicado)

* `scripts_originales/contexto proyecto logica fuzzy.txt`: notas de planificación
  de las fases 1–5.
* `scripts_originales/comparacion_propuesta_notebooklm.txt`: borrador de los
  criterios de comparación. El resultado depurado está en
  [`docs/05_analisis_comparativo.md`](../docs/05_analisis_comparativo.md).
* `material_clase/`: 15 PDFs sobre lógica fuzzy y el informe de trabajo de grado
  **F-DC-125** en `.docx`. Referencias teóricas utilizadas en
  `docs/01_metodologia.md`:

  * `Fuzzy_C_Means_Thayza_final.pdf`
  * `Extração de Regras - slides.pdf`
  * `Generating Fuzzy Rules by Learning from Examples.pdf`
  * `parte_7_sistemas_fuzzy.pdf` (inferencia, defuzzificación, TSK)
  * `main.pdf` (controladores fuzzy, TSK en universos discretizados)

> Nota: el artículo de *TinyML Benchmarking* no está incluido en los archivos
> locales; sus cifras se citan solo como antecedente.
