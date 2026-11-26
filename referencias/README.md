# Material antecedente (copias íntegras)

Los scripts y documentos de texto se publican en el repositorio para que sea
reproducible. **El material de clase (`material_clase/`) se conserva solo en
local y está excluido de GitHub mediante `.gitignore`; no se modificó ningún
archivo.**

## `scripts_originales/`

| Archivo | Papel en el proyecto |
| :--- | :--- |
| `recorder_dongle.py` | Captura sEMG del Myo vía dongle BLED112 a 200 Hz y escribe `dataset_completo.csv`. |
| `train_model_dongle.py` | Pipeline CNN-1D + BiLSTM con filtrado, ventaneo (W=150, HOP=50, TRIM=50) y exportación TFLite FP16. Referencia de preprocesamiento y de la línea base "caja negra". |
| `realtime_dongle.py` | Inferencia TFLite en tiempo real con búfer circular, z-score y suavizado por votación. |
| `realtime_servos.py` | **Versión corregida**: misma inferencia TFLite + control de 5 servos por PCA9685 (poses, rampa y cooldown). Base de `src/04_inferencia_tiempo_real.py`. |
| `contexto proyecto logica fuzzy.txt` | Planteamiento del proyecto, fases 1–5 y marco comparativo. |
| `comparacion_propuesta_notebooklm.txt` | Propuesta de **análisis comparativo multidimensional** (benchmarking multiparadigma, complejidad, SHAP, fisiología, Pareto). Base de `src/07_analisis_comparativo.py` y de [`docs/05_analisis_comparativo.md`](../docs/05_analisis_comparativo.md). |

## `material_clase/` (solo local, NO publicado)

16 archivos: 15 PDFs de la disciplina Lógica Fuzzy (PUC-Rio) y el informe final
de trabajo de grado **F-DC-125** en `.docx`. Esta carpeta existe en la copia
local de trabajo, pero no se sube al repositorio remoto.

Los más utilizados en `docs/01_metodologia.md` son:

* `Fuzzy_C_Means_Thayza_final.pdf`
* `Extração de Regras - slides.pdf`
* `Generating Fuzzy Rules by Learning from Examples.pdf`
* `parte_7_sistemas_fuzzy.pdf` (inferencia, defuzzificación, TSK)
* `main.pdf` (controladores fuzzy, TSK en universos discretizados)

> Nota: el artículo de *TinyML Benchmarking* mencionado en el contexto no está
> incluido en la carpeta original; sus cifras se citan solo como antecedente.
