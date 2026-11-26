# Metodología y fundamentos

## 1. El problema y por qué lógica fuzzy

La prótesis mioeléctrica debe decidir, cada 250 ms, qué gesto está haciendo el
usuario. Un CNN-BiLSTM alcanza ~99 % de exactitud, pero:

* es una **caja negra**: no se puede auditar por qué falló un movimiento;
* su inferencia (10–35 ms) y su tamaño son restrictivos en un microcontrolador;
* la explicabilidad post-hoc (SHAP) es inconsistente: en el artículo previo,
  TCN y Random Forest atribuían importancia a canales distintos (ρ = 0,262).

La alternativa propuesta es un sistema **intrínsecamente interpretable**: cada
regla es una ecuación que se puede imprimir y revisar con un profesional clínico.

## 2. Señal y espacio de características

* Adquisición: Myo Armband, 8 canales sEMG diferenciales a 200 Hz vía dongle
  BLED112 (`recorder_dongle.py`).
* Filtrado: pasabanda 20–99 Hz (el límite superior de 450 Hz del script original
  se recorta a 99 Hz porque Nyquist = 100 Hz) + notch 50 y 60 Hz.
* Ventanas: 150 muestras = 0,75 s, con salto de 50 muestras = 0,25 s.
  Se segmenta **dentro de cada bloque continuo** de la misma etiqueta para no
  mezclar gestos en una ventana. Con `TRIM=0` se obtienen **2.139 ventanas**;
  con `TRIM=50` (script original) se obtienen 2.125.
* Características de **Hudgins** por canal (8 × 4 = 32):

| Característica | Definición |Qué capta |
| :--- | :--- | :--- |
| RMS | √(media(x²)) | energía de contracción |
| MAV | media(\|x\|) | amplitud media |
| WL | Σ\|x[t+1] − x[t]\| | actividad / fatiga |
| ZC | nº de cambios de signo | frecuencia del patrón |

* Normalización z-score ajustada **solo con entrenamiento** (evita fuga).

## 3. Fase 2: Fuzzy C-Means (antecedentes)

FCM minimiza

$$J_m=\sum_{i=1}^{n}\sum_{k=1}^{c}u_{ik}^{m}\|x_i-v_k\|^2,
\qquad \sum_{k=1}^{c}u_{ik}=1$$

con $m=2$, $c=5$. A diferencia de k-means, un mismo vector puede pertenecer a
varios clusters con grados $0\le u_{ik}\le 1$. Eso modela la co-contracción y la
fatiga: un gesto no "salta" de uno a otro, transita.

Se inicializa con los prototipos medios de cada clase para que las 5 reglas
tengan semántica de gesto. Aun así, FCM converge a **3 centros útiles**: los
pares *mano abierta/pinza* y *puño cerrado/agarre vaso* comparten estadísticas de
Hudgins. Por eso se incluye también una variante con 5 gaussianas ancladas a
cada gesto.

### Índices de validación

$$PC=\frac1n\sum_i\sum_k u_{ik}^2,\qquad
PE=-\frac1n\sum_i\sum_k u_{ik}\ln u_{ik}$$

Ojo: se promedia **por fila**, no por elemento del arreglo. PC cercano a 1
significa partición rígida; PC cercano a 1/c = 0,2 significa superposición
máxima. PE máximo es ln(5) ≈ 1,609. Nuestros valores (0,3454 y 1,2856) indican
superposición moderada-alta, muy cercana a la reportada en el contexto.

## 4. Fase 3: consecuentes TSK de orden 1

Regla $k$:

> SI $x$ pertenece al cluster $k$ ENTONCES $y_k = P_k x + q_k$

donde $y_k$ es un vector de 5 puntajes (uno por gesto). Los parámetros se
estiman por **mínimos cuadrados ponderados**:

$$\theta_k=(A_k^\top W_k A_k + \lambda I)^{-1}A_k^\top W_k Y,
\qquad A=[X\;\;1],\; W_k=\operatorname{diag}(u_{ik}^m)$$

La salida final es la defuzzificación algebraica:

$$\hat y(x)=\frac{\sum_k w_k(x)\,(P_kx+q_k)}{\sum_k w_k(x)}$$

**TSK vs. Mamdani:** Mamdani requiere integrar un centroide sobre áreas difusas;
TSK solo evalúa 5 hiperplanos y una suma ponderada. Por eso la latencia cae del
orden de 10⁻² ms a 10⁻³ ms en C embebido y no necesita universos discretizados.

## 5. Fase 4: evaluación

* Exactitud, F1-macro, matriz de confusión e informe por clase.
* Comparación con k-means rígido, KNN y Random Forest sobre el **mismo split**.
* Latencia medida en Python/NumPy (no comparable con la medición en C del
  artículo). Esta distinción debe explicitarse en cualquier reporte.
* Interpretabilidad: `outputs/models/modelo_fcm_tsk.json` contiene los 5
  centros, las matrices $P_k$ (5×5×32) y los vectores $q_k$ (5×5).

## 6. Fase 5: despliegue

`04_inferencia_tiempo_real.py` conserva la cadena del script original
`realtime_servos.py`: filtros causales con estado `sosfilt_zi`, búfer circular de
150 muestras, inferencia cada `HOP` muestras, votación mayoritaria (`SMOOTH_K=7`),
rampa de servos y cooldown de 0,7 s. Lo único que cambia es el clasificador:
TFLite se sustituye por las ecuaciones FCM+TSK en NumPy.

## 7. Mapa con el material de la carpeta

| Tema | Archivo |
| :--- | :--- |
| Formas de imprecisión y conjuntos fuzzy | `Parte 1`, `Parte 2`, `Parte 3` |
| Hedges y relaciones | `Parte 4`, `Parte 5` |
| Lógica fuzzy y sistemas de inferencia | `Parte 6`, `parte_7_sistemas_fuzzy.pdf` |
| Controladores y defuzzificación | `Controladores Fuzzy.pdf`, `main.pdf` |
| Fuzzy C-Means | `Fuzzy_C_Means_Thayza_final.pdf` |
| Extracción de reglas | `Extração de Regras - slides.pdf`, `Generating Fuzzy Rules…pdf` |
| Antecedente experimental | tesis F-DC-125 y los 4 scripts `.py` |

> Los PDF y la tesis mencionados arriba están en `referencias/material_clase/`,
> una carpeta **solo local** excluida de GitHub. Los scripts `.py` sí se
> publican en `referencias/scripts_originales/`.

## 8. De los resultados a la discusión comparativa

La evaluación (sección 5) responde *qué tan bien funciona* el modelo. La
pregunta central es *por qué esa configuración y no otra*. El análisis
comparativo en [`05_analisis_comparativo.md`](05_analisis_comparativo.md)
amplía esta metodología con cinco ejes que no caben en un informe de resultados:

1. **Multiparadigma:** CNN-1D+BiLSTM, TCN int8, Random Forest, Mamdani, ANFIS y
   FCM+TSK, comparados con una regla explícita: las cifras de literatura y las
   mediciones locales se presentan en planos separados porque sus protocolos no
   son homogéneos.
2. **Complejidad y huella:** ~985 operaciones escalares y 3,85 KB para el TSK,
   frente a 140,4 KB del TCN cuantizado; justificación algebraica de por qué
   TSK sustituye a la defuzzificación por centroide de Mamdani.
3. **Interpretabilidad:** intrínseca (coeficientes dentro de la regla) frente a
   post-hoc (SHAP), con el hallazgo de correlación ρ = 0,262 entre explicadores
   de modelos distintos.
4. **Fisiología:** diafonía, fatiga y co-contracción como causa de la
   superposición difusa y de la confusión `mano_abierta` ↔ `pinza`.
5. **Pareto:** la latencia de inferencia (0,003 ms en C) no es la latencia
   end-to-end; la ventana de 750 ms y la rampa mecánica dominan el tiempo
   percibido (~800 ms).

Ese documento se genera con `python src/07_analisis_comparativo.py` a partir de
los resultados de `src/03_evaluar.py` y de los criterios de comparación
documentados en el propio análisis.
