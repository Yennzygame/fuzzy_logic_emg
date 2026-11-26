# Banco de preguntas probables (Prof. Tiago / defensa)

## 1. ¿Por qué TSK de orden 1 y no Mamdani?

Mamdani defuzzifica con el centroide $\int y\mu(y)dy/\int\mu(y)dy$, lo que
exige discretizar el universo de salida en cada ciclo. En TSK la salida ya es
numérica: $\hat y=\sum w_i(p_ix+q_i)/\sum w_i$. Es álgebra lineal $O(c\cdot d)$
(5 reglas × 32 características), sin integración geométrica. Por eso la latencia
cae del orden de 0,018 ms (Mamdani reportado) a 0,003 ms en C embebido.

## 2. ¿Qué significan PC = 0,3454 y PE = 1,2856?

PC mide la "dureza" de la partición: 1 = partición rígida, 1/c = 0,2 =
superposición máxima. PE mide entropía: 0 = certeza total, ln 5 = 1,609 =
incertidumbre máxima. Nuestros valores están entre ambos extremos: hay
pertenencia simultánea a varios gestos, coherente con co-contracción y fatiga.
Se calculan **por fila** de la matriz $U$; dividir por el total de elementos
daría valores artificialmente bajos.

## 3. Los números del contexto eran PC 0,3905 / PE 1,1849 y 97,66 %. ¿Por qué no coinciden?

Porque son configuraciones distintas (espacio de características, recorte de
transiciones, inicialización y split). Esta carpeta es una **reproducción
auditable**: publica el dataset, el código y los resultados reales. La diferencia
es pequeña y se documenta; además se incluye el barrido de `m`
(`outputs/metrics/barrido_m.csv`) para mostrar cómo cambian PC y PE.

## 4. FCM convergió a 3 centros. ¿No eran 5 reglas?

Sí: 5 reglas, pero dos pares de gestos tienen medias de Hudgins casi idénticas
y FCM los fusiona. Es un hallazgo, no un error: demuestra que los prototipos
solos no separan esos gestos, mientras que los consecuentes lineales sí lo
logran. Por eso se añadió la variante con 5 gaussianas por gesto (98,28 %).

## 5. ¿La exactitud es comparable con la del artículo de TinyML?

Solo parcialmente. El split 70/30 es aleatorio por ventanas y hay ventanas
solapadas del mismo bloque en train y test (fuga de información). El valor es
optimista; para un reporte clínico se necesitaría separar por sesión/sujeto.
Además, la latencia de Python/NumPy (~0,18 ms) no es comparable con la de C
embebido (0,003 ms).

## 6. ¿Dónde está la interpretabilidad si hay 32 coeficientes por regla?

En que cada regla es un especialista local con ecuación explícita. Se puede
inspeccionar `outputs/models/modelo_fcm_tsk.json` y ver qué canal/característica
empuja la decisión (`importancia_canales.csv` y la figura 05). No se necesita un
explicador externo: a diferencia de SHAP, la atribución no depende de un
algoritmo post-hoc.

## 7. ¿Qué aporta el trabajo de grado F-DC-125?

El protocolo experimental validado: pulsera Myo, 8 canales a 200 Hz, Raspberry
Pi, mano robótica de 5 servos MG90S con PCA9685 y el pipeline CNN-1D + BiLSTM
(99,59 % offline). Ese trabajo aporta el hardware y los datos; este proyecto
sustituye la caja negra por un modelo auditable y ligero.

## 8. Si el profesor pide "hazlo andar en la prótesis", ¿qué respondo?

`04_inferencia_tiempo_real.py --fuente myo --servos` mantiene el filtrado
causal, el búfer, el voto mayoritario, la rampa y el cooldown del script
corregido `realtime_servos.py`, y cambia solo el intérprete TFLite por las
ecuaciones TSK. Antes de mover motores, se valida en modo `replay` y en
`--fuente myo` sin `--servos`.

## 9. ¿Cómo comparas tu modelo con la literatura si usaste otro protocolo?

No mezclo las cifras. En [`05_analisis_comparativo.md`](05_analisis_comparativo.md)
separo dos planos: **literatura** (CNN-BiLSTM, TCN int8, RF en C, Mamdani,
ANFIS; sirven para mapear el espacio de diseño) y **reproducción local** (mismo
dataset y mismo split 70/30; sirven para comparar modelos entre sí). Decir
"supero a X" usando cifras de otro artículo sería una comparación inválida.

## 10. ¿Cuál es la ventaja real de TSK frente a Mamdani si ambos son difusos?

Mamdani defuzzifica con el centroide $\int y\mu(y)dy/\int\mu(y)dy$ sobre un
universo discretizado: costo $O(RN)$ y acceso a memoria. TSK sustituye esa
integración por una media ponderada de hiperplanos. En este trabajo: ~985
operaciones escalares y 3,85 KB de parámetros. Es el mismo argumento de las
notas *Parte 7* y `main.pdf`, aplicado a sEMG real.

## 11. ¿Qué es eso de la correlación ρ = 0,262 y por qué importa?

En el artículo de TinyML, SHAP explicaba el TCN y el Random Forest con
importancias de canal distintas: Spearman ρ = 0,262. Es decir, el explicador
post-hoc producía una explicación dependiente del método, no del fenómeno.
Nuestro modelo es **intrínsecamente interpretable**: la atribución son los
coeficientes $P_k$ dentro de cada regla, sin un segundo algoritmo que pueda
contradecirla. Aun así, transparencia ≠ corrección fisiológica: hace falta
validación clínica.

## 12. Si tu inferencia tarda 0,18 ms, ¿por qué dices que la prótesis responde en ~800 ms?

Porque 0,18 ms es solo el clasificador. El sistema debe llenar una ventana de
150 muestras a 200 Hz = **750 ms** antes de decidir, y luego la rampa de los
servos añade ~65 ms en el peor caso. El aporte de TSK es eliminar el cuello de
botella computacional para que el límite lo ponga la mecánica, no el modelo.
Reducir la ventana sin perder exactitud es el siguiente frente de trabajo.

## 13. ¿Por qué ANFIS reporta 0,001 ms y tú no lo implementaste así?

ANFIS con grid partition puede explotar el número de reglas en 32 dimensiones.
Por eso usamos FCM con 5 reglas y consecuentes estimados por mínimos cuadrados
ponderados. El análisis compara enfoques Wang & Mendel, ANFIS, NEFCON y este
trabajo en una tabla específica, y marca como trabajo futuro implementar ANFIS
y Mamdani **sobre este mismo dataset** para una comparación estadísticamente
homogénea.
