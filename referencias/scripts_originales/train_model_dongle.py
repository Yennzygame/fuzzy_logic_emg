# -- coding: utf-8 --
"""
Entrenamiento EMG (fix TFLite):
- CNN-1D + BiLSTM(unroll=True) + rama de envolvente
- tf.data con steps_per_epoch (progreso visible)
- Exporta:
  1) modelo_emg_tf.tflite (FP16)  ← vía LSTM unrolled (recomendado)
  2) modelo_emg_tf_select_ops.tflite (fallback con Select TF Ops)
- Guarda modelo_emg_params_tf.json
"""

import os, json, time, random, warnings, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, sos2tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from collections import Counter

# ========= CONFIG =========
SEED = 13
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

FS   = 200
W    = 150           # 0.75 s
HOP  = 50
TRIM = int(0.25*FS)  # recortar transiciones

DATA = "./dataset_emg/dataset_completo.csv"

EPOCHS = 60
BATCH  = 128
CKPT   = "models/best_emg.keras"

# ---- modo rápido para validar (pon False para entreno completo) ----
FAST_DEBUG = False
MAX_TR, MAX_VA, MAX_TE = 2000, 500, 500
STEPS_PER_EPOCH_CAP = 600
VAL_STEPS_CAP       = 200

# ========= UTILIDADES =========
def log(step, msg): print(f"[{step}] {msg}", flush=True)

def require_cols(df, req):
    for c in req:
        if c not in df.columns: raise ValueError(f"Falta columna {c} en el CSV.")

def bandpass_sos(x, fs, f1=20, f2=None, order=4):
    if f2 is None: f2 = min(450, fs/2 - 1)
    sos = butter(N=order, Wn=[f1/(fs/2), f2/(fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, x, axis=0), sos

def notch_sos(x, fs, f0=50, q=35):
    b, a = iirnotch(w0=f0/(fs/2), Q=q); sos = tf2sos(b, a)
    return sosfiltfilt(sos, x, axis=0), sos

def filter_emg(x, fs):
    y, bp_sos   = bandpass_sos(x, fs)
    y, n50_sos  = notch_sos(y, fs, 50, 35)
    y, n60_sos  = notch_sos(y, fs, 60, 35)
    bp_b,  bp_a  = sos2tf(bp_sos)
    n50_b, n50_a = sos2tf(n50_sos)
    n60_b, n60_a = sos2tf(n60_sos)
    return y, (bp_b.tolist(), bp_a.tolist(),
               n50_b.tolist(), n50_a.tolist(),
               n60_b.tolist(), n60_a.tolist())

def window_blocks_trim(X, G, W, hop, trim):
    G = np.asarray(G).astype(str); N = len(G)
    change = np.ones(N, dtype=bool); change[1:] = (G[1:] != G[:-1])
    bS = np.where(change)[0]; bE = np.r_[bS[1:], N] - 1
    Xs, Ys = [], []
    for s, e in zip(bS, bE):
        s2 = min(max(s + trim, s), e); e2 = max(min(e - trim, e), s2)
        seg = X[s2:e2+1, :]; 
        if seg.shape[0] < W: continue
        lab = G[s]; L = seg.shape[0]
        for i in range(0, L - W + 1, hop):
            Xs.append(seg[i:i+W, :].astype(np.float32)); Ys.append(lab)
    return Xs, Ys

def stratified_cap(X, y, kmax):
    if not FAST_DEBUG or kmax <= 0 or len(X) <= kmax: return X, y
    X = np.array(X, dtype=object); y = np.array(y)
    outX, outy = [], []; classes = np.unique(y); per = max(1, kmax // len(classes))
    for c in classes:
        idx = np.where(y==c)[0]; sel = np.random.permutation(idx)[:min(per, len(idx))]
        outX += list(X[sel]); outy += list(y[sel])
    if len(outX) < kmax:
        rest = np.setdiff1d(np.arange(len(X)), np.array([], dtype=int))
        extra = np.random.permutation(rest)[:(kmax - len(outX))]
        outX += list(X[extra]); outy += list(y[extra])
    return outX, outy

# Estratificación robusta
def stratified_split_indices(Y, train=0.70, val=0.15, seed=SEED):
    rng = np.random.RandomState(seed)
    Y = np.asarray(Y)
    tr_idx, va_idx, te_idx = [], [], []
    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(train * n))
        nva = int(round(val * n))
        tr_idx.extend(idx[:ntr])
        va_idx.extend(idx[ntr:ntr+nva])
        te_idx.extend(idx[ntr+nva:])
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)
    return tr_idx, va_idx, te_idx

# ---- Modelo: BiLSTM con unroll=True (TFLite-friendly) ----
def build_cnn_bilstm_envelope(nc, W=150, C=8, ls=0.05):
    inp = layers.Input(shape=(W, C))
    x = layers.Conv1D(32, 7, padding="same", use_bias=False)(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x);   x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    env = layers.Lambda(lambda t: tf.abs(t))(inp)
    env = layers.AveragePooling1D(pool_size=5, strides=1, padding="same")(env)
    env = layers.Conv1D(16, 3, padding="same", activation="relu")(env)
    f = layers.Concatenate()([x, env])
    f = layers.Bidirectional(layers.LSTM(64, return_sequences=False, unroll=True))(f)
    f = layers.Dropout(0.3)(f)
    out = layers.Dense(nc, activation="softmax")(f)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls),
              metrics=["accuracy"])
    return m

# ========= MAIN =========
def main():
    t0 = time.time()

    # 0) Asegurar carpeta models
    if not os.path.exists("models"):
        os.makedirs("models")

    # 1) Leer CSV
    log("1/12", f"Leyendo CSV: {DATA}")
    if not os.path.isfile(DATA):
        raise FileNotFoundError(f"No encuentro {DATA}")
    df = pd.read_csv(DATA, memory_map=True)
    require_cols(df, [f"CH{i}" for i in range(1,9)] + ["G"])
    log("1/12", f"Dataset shape = {df.shape}")

    # 2) Etiquetas
    log("2/12", "Normalizando etiquetas…")
    G = df["G"].astype(str).str.strip().str.lower()
    G = G.replace({
        "puno_cerrado": "puño_cerrado",
        "puño cerrado": "puño_cerrado",
        "mano abierta": "mano_abierta",
        "agarre vaso": "agarre_vaso",
    })
    print("Etiquetas limpias (conteo crudo):", Counter(G.values))

    # 3) Señal cruda
    Xraw = df[[f"CH{i}" for i in range(1,9)]].values.astype(np.float64); del df

    # 4) Filtrado
    log("3/12", "Filtrando EMG (20–450 + notch 50/60)…")
    Xf, (bp_b, bp_a, n50_b, n50_a, n60_b, n60_a) = filter_emg(Xraw, FS); del Xraw

    # 5) Ventaneo
    log("4/12", f"Ventaneando (W={W}, HOP={HOP}, TRIM={TRIM/FS:.2f}s)…")
    Xall, Yall = window_blocks_trim(Xf, G.values, W, HOP, TRIM)
    total_win = len(Xall); log("4/12", f"Total de ventanas: {total_win:,}")
    print("Etiquetas por ventana:", Counter(Yall))

    # 6) Split estratificado
    log("5/12", "Split 70/15/15 (estratificado)…")
    tr, va, te = stratified_split_indices(Yall, train=0.70, val=0.15, seed=SEED)
    XTr = [Xall[i] for i in tr]; YTr = [Yall[i] for i in tr]
    XVa = [Xall[i] for i in va]; YVa = [Yall[i] for i in va]
    XTe = [Xall[i] for i in te]; YTe = [Yall[i] for i in te]

    print("Tamaños -> Train:", len(XTr), " Val:", len(XVa), " Test:", len(XTe))

    # 7) Clases
    log("6/12", "Codificando clases…")
    classes = sorted(pd.unique(Yall))
    le = LabelEncoder(); le.fit(classes)
    nc = len(classes)

    ytr = le.transform(YTr)
    yva = le.transform(YVa)
    yte = le.transform(YTe)

    log("6/12", f"Clases ({nc}): {list(le.classes_)}")

    # 8) (Opcional) recorte para debug
    if FAST_DEBUG:
        XTr, ytr = stratified_cap(XTr, ytr, MAX_TR)
        XVa, yva = stratified_cap(XVa, yva, MAX_VA)
        XTe, yte = stratified_cap(XTe, yte, MAX_TE)
        log("7/12", f"[DEBUG] Ventanas -> Train:{len(XTr)} Val:{len(XVa)} Test:{len(XTe)}")

    # 9) Normalización
    log("8/12", "Normalizando (μ/σ de TRAIN)…")
    allTr = np.concatenate(XTr, axis=0).astype(np.float32)
    mu = allTr.mean(axis=0).astype(np.float32)
    sg = (allTr.std(axis=0) + 1e-8).astype(np.float32)

    def z(lst):
        if len(lst) == 0:
            return np.empty((0, W, 8), dtype=np.float32)
        A = np.stack(lst, axis=0).astype(np.float32)
        return (A - mu.reshape(1,1,8)) / (sg.reshape(1,1,8))

    Xtr = z(XTr); Xv = z(XVa); Xt = z(XTe)
    Ytr_oh = tf.one_hot(ytr, nc).numpy().astype(np.float32)
    Yv_oh  = tf.one_hot(yva, nc).numpy().astype(np.float32)

    # 10) Modelo + class weights
    log("9/12", "Compilando modelo…")
    cnt = np.bincount(ytr, minlength=nc).astype(np.float32)
    w = (1.0 / np.maximum(cnt, 1)); w = (w / w.sum()) * len(w)
    class_weight = {i: float(w[i]) for i in range(nc)}
    model = build_cnn_bilstm_envelope(nc, W=W, C=8, ls=0.05)

    cbs = [
        callbacks.ModelCheckpoint(CKPT, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # 11) tf.data + progreso por pasos
    log("10/12", "Preparando tf.data…")
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr_oh)).shuffle(8192, seed=SEED).repeat().batch(BATCH).prefetch(tf.data.AUTOTUNE)
    ds_va = tf.data.Dataset.from_tensor_slices((Xv,  Yv_oh)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    steps_full = max(1, math.ceil(len(Xtr) / BATCH))
    val_steps  = max(1, min(math.ceil(len(Xv) / BATCH), VAL_STEPS_CAP))
    steps_per_epoch = min(STEPS_PER_EPOCH_CAP, steps_full)
    print(f"steps_full={steps_full}, using steps_per_epoch={steps_per_epoch}, val_steps={val_steps}", flush=True)

    log("11/12", f"Entrenando {EPOCHS} epochs…")
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
              class_weight=class_weight, verbose=1, callbacks=cbs)

    # 12) Evaluación + guardados
    log("12/12", "Evaluando en TEST…")
    if len(Xt) > 0:
        yhat = model.predict(Xt, batch_size=BATCH, verbose=0).argmax(1)
        print("\nReporte TEST:\n", classification_report(yte, yhat, target_names=classes, digits=3))
        print("Matriz de confusión:\n", confusion_matrix(yte, yhat))
    else:
        print("⚠️ No hay ventanas en TEST para evaluar.")

    model.export("models/modelo_emg_tf_saved")
    model.save("models/modelo_emg_tf.h5", include_optimizer=False)

    # ---------- Export TFLite: Opción A (FP16 normal) ----------
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        tflite = conv.convert()
        open("models/modelo_emg_tf.tflite","wb").write(tflite)
        print("💾 TFLite FP16 guardado: models/modelo_emg_tf.tflite", flush=True)
    except Exception as e:
        print("⚠ TFLite FP16 falló. Intentando Select TF Ops…", e, flush=True)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv._experimental_lower_tensor_list_ops = False
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite2 = conv.convert()
        open("models/modelo_emg_tf_select_ops.tflite","wb").write(tflite2)
        print("💾 TFLite (Select TF Ops) guardado: models/modelo_emg_tf_select_ops.tflite", flush=True)

    params = {
        "classes": classes,
        "mu": mu.tolist(), "sig": sg.tolist(),
        "fs": FS, "windowSize": W, "hop": HOP,
        "bp_b": bp_b, "bp_a": bp_a,
        "not50_b": n50_b, "not50_a": n50_a,
        "not60_b": n60_b, "not60_a": n60_a
    }
    json.dump(params, open("models/modelo_emg_params_tf.json","w"), indent=2, ensure_ascii=False)
    print(f"\n✅ Fin en {time.time()-t0:.1f} s. Artefactos listos.", flush=True)

if __name__ == "__main__":
    main()