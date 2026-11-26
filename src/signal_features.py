"""Filtrado, ventaneo y características de Hudgins (RMS, MAV, WL, ZC).

Orden de las 32 características: CH1_RMS, CH1_MAV, CH1_WL, CH1_ZC, CH2_RMS, ...
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from config import (
    BAND_HIGH, BAND_LOW, FILTER_ORDER, FS, NOTCHES, NOTCH_Q,
)


def build_filters(fs: int = FS):
    """Devuelve (sos_pasabanda, [sos_notch_50, sos_notch_60])."""
    nyq = fs / 2.0
    bp = butter(
        FILTER_ORDER,
        [BAND_LOW / nyq, min(BAND_HIGH, nyq - 1.0) / nyq],
        btype="band", output="sos",
    )
    notches = [tf2sos(*iirnotch(f / nyq, NOTCH_Q)) for f in NOTCHES if f < nyq]
    return bp, notches


def filter_emg(signal: np.ndarray, filters) -> np.ndarray:
    """Filtrado bidireccional (sin desplazamiento de fase) de (muestras, 8)."""
    sos_bp, sos_notches = filters
    y = sosfiltfilt(sos_bp, signal, axis=0)
    for sos in sos_notches:
        y = sosfiltfilt(sos, y, axis=0)
    return y


def segment_blocks(signal, labels, window: int, hop: int, trim: int = 0):
    """Ventanas dentro de cada bloque continuo de la misma etiqueta.

    Devuelve X (n, window, 8), y (n,), group (n,) = índice del bloque.
    """
    labels = np.asarray(labels).astype(str)
    n = len(labels)
    change = np.ones(n, dtype=bool)
    change[1:] = labels[1:] != labels[:-1]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:], n] - 1

    windows, targets, groups = [], [], []
    for g, (s, e) in enumerate(zip(starts, ends)):
        s2 = min(max(s + trim, s), e)
        e2 = max(min(e - trim, e), s2)
        segment = signal[s2:e2 + 1]
        if len(segment) < window:
            continue
        for i in range(0, len(segment) - window + 1, hop):
            windows.append(segment[i:i + window].astype(np.float32))
            targets.append(labels[s])
            groups.append(g)
    return np.stack(windows), np.asarray(targets), np.asarray(groups)


def hudgins_features(windows: np.ndarray, zc_threshold: float = 0.0) -> np.ndarray:
    """(n, 150, 8) -> (n, 32) con RMS, MAV, WL y ZC por canal."""
    x = np.asarray(windows, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2, axis=1))
    mav = np.mean(np.abs(x), axis=1)
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1)
    diff = np.diff(x, axis=1)
    zc = np.sum((x[:, :-1, :] * x[:, 1:, :] < 0) & (np.abs(diff) >= zc_threshold), axis=1)
    return np.stack([rms, mav, wl, zc], axis=2).reshape(len(x), -1).astype(np.float32)


def stratified_split(y, train_fraction: float = 0.70, seed: int = 13):
    """Índices de entrenamiento / prueba estratificados por clase."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = int(round(train_fraction * len(idx)))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.asarray(train_idx), np.asarray(test_idx)


def normalize(train, *others):
    """z-score ajustado únicamente con la media/desviación de entrenamiento."""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0) + 1e-8
    return ((train - mu) / sigma, *[(o - mu) / sigma for o in others]), mu, sigma
