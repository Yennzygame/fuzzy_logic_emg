"""Fase 5: inferencia en vivo FCM + TSK y mapeo a servos PCA9685.

Adaptado de `referencias/scripts_originales/realtime_servos.py`: se conservan
el filtrado causal, el búfer circular, el suavizado por votación, la rampa de
movimiento y el cooldown, pero se sustituye TFLite por las ecuaciones TSK.

Modos:
  --fuente replay  : lee el CSV (sin hardware)
  --fuente myo     : Myo Armband + dongle BLED112 (requiere pyomyo)
  --servos         : activa PCA9685 (requiere adafruit_servokit)
  sin --servos     : modo "dry run", solo imprime la pose que se enviaría
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import unicodedata
from collections import Counter, deque
from queue import Queue

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from config import (
    CHANNELS, CLASSES, DATA_RAW, FS, HOP, MODELS, MOVE_COOLDOWN,
    POSES, RAMP_DT, RAMP_STEP, SERVO_CHANNELS, SMOOTH_K, WINDOW,
    ZC_THRESHOLD,
)
from signal_features import build_filters, hudgins_features


def normalizar_etiqueta(s: str) -> str:
    """Igual que realtime_servos.py: sin tildes, 'ñ' -> 'n', espacios -> '_'."""
    s = (s or "").strip().lower().replace("Ã±", "ñ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("ñ", "n").replace(" ", "_").replace("-", "_")


POSES_NORMALIZADAS = {normalizar_etiqueta(k): v for k, v in POSES.items()}


def cargar_modelo(ruta):
    m = json.loads(ruta.read_text(encoding="utf-8"))
    return {
        "centros": np.asarray(m["centros"], float),
        "P": np.asarray(m["P"], float),
        "q": np.asarray(m["q"], float),
        "mu": np.asarray(m["mu_zscore"], float),
        "sigma": np.asarray(m["sigma_zscore"], float),
        "m": float(m["m"]),
        "clases": m["clases"],
    }


class ProcesadorEMG:
    """Cadena en tiempo real: filtro causal -> ventana -> Hudgins -> FCM+TSK -> voto."""

    def __init__(self, modelo, smooth_k=SMOOTH_K):
        self.modelo = modelo
        sos_bp, notches = build_filters(FS)
        self.filtros = [sos_bp] + list(notches)
        self.estados = [
            np.transpose(np.array([sosfilt_zi(sos) for _ in range(8)], dtype=np.float64), (1, 2, 0))
            for sos in self.filtros
        ]
        self.ring = deque(maxlen=WINDOW)
        self.historial = deque(maxlen=smooth_k)
        self.smooth_k = smooth_k
        self.contador = 0

    def procesar_muestra(self, muestra):
        x = np.asarray(muestra, dtype=np.float64)[None, :]
        for i, sos in enumerate(self.filtros):
            x, self.estados[i] = sosfilt(sos, x, axis=0, zi=self.estados[i])
        self.ring.append(x[0])
        self.contador += 1
        if len(self.ring) < WINDOW or self.contador % HOP != 0:
            return None

        t0 = time.perf_counter_ns()
        ventana = np.asarray(self.ring)[None, :, :]
        features = ((hudgins_features(ventana, ZC_THRESHOLD) - self.modelo["mu"]) /
                    self.modelo["sigma"])
        centros = self.modelo["centros"]
        d = np.maximum(np.sum((features[:, None, :] - centros) ** 2, axis=2), 1e-12)
        u = 1.0 / d ** (1.0 / (self.modelo["m"] - 1.0))
        w = u / u.sum(axis=1, keepdims=True)
        num = np.zeros((1, len(self.modelo["clases"])))
        for i in range(len(w[0])):
            num += w[:, i, None] * (features @ self.modelo["P"][i].T + self.modelo["q"][i])
        scores = (num / np.maximum(w.sum(axis=1, keepdims=True), 1e-12))[0]
        idx = int(np.argmax(scores))
        self.historial.append(idx)
        latencia = (time.perf_counter_ns() - t0) / 1e6

        if len(self.historial) < self.smooth_k:
            return None
        votos = Counter(self.historial)
        ganador, n_votos = votos.most_common(1)[0]
        return {
            "gesto": self.modelo["clases"][ganador],
            "confianza_voto": n_votos / self.smooth_k,
            "puntaje": float(scores[ganador]),
            "latencia_ms": latencia,
        }


# ---------- Servos (misma lógica que el script corregido) ----------
def make_servo_controller():
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)
    for ch in SERVO_CHANNELS:
        kit.servo[ch].actuation_range = 180
        kit.servo[ch].set_pulse_width_range(500, 2500)
    return kit


def rampa(kit, actual, objetivo, paso=RAMP_STEP, dt=RAMP_DT):
    actual, objetivo = [float(v) for v in actual], [float(v) for v in objetivo]
    while True:
        hecho = True
        for i in range(len(actual)):
            if abs(actual[i] - objetivo[i]) > 1e-3:
                hecho = False
                actual[i] += paso if actual[i] < objetivo[i] else -paso
                actual[i] = min(actual[i], objetivo[i]) if actual[i] < objetivo[i] else max(actual[i], objetivo[i])
        if kit is not None:
            for ch, ang in zip(SERVO_CHANNELS, actual):
                kit.servo[ch].angle = float(ang)
        time.sleep(dt)
        if hecho:
            break
    return actual


def main():
    ap = argparse.ArgumentParser(description="Inferencia FCM+TSK en tiempo real")
    ap.add_argument("--fuente", choices=["replay", "myo"], default="replay")
    ap.add_argument("--csv", type=__import__("pathlib").Path, default=DATA_RAW)
    ap.add_argument("--modelo", type=__import__("pathlib").Path,
                    default=MODELS / "modelo_fcm_tsk.json")
    ap.add_argument("--servos", action="store_true", help="Mover servos reales (PCA9685)")
    ap.add_argument("--max-muestras", type=int, default=20000)
    ap.add_argument("--tiempo-real", action="store_true", help="Reproducir CSV a 200 Hz")
    ap.add_argument("--smooth-k", type=int, default=SMOOTH_K)
    args = ap.parse_args()

    modelo = cargar_modelo(args.modelo)
    procesador = ProcesadorEMG(modelo, args.smooth_k)
    print("Modelo:", args.modelo, "| clases:", modelo["clases"])

    kit = make_servo_controller() if args.servos else None
    pose_actual = POSES_NORMALIZADAS["reposo"][:]
    if kit is not None:
        for ch, ang in zip(SERVO_CHANNELS, pose_actual):
            kit.servo[ch].angle = float(ang)

    cola_servos, activo = Queue(maxsize=5), True
    ultimo_gesto, ultimo_movimiento, ultima_impresion = None, 0.0, 0.0

    def trabajador_servos():
        nonlocal pose_actual
        while activo:
            gesto = cola_servos.get()
            if gesto is None:
                break
            objetivo = POSES_NORMALIZADAS.get(gesto)
            if objetivo:
                pose_actual = rampa(kit, pose_actual, objetivo)
                print(f"   -> servos a la pose '{gesto}': {pose_actual}")

    hilo = threading.Thread(target=trabajador_servos, daemon=True)
    hilo.start()

    def emitir(resultado):
        nonlocal ultimo_gesto, ultimo_movimiento, ultima_impresion
        ahora = time.time()
        if ahora - ultima_impresion > 0.2:
            print(f"[{time.strftime('%H:%M:%S')}] {resultado['gesto']:<14} "
                  f"voto={resultado['confianza_voto']:.0%} "
                  f"score={resultado['puntaje']:.3f} "
                  f"inferencia={resultado['latencia_ms']:.4f} ms")
            ultima_impresion = ahora
        g = normalizar_etiqueta(resultado["gesto"])
        if g != ultimo_gesto and (ahora - ultimo_movimiento) > MOVE_COOLDOWN and g in POSES_NORMALIZADAS:
            while not cola_servos.empty():
                cola_servos.get_nowait()
            cola_servos.put_nowait(g)
            ultimo_gesto, ultimo_movimiento = g, ahora

    try:
        if args.fuente == "replay":
            print(f"Reproduciendo {args.csv} (modo {'tiempo real' if args.tiempo_real else 'rápido'})…")
            with open(args.csv, newline="", encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    muestra = [float(fila[c]) for c in CHANNELS]
                    salida = procesador.procesar_muestra(muestra)
                    if salida:
                        emitir(salida)
                    if args.tiempo_real:
                        time.sleep(1 / FS)
                    if procesador.contador >= args.max_muestras:
                        break
        else:
            from pyomyo import Myo, emg_mode
            from serial.tools import list_ports
            puerto = None
            for p in list_ports.comports():
                if "BLED112" in p.description or "2458:0001" in p.hwid:
                    puerto = p.device
                    break
            puerto = puerto or "/dev/ttyACM0"
            datos = Queue()

            def on_emg(emg, moving):
                datos.put_nowait(np.asarray(emg, dtype=np.float64))

            m = Myo(tty=puerto, mode=emg_mode.RAW)
            m.add_emg_handler(on_emg)
            m.connect()
            threading.Thread(target=lambda: [m.run() for _ in iter(int, 1)], daemon=True).start()
            print("Conectado al Myo. Ctrl+C para salir.")
            while True:
                if not datos.empty():
                    salida = procesador.procesar_muestra(datos.get())
                    if salida:
                        emitir(salida)
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nFinalizado por el usuario.")
    finally:
        cola_servos.put_nowait(None)


if __name__ == "__main__":
    main()
