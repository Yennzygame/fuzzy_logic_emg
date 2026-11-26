# -- coding: utf-8 --
"""
realtime_servos.py
Inferencia en tiempo real EMG con Myo Armband via DONGLE USB (Bluegiga BLED112)
+ Control de 5 servomotores usando PCA9685 (I2C) en Raspberry Pi.

Flujo:
- Carga modelo TFLite + parÃ¡metros (mu/sig + filtros)
- Lee EMG en vivo (pyomyo + dongle)
- Ventana W, salto HOP, suavizado SMOOTH_K -> "vote" (gesto final)
- Cuando el gesto cambia (estable), mueve servos a la "pose" asignada

Notas importantes:
- Alimenta los servos con fuente externa (5V) y une GND con Raspberry/PCA.
- Ajusta POSES y el orden de SERVOS segÃºn tu mano/mecÃ¡nica.
"""

import os, json, time, concurrent.futures, threading, unicodedata
import numpy as np
from collections import deque, Counter
from queue import Queue

from scipy.signal import sosfilt, sosfilt_zi, tf2sos
from serial.tools import list_ports

import asyncio

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from adafruit_servokit import ServoKit


#######################
# CONFIG IA
#######################
MODEL  = "models/modelo_emg_tf.tflite"
PARAMS = "models/modelo_emg_params_tf.json"
W, HOP, SMOOTH_K = 150, 20, 7

#######################
# CONFIG SERVOS (PCA9685)
#######################
SERVOS = [0, 1, 2, 3, 4]  # Canales PCA9685 usados
MOVE_COOLDOWN = 0.7       # s: evita re-disparos muy frecuentes

# Velocidad/agresividad del movimiento
RAMP_STEP = 8             # grados por paso
RAMP_DT   = 0.005         # s entre pasos

# Ajuste de pulsos (rango tÃ­pico)
PULSE_MIN_US = 500
PULSE_MAX_US = 2500

# Poses por gesto (5 Ã¡ngulos = [servo0..servo4])
# AJUSTA ESTO a tu mano real (pulgar/Ã­ndice/medio/anular/meÃ±ique) y sentido de giro.
# Orden: [menique, anular, medio, indice, pulgar]
OPEN_HAND = [10, 10, 10, 10, 10]         # mano abierta
REST_HAND = [50, 50, 50, 50, 50]         # reposo (m�s natural)
CLOSE_HAND = [150,150,150,150,150]       # pu�o
PINZA_HAND = [10, 10, 10, 150, 150]      # indice+pulgar
VASO_HAND  = [120,120,120,120,130]       # agarre vaso


# Orden: [menique, anular, medio, indice, pulgar]
POSES = {
    "reposo":       REST_HAND,
    "mano_abierta": OPEN_HAND,
    "puno_cerrado": CLOSE_HAND,
    "punio_cerrado": CLOSE_HAND,
    "pinza":        PINZA_HAND,
    "agarre_vaso":  VASO_HAND,
    "vaso":         VASO_HAND,
}




def normalize_label(s: str) -> str:
    """Normaliza: minúsculas, arregla 'Ã±', quita tildes, espacios -> '_'."""
    s = (s or "").strip().lower()

    # Arreglos por codificación rota
    s = s.replace("Ã±", "ñ").replace("Ã", "ñ")

    # Normalización de tildes
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # ñ -> n, y formato
    s = s.replace("ñ", "n")
    s = s.replace(" ", "_").replace("-", "_")
    return s



def load_params():
    with open(PARAMS, "r", encoding="utf-8", errors="replace") as f:
        p = json.load(f)
    return (
        p["classes"],
        np.array(p["mu"], dtype=np.float32),
        np.array(p["sig"], dtype=np.float32),
        tf2sos(p["bp_b"], p["bp_a"]),
        tf2sos(p["not50_b"], p["not50_a"]),
        tf2sos(p["not60_b"], p["not60_a"]),
    )


# El ejecutor de hilos permite que la IA corra en segundo plano
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_tflite_inference(interpreter, input_data, in_idx, out_idx):
    """Ejecuta la IA en un hilo separado."""
    interpreter.set_tensor(in_idx, input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0]


# --- Protocolo Myo Dongle (BLED112) ---
class MyoDongle:
    def __init__(self, port=None):
        self.port = port

    def find_dongle(self):
        if self.port:
            return self.port
        ports = list(list_ports.comports())
        for p in ports:
            # PID 2458:0001 is BLED112
            if "BLED112" in p.description or "Bluegiga" in p.description or "2458:0001" in p.hwid:
                return p.device
        # Intentar comunes
        for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]:
            if os.path.exists(p):
                return p
        return None


#######################
# SERVOS: utilidades
#######################
def make_servo_controller():
    kit = ServoKit(channels=16)
    for ch in SERVOS:
        kit.servo[ch].actuation_range = 180
        kit.servo[ch].set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
    return kit


def set_all(kit, angles):
    for ch, ang in zip(SERVOS, angles):
        kit.servo[ch].angle = float(ang)


def ramp_all(kit, start, target, step=RAMP_STEP, dt=RAMP_DT):
    """Rampa sincronizada: actualiza los 5 servos por cada paso."""
    cur = [float(x) for x in start]
    tgt = [float(x) for x in target]

    done = False
    while not done:
        done = True
        for i in range(len(cur)):
            if abs(cur[i] - tgt[i]) > 0.001:
                done = False
                if cur[i] < tgt[i]:
                    cur[i] = min(cur[i] + step, tgt[i])
                else:
                    cur[i] = max(cur[i] - step, tgt[i])

        set_all(kit, cur)
        time.sleep(dt)


async def run_inference():
    if not os.path.isfile(MODEL) or not os.path.isfile(PARAMS):
        print("Error: No se encuentran archivos de modelo.")
        return

    classes, mu, sig, bp_sos, n50_sos, n60_sos = load_params()

    print("Cargando TFLite...")
    interpreter = Interpreter(model_path=MODEL, num_threads=4)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    data_queue = asyncio.Queue()
    loop_status = {"active": True}

    # =========================
    # SERVO SETUP (PCA9685)
    # =========================
    kit = make_servo_controller()

    # Estado actual y cola de comandos
    current_pose = POSES["reposo"][:]
    set_all(kit, current_pose)

    servo_cmd_q = Queue(maxsize=5)

    def servo_worker():
        nonlocal current_pose
        while loop_status["active"]:
            gesture = servo_cmd_q.get()
            if gesture is None:
                break
            g = normalize_label(gesture)
            if g not in POSES:
                continue
            target = POSES[g]
            try:
                ramp_all(kit, current_pose, target)
                current_pose = target[:]
            except Exception as e:
                print(f"[ServoError] {e}")

    t_servo = threading.Thread(target=servo_worker, daemon=True)
    t_servo.start()

    async def worker():
        """Bucle principal de procesamiento de datos."""
        ring = deque(maxlen=W)
        preds_history = deque(maxlen=SMOOTH_K)
        sample_counter = 0
        last_emit = time.time()

        last_sent_gesture = None
        last_move_t = 0.0

        st_bp = np.transpose(np.array([sosfilt_zi(bp_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))
        st_n50 = np.transpose(np.array([sosfilt_zi(n50_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))
        st_n60 = np.transpose(np.array([sosfilt_zi(n60_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))

        print("ð Procesador de seÃ±ales listo (IA + Servos).")

        while loop_status["active"]:
            try:
                q_size = data_queue.qsize()
                if q_size == 0:
                    await asyncio.sleep(0.01)
                    continue

                batch = []
                for _ in range(min(q_size, 40)):
                    batch.append(await data_queue.get())

                X_batch = np.array(batch, dtype=np.float32)
                X_f, st_bp = sosfilt(bp_sos, X_batch, axis=0, zi=st_bp)
                X_f, st_n50 = sosfilt(n50_sos, X_f, axis=0, zi=st_n50)
                X_f, st_n60 = sosfilt(n60_sos, X_f, axis=0, zi=st_n60)

                for i in range(len(X_f)):
                    ring.append(X_f[i])
                    sample_counter += 1

                    if len(ring) == W and (sample_counter % HOP == 0):
                        X = np.array(ring)
                        Xn = ((X - mu) / (sig + 1e-8)).astype(np.float32)[None, ...]

                        loop = asyncio.get_event_loop()
                        probs = await loop.run_in_executor(
                            executor, run_tflite_inference, interpreter, Xn, in_idx, out_idx
                        )

                        cls = classes[int(np.argmax(probs))]
                        preds_history.append(cls)

                        if len(preds_history) >= SMOOTH_K:
                            vote = Counter(preds_history).most_common(1)[0][0]
                            now = time.time()

                            # Log cada 0.2 s
                            if now - last_emit > 0.2:
                                status = "â" if q_size < 15 else "â ï¸"
                                print(f"[{time.strftime('%H:%M:%S')}] {vote:15} | Conf: {np.max(probs):.2f} {status}")
                                last_emit = now

                            # Disparo de servos (solo si cambia y pasa cooldown)
                            v_norm = normalize_label(vote)
                            if v_norm != last_sent_gesture and (now - last_move_t) > MOVE_COOLDOWN:
                                # Si el gesto no estÃ¡ mapeado, no movemos
                                if v_norm in POSES:
                                    # Vaciar cola para quedarnos con el gesto mÃ¡s reciente
                                    while not servo_cmd_q.empty():
                                        try:
                                            servo_cmd_q.get_nowait()
                                        except Exception:
                                            break
                                    try:
                                        servo_cmd_q.put_nowait(v_norm)
                                        last_sent_gesture = v_norm
                                        last_move_t = now
                                    except Exception:
                                        pass

            except Exception as e:
                print(f"\n[Error Worker]: {e}")
                await asyncio.sleep(1)

    # Iniciamos el procesador
    asyncio.create_task(worker())

    # --- DRIVER DONGLE ---
    try:
        from pyomyo import Myo, emg_mode
        print("â LibrerÃ­a pyomyo detectada. Usando driver de alto nivel.")

        def on_emg(emg, moving):
            data_queue.put_nowait(np.array(emg, dtype=np.float32))

        dongle_port = MyoDongle().find_dongle()
        if not dongle_port:
            print("â No se encontrÃ³ el Dongle USB.")
            loop_status["active"] = False
            servo_cmd_q.put_nowait(None)
            return

        m = Myo(tty=dongle_port, mode=emg_mode.RAW)
        m.add_emg_handler(on_emg)
        m.connect()

        print("â Conectado vÃ­a Dongle. Esperando datos...")
        while loop_status["active"]:
            m.run()
            await asyncio.sleep(0.001)

    except ImportError:
        print("\nâ ï¸  No se encontrÃ³ la librerÃ­a 'pyomyo'.")
        print("InstÃ¡lala con: pip install pyomyo")
        await asyncio.sleep(5)

    finally:
        loop_status["active"] = False
        try:
            servo_cmd_q.put_nowait(None)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(run_inference())
    except KeyboardInterrupt:
        print("\n\nð Finalizado por el usuario.")
        os._exit(0)
