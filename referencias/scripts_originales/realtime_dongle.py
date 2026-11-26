# -- coding: utf-8 --
"""
Inferencia en tiempo real EMG con Myo Armband via DONGLE USB (Bluegiga BLED112)
- Versión con comunicación Serial para máxima estabilidad.
- Raspberry Pi 4 + PySerial + TFLite
"""

import os, json, time, struct, concurrent.futures, threading
import numpy as np
from collections import deque, Counter
from scipy.signal import sosfilt, sosfilt_zi, tf2sos
import serial
from serial.tools import list_ports

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

import asyncio

#######################
# CONFIG
#######################
MODEL  = "models/modelo_emg_tf.tflite"
PARAMS = "models/modelo_emg_params_tf.json"
W, HOP, SMOOTH_K = 150, 20, 7

def load_params():
    with open(PARAMS, "r", encoding="utf-8", errors="replace") as f:
        p = json.load(f)
    return p["classes"], np.array(p["mu"], dtype=np.float32), np.array(p["sig"], dtype=np.float32), \
           tf2sos(p["bp_b"], p["bp_a"]), tf2sos(p["not50_b"], p["not50_a"]), tf2sos(p["not60_b"], p["not60_a"])

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
        self.ser = None

    def find_dongle(self):
        if self.port: return self.port
        ports = list(list_ports.comports())
        for p in ports:
            # PID 2458:0001 is BLED112
            if "BLED112" in p.description or "Bluegiga" in p.description or "2458:0001" in p.hwid:
                return p.device
        # Intentar comunes
        for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]:
            if os.path.exists(p): return p
        return None

async def run_inference():
    if not os.path.isfile(MODEL) or not os.path.isfile(PARAMS):
        print(f"Error: No se encuentran archivos de modelo.")
        return

    classes, mu, sig, bp_sos, n50_sos, n60_sos = load_params()

    print("Cargando TFLite...")
    interpreter = Interpreter(model_path=MODEL, num_threads=4)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    data_queue = asyncio.Queue()
    loop_status = {"active": True}
    
    async def worker():
        """Bucle principal de procesamiento de datos."""
        ring = deque(maxlen=W)
        preds_history = deque(maxlen=SMOOTH_K)
        sample_counter = 0
        last_emit = time.time()
        
        st_bp  = np.transpose(np.array([sosfilt_zi(bp_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))
        st_n50 = np.transpose(np.array([sosfilt_zi(n50_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))
        st_n60 = np.transpose(np.array([sosfilt_zi(n60_sos) for _ in range(8)], dtype=np.float32), (1, 2, 0))

        print("🚀 Procesador de señales listo.")

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
                X_f, st_bp  = sosfilt(bp_sos,  X_batch, axis=0, zi=st_bp)
                X_f, st_n50 = sosfilt(n50_sos, X_f,     axis=0, zi=st_n50)
                X_f, st_n60 = sosfilt(n60_sos, X_f,     axis=0, zi=st_n60)

                for i in range(len(X_f)):
                    ring.append(X_f[i])
                    sample_counter += 1

                    if len(ring) == W and (sample_counter % HOP == 0):
                        X = np.array(ring)
                        Xn = ((X - mu) / (sig + 1e-8)).astype(np.float32)[None, ...]

                        loop = asyncio.get_event_loop()
                        probs = await loop.run_in_executor(executor, run_tflite_inference, interpreter, Xn, in_idx, out_idx)
                        
                        cls = classes[int(np.argmax(probs))]
                        preds_history.append(cls)

                        if len(preds_history) >= SMOOTH_K:
                            vote = Counter(preds_history).most_common(1)[0][0]
                            now = time.time()
                            if now - last_emit > 0.2:
                                status = '✅' if q_size < 15 else '⚠️'
                                print(f"[{time.strftime('%H:%M:%S')}] {vote:15} | Conf: {np.max(probs):.2f} {status}")
                                last_emit = now
            except Exception as e:
                print(f"\n[Error Worker]: {e}")
                await asyncio.sleep(1)

    # Iniciamos el procesador
    asyncio.create_task(worker())

    # --- DRIVER DONGLE ---
    try:
        from pyomyo import Myo, emg_mode
        print("✅ Librería pyomyo detectada. Usando driver de alto nivel.")
        
        def on_emg(emg, moving):
            data_queue.put_nowait(np.array(emg, dtype=np.float32))

        dongle_port = MyoDongle().find_dongle()
        if not dongle_port:
            print("❌ No se encontró el Dongle USB.")
            return

        m = Myo(tty=dongle_port, mode=emg_mode.RAW)
        m.add_emg_handler(on_emg)
        m.connect()
        
        print("✅ Conectado vía Dongle. Esperando datos...")
        while loop_status["active"]:
            m.run()
            await asyncio.sleep(0.001)

    except ImportError:
        print("\n⚠️  No se encontró la librería 'pyomyo'.")
        print("Instálala con: pip install pyomyo")
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(run_inference())
    except KeyboardInterrupt:
        print("\n\n👋 Finalizado por el usuario.")
        os._exit(0)
