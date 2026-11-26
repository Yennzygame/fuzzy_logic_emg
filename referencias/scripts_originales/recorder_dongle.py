# -*- coding: utf-8 -*-
"""
Captura de señales EMG del Myo Armband via DONGLE USB (Bluegiga)
-----------------------------------------------------------------------------
- Versión para scripts_dongle usando pyomyo
- Mayor estabilidad para Raspberry Pi
"""

import os, sys, time, csv, threading
import numpy as np
from pyomyo import Myo, emg_mode

# ================== CONFIGURACIÓN ==================
OUT_CSV = "./dataset_emg/dataset_completo.csv"
TIEMPO_CAPTURA = 60  
GESTOS = ["mano_abierta", "puño_cerrado", "pinza", "agarre_vaso", "reposo"]

# ================== UTILIDADES ==================
def ensure_header(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"CH{i}" for i in range(1, 9)] + ["G"])

class Recorder:
    def __init__(self):
        self.filas = []
        self.capturando = False
        self.gesto_actual = ""
        self.muestras = 0

    def on_emg(self, emg, moving):
        if self.capturando:
            self.filas.append(list(emg) + [self.gesto_actual])
            self.muestras += 1

    def capturar(self, gesto, segundos):
        self.filas = []
        self.muestras = 0
        self.gesto_actual = gesto
        
        print(f"\n🚀 Iniciando captura de '{gesto}' por {segundos}s...")
        print("Mantén el gesto firme.")
        
        for i in range(3, 0, -1):
            print(f"⏳ {i}...")
            time.sleep(1)
        
        print("🎬 ¡GRABANDO!")
        self.capturando = True
        start = time.time()
        
        try:
            while (time.time() - start) < segundos:
                elapsed = time.time() - start
                remaining = segundos - elapsed
                freq = self.muestras / (elapsed + 0.1)
                print(f"\r⏳ {int(remaining)}s restantes | Muestras: {self.muestras} | Freq: {freq:.1f}Hz", end="")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido por el usuario.")
        
        self.capturando = False
        print(f"\n✅ Captura finalizada. Total: {len(self.filas)} muestras.")
        
        if self.filas:
            ensure_header(OUT_CSV)
            with open(OUT_CSV, "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.filas)
            print(f"💾 Datos guardados en {OUT_CSV}")

def main():
    print("="*50)
    print("      RECORDER MYO - DONGLE VERSION")
    print("="*50)
    
    recorder = Recorder()
    
    # Intentar detectar puerto
    from serial.tools import list_ports
    port = None
    for p in list_ports.comports():
        if "2458:0001" in p.hwid or "BLED112" in p.description:
            port = p.device
            break
    
    if not port:
        port = "/dev/ttyACM0" # Default
        print(f"⚠️ Dongle no detectado, usando default: {port}")
    else:
        print(f"✅ Dongle detectado en: {port}")

    try:
        m = Myo(tty=port, mode=emg_mode.RAW)
        m.add_emg_handler(recorder.on_emg)
        m.connect()
        
        # Hilo para correr el Myo
        def run_myo():
            while True:
                m.run()
        
        t = threading.Thread(target=run_myo, daemon=True)
        t.start()
        
        while True:
            print("\n" + "-"*30)
            print("MENÚ DE CAPTURA")
            for i, g in enumerate(GESTOS, 1):
                print(f"{i} - {g}")
            print("0 - Salir")
            print("-"*30)
            
            opcion = input("Selecciona gesto: ").strip()
            if opcion == "0": break
            
            if opcion.isdigit() and 1 <= int(opcion) <= len(GESTOS):
                gesto = GESTOS[int(opcion)-1]
                recorder.capturar(gesto, TIEMPO_CAPTURA)
            else:
                print("❌ Opción inválida.")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de tener permisos (sudo chmod 666 /dev/ttyACM0)")

if __name__ == "__main__":
    main()
