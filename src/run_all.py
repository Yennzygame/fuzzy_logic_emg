"""Ejecuta el pipeline: datos -> modelo -> evaluación -> comparación -> proyecciones."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def main():
    for script in (
        "01_preparar_datos.py", "02_entrenar_fcm_tsk.py", "03_evaluar.py",
        "07_analisis_comparativo.py", "08_scatter_centroides.py",
    ):
        print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=SRC.parent)
    print("\nPipeline completo. Revisa docs/03_resultados.md y docs/05_analisis_comparativo.md")


if __name__ == "__main__":
    main()
