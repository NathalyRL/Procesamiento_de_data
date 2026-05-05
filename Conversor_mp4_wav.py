import os
import subprocess
from pathlib import Path

# ================= CONFIGURACIÓN =================
# 1. PEGA AQUÍ LA RUTA AL ARCHIVO FFMPEG.EXE
# Ejemplo: r"C:\ffmpeg\bin\ffmpeg.exe"
ruta_ffmpeg = r"C:\TU_RUTA_AQUI\bin\ffmpeg.exe" 

# 2. RUTAS DE TUS ARCHIVOS
carpeta_videos = Path(r"C:\Ruta\A\Tus\Videos_MP4")
carpeta_salida = Path(r"C:\Ruta\A\Tus\Audios_WAV")
# =================================================

# Crear carpeta de salida si no existe
carpeta_salida.mkdir(parents=True, exist_ok=True)

# Contador para saber cuántos procesamos
convertidos = 0

print("--- Iniciando conversión ---")

for archivo in carpeta_videos.glob("*.mp4"):
    entrada = str(archivo)
    salida = str(carpeta_salida / f"{archivo.stem}.wav")
    
    print(f"Procesando: {archivo.name}...")

    # El comando usando la ruta directa al ejecutable
    comando = [
        ruta_ffmpeg,
        "-i", entrada,
        "-vn",               # No video
        "-acodec", "pcm_s16le", # Calidad WAV estándar
        "-ar", "44100",      # Frecuencia de muestreo
        "-y",                # Sobrescribir si ya existe
        salida
    ]

    try:
        # Ejecutamos el proceso
        resultado = subprocess.run(comando, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"✅ Convertido: {archivo.stem}.wav")
            convertidos += 1
        else:
            print(f"❌ Error en {archivo.name}: {resultado.stderr}")
            
    except Exception as e:
        print(f"🔥 Error crítico: {e}")

print(f"\n--- ¡Listo! Se convirtieron {convertidos} archivos ---")