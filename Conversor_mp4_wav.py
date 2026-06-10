import os
import subprocess
from pathlib import Path

# Este código hace uso de la herramienta ffmpeg para convertir archivos .mp4 a .wav
# Asegúrate de tener ffmpeg instalado y de actualizar la ruta en la configuración masiva. 
# Luego, simplemente ejecuta este script y se procesarán todos los archivos .mp4 en la 
# carpeta especificada, guardando los resultados en la carpeta de salida.


# ================= CONFIGURACIÓN MASIVA =================
# 1. Ruta al ejecutable (la misma que usaste en la prueba)
ruta_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin\ffmpeg.exe"

# 2. Carpeta donde están los 40 videos .mp4
carpeta_videos = Path(r"D:\Documentos\Ayudante de Investigacion\CEL_GO_PRO2")

# 3. Carpeta donde quieres guardar los audios (puede ser la misma o una nueva)
carpeta_salida = Path(r"D:\Documentos\Ayudante de Investigacion\Codigos\Archivos_wav_convertidos_2")
# ========================================================

# Crear la carpeta de salida si no existe
carpeta_salida.mkdir(parents=True, exist_ok=True)

# Contador para el resumen final
exitos = 0
errores = 0

print(f"--- Iniciando conversión masiva en: {carpeta_videos} ---")

# Buscamos todos los archivos .mp4
for archivo in carpeta_videos.glob("*.m4a"):
    # Construir el nuevo nombre: nombreOriginal_1.wav
    nuevo_nombre = f"{archivo.stem}_1.wav"
    ruta_salida = carpeta_salida / nuevo_nombre
    
    print(f"Procesando: {archivo.name} -> {nuevo_nombre}")

    comando = [
        ruta_ffmpeg,
        "-i", str(archivo),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-y",
        str(ruta_salida)
    ]

    try:
        # Ejecutar conversión
        resultado = subprocess.run(comando, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"  ✅ OK")
            exitos += 1
        else:
            print(f"  ❌ ERROR en este archivo: {resultado.stderr}")
            errores += 1
            
    except Exception as e:
        print(f"  🔥 Error inesperado: {e}")
        errores += 1

print("\n========================================")
print(f"PROCESO FINALIZADO")
print(f"Archivos convertidos con éxito: {exitos}")
print(f"Archivos con error: {errores}")
print("========================================")