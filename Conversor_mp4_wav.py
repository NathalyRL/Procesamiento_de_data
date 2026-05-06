import subprocess

# ================= CONFIGURACIÓN DE PRUEBA =================
# 1. Ruta exacta al ejecutable (ejemplo: r"C:\ffmpeg\bin\ffmpeg.exe")
ruta_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin\ffmpeg.exe"

# 2. Ruta del video MP4 que quieres probar
video_entrada = r"D:\Documentos\Ayudante de Investigacion\Codigos\02_00_05.MP4"

# 3. Ruta donde quieres que se guarde el audio resultante
audio_salida = r"D:\Documentos\Ayudante de Investigacion\Codigos\Archivo_wav_convertidos\02_00_05.wav"
# ===========================================================

print(f"Intentando convertir: {video_entrada}...")

comando = [
    ruta_ffmpeg,
    "-i", video_entrada,
    "-vn",                   # Extraer solo audio
    "-acodec", "pcm_s16le",     # Formato WAV (16 bits)
    "-ar", "44100",          # Calidad estándar (44.1kHz)
    "-y",                    # Sobrescribir si ya existe
    audio_salida
]

try:
    # Ejecutamos el comando y capturamos la salida para ver errores
    proceso = subprocess.run(comando, capture_output=True, text=True)

    if proceso.returncode == 0:
        print("✅ ¡Prueba exitosa! El archivo .wav se creó correctamente.")
        print(f"Ubicación: {audio_salida}")
    else:
        print("❌ Error al convertir:")
        print(proceso.stderr) # Esto te dirá exactamente qué falló

except FileNotFoundError:
    print("❌ ERROR: No se encontró el archivo ffmpeg.exe.")
    print("Verifica que la 'ruta_ffmpeg' sea la correcta y termine en .exe")
except Exception as e:
    print(f"🔥 Ocurrió un error inesperado: {e}")