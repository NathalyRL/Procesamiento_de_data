# CORTAR VIDEO ACORDE A UN PITIDO DETECTADO EN EL AUDIO
import librosa
import numpy as np
import os
import shutil
import subprocess
import tempfile

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

# --- CONFIGURACIÓN MASIVA ---
# Carpeta donde están tus videos originales (con el audio de los pitidos)
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_I_H264"
# Carpeta donde se guardarán todos los recortes de VIDEO (sin audio)
carpeta_salida_master = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_CORTES"

extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

frecuencia_pitido = 1198
umbral_sensibilidad = 0.495
FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "18"


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Revisa la variable 'ruta_bin_ffmpeg' al inicio del script.")
        return False
    return True


def extraer_audio_temporal(ruta_video):
    """Extrae la pista de audio del video a un .wav temporal, para poder
    analizarla con librosa exactamente igual que en el script original."""
    carpeta_temp = tempfile.gettempdir()
    nombre_temp = f"_tmp_audio_{os.path.splitext(os.path.basename(ruta_video))[0]}.wav"
    ruta_temp = os.path.join(carpeta_temp, nombre_temp)

    comando = [
        'ffmpeg', '-y',
        '-i', ruta_video,
        '-vn',                 # sin video, solo audio
        '-acodec', 'pcm_s16le',
        ruta_temp
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    if resultado.returncode != 0 or not os.path.exists(ruta_temp):
        print(f"    ❌ No se pudo extraer el audio: {resultado.stderr.strip()[-400:]}")
        return None
    return ruta_temp


def detectar_pitidos(ruta_wav, beep_freq, threshold):
    """Misma lógica de detección que el script original, sin ningún cambio."""
    y, sr = librosa.load(ruta_wav)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    idx = (np.abs(freqs - beep_freq)).argmin()

    energia = S[idx]
    energia = energia / (np.max(energia) + 1e-9)

    frames = np.where(energia > threshold)[0]
    tiempos = librosa.frames_to_time(frames, sr=sr)

    pitidos_finales = []
    if len(tiempos) > 0:
        pitidos_finales.append(tiempos[0])
        for t in tiempos:
            if t - pitidos_finales[-1] > 2.0:
                pitidos_finales.append(t)

    return pitidos_finales


def cortar_segmento_video(ruta_video, inicio_seg, duracion_seg, ruta_salida):
    """Corta un segmento del VIDEO (sin audio) usando seek híbrido:
    un salto grueso rápido antes de -i, y un ajuste fino después de -i,
    para lograr velocidad y precisión de corte a la vez."""
    margen = 5.0
    salto_grueso = max(0.0, inicio_seg - margen)
    ajuste_fino = inicio_seg - salto_grueso

    comando = [
        'ffmpeg', '-y',
        '-ss', str(salto_grueso),
        '-i', ruta_video,
        '-ss', str(ajuste_fino),
        '-t', str(duracion_seg),
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-an',                 # sin audio en el resultado
        ruta_salida
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)

    if resultado.returncode != 0:
        print(f"    ❌ Error generando {os.path.basename(ruta_salida)}: "
              f"{resultado.stderr.strip()[-400:]}")
        return False
    if not os.path.exists(ruta_salida) or os.path.getsize(ruta_salida) == 0:
        print(f"    ❌ {os.path.basename(ruta_salida)} no se generó o quedó vacío.")
        return False
    return True


def procesar_archivo(ruta_video, carpeta_destino, beep_freq, threshold):
    nombre_base = os.path.splitext(os.path.basename(ruta_video))[0]

    print(f"\n>>> Analizando: {nombre_base}")

    ruta_audio_temp = extraer_audio_temporal(ruta_video)
    if ruta_audio_temp is None:
        print(f"    ⚠️ Se omite '{nombre_base}' porque no se pudo extraer el audio.")
        return

    try:
        pitidos_finales = detectar_pitidos(ruta_audio_temp, beep_freq, threshold)
    finally:
        if os.path.exists(ruta_audio_temp):
            os.remove(ruta_audio_temp)

    print(f"    Se encontraron {len(pitidos_finales)} pitidos.")

    if not pitidos_finales:
        print(f"    ⚠️ No se detectaron pitidos en '{nombre_base}'. Se omite.")
        return

    carpeta_destino_video = os.path.join(carpeta_destino, nombre_base)
    os.makedirs(carpeta_destino_video, exist_ok=True)

    cortes_generados = 0
    for i, tiempo_pitido in enumerate(pitidos_finales):
        inicio_seg = tiempo_pitido + 0.3   # +300ms, igual que el original (+300 ms en milisegundos)

        if i < len(pitidos_finales) - 1:
            fin_seg = pitidos_finales[i + 1] - 0.05   # -50ms antes del siguiente pitido
        else:
            fin_seg = inicio_seg + 4.59               # mismo valor fijo que el original

        duracion_seg = fin_seg - inicio_seg
        if duracion_seg <= 0:
            print(f"    ⚠️ Segmento {i+1} con duración inválida, se omite.")
            continue

        nombre_clip = f"{nombre_base}_{i+1:03d}.mp4"
        ruta_final = os.path.join(carpeta_destino_video, nombre_clip)

        if cortar_segmento_video(ruta_video, inicio_seg, duracion_seg, ruta_final):
            cortes_generados += 1

    print(f"    ✓ {nombre_base} procesado: {cortes_generados}/{len(pitidos_finales)} cortes generados.")


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    os.makedirs(carpeta_salida_master, exist_ok=True)

    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(extensiones_validas)]

    total = len(archivos)
    print(f"Se encontraron {total} videos para procesar.")

    for index, nombre_archivo in enumerate(archivos):
        ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
        print(f"[{index + 1}/{total}]", end="")
        try:
            procesar_archivo(ruta_completa, carpeta_salida_master, frecuencia_pitido, umbral_sensibilidad)
        except Exception as e:
            print(f"❌ Error inesperado procesando {nombre_archivo}: {e}")

    print("\n========================================")
    print("¡PROCESAMIENTO MASIVO COMPLETADO!")
    print(f"Los clips de video están en: {carpeta_salida_master}")
    print("========================================")