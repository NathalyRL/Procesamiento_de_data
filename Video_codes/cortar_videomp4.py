import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# --- CONFIGURACIÓN ---
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\Cel Go Pro"
carpeta_salida_master = r"D:\Documentos\Ayudante de Investigacion\Codigos\VIDEO\Cortes_izq_video"

frecuencia_pitido = 1198   # Hz — ajusta si tu pitido tiene otra frecuencia
umbral_sensibilidad = 0.4  # 0.0–1.0 — baja si no detecta, sube si detecta de más

# Extensiones de video aceptadas
EXTENSIONES_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v")


def detectar_pitidos(ruta_video, beep_freq, threshold):
    """
    Extrae el audio del video y detecta los instantes (en segundos)
    donde aparece la frecuencia del pitido por encima del umbral.
    Devuelve una lista de tiempos [t0, t1, t2, ...].
    """
    # librosa carga directamente video si tiene ffmpeg instalado,
    # pero es más robusto extraer el audio temporal primero.
    ruta_audio_tmp = ruta_video + "_tmp_audio.wav"

    # Extraer audio con moviepy
    with VideoFileClip(ruta_video) as clip:
        if clip.audio is None:
            print("    ⚠ El video no tiene pista de audio. Se omite.")
            return []
        clip.audio.write_audiofile(ruta_audio_tmp, logger=None)

    # Analizar con librosa
    y, sr = librosa.load(ruta_audio_tmp, sr=None)
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
            if t - pitidos_finales[-1] > 2.0:  # mínimo 2 s entre pitidos
                pitidos_finales.append(t)

    # Limpiar audio temporal
    os.remove(ruta_audio_tmp)

    return pitidos_finales


def procesar_video(ruta_video, carpeta_destino, beep_freq, threshold):
    nombre_base = os.path.splitext(os.path.basename(ruta_video))[0]
    print(f"\n>>> Analizando: {nombre_base}")

    # Detectar pitidos
    pitidos_finales = detectar_pitidos(ruta_video, beep_freq, threshold)
    print(f"    Se encontraron {len(pitidos_finales)} pitidos.")

    if not pitidos_finales:
        print("    ⚠ Sin pitidos detectados. Se omite el archivo.")
        return

    # Crear subcarpeta por archivo
    carpeta_clips = os.path.join(carpeta_destino, nombre_base)
    os.makedirs(carpeta_clips, exist_ok=True)

    with VideoFileClip(ruta_video) as video:
        duracion_total = video.duration  # segundos

        for i, tiempo_pitido in enumerate(pitidos_finales):
            inicio_s = tiempo_pitido + 0.3  # +300 ms tras el pitido

            if i < len(pitidos_finales) - 1:
                fin_s = pitidos_finales[i + 1] - 0.05  # -50 ms antes del siguiente
            else:
                fin_s = min(inicio_s + 4.59, duracion_total)  # último segmento: 4590 ms

            # Seguridad: evitar tiempos fuera de rango
            inicio_s = max(0, inicio_s)
            fin_s = min(fin_s, duracion_total)

            if fin_s <= inicio_s:
                print(f"    ⚠ Segmento {i+1} inválido ({inicio_s:.2f}s → {fin_s:.2f}s). Se omite.")
                continue

            nombre_clip = f"{nombre_base}_{i+1:03d}_01.mp4"
            ruta_final = os.path.join(carpeta_clips, nombre_clip)

            segmento = video.subclipped(inicio_s, fin_s).without_audio()
            segmento.write_videofile(
                ruta_final,
                codec="libx264",
                audio=False,
                logger=None,   # cambia a "bar" si quieres barra de progreso por clip
            )
            print(f"    ✓ Clip {i+1} guardado: {nombre_clip}  ({inicio_s:.2f}s → {fin_s:.2f}s)")

    print(f"    ✓ {nombre_base} procesado con éxito.")


if __name__ == "__main__":
    os.makedirs(carpeta_salida_master, exist_ok=True)

    archivos = [
        f for f in os.listdir(carpeta_entrada)
        if f.lower().endswith(EXTENSIONES_VIDEO)
    ]

    total = len(archivos)
    print(f"Se encontraron {total} archivos de video para procesar.")

    for index, nombre_archivo in enumerate(archivos):
        ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
        print(f"[{index + 1}/{total}]", end="")
        procesar_video(ruta_completa, carpeta_salida_master, frecuencia_pitido, umbral_sensibilidad)

    print("\n========================================")
    print("¡PROCESAMIENTO MASIVO COMPLETADO!")
    print(f"Los clips están en: {carpeta_salida_master}")
    print("========================================")