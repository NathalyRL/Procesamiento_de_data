import os
import shutil
import subprocess

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_REDIMENSION"
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_BLUR"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')

SUFIJO = "_06"

# =============================================================================
# NIVEL DE DESENFOQUE (sigma del filtro gaussiano)
# 0    = sin cambios
# Rango recomendado para augmentation: 0.5 a 3.0 aprox.
#   0.5-1.0 = desenfoque sutil
#   1.5-2.5 = desenfoque moderado (rango típico para augmentation facial)
#   >3.0    = pierde demasiado detalle facial, usar con cuidado
# =============================================================================
NIVEL_BLUR = 1.5

# Codificación por CPU (libx264). 'veryfast' da buen balance velocidad/calidad
# para clips chicos como estos; 'ultrafast' si necesitás priorizar velocidad
# al máximo aunque el archivo pese un poco más.
FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "18"
FFMPEG_THREADS = "0"  # 0 = FFmpeg elige automáticamente según núcleos disponibles


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        return False
    return True


def obtener_rutas_a_procesar(ruta_entrada):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: La ruta '{ruta_entrada}' no existe.")
        return []
    rutas = []
    for raiz, _, archivos in os.walk(ruta_entrada):
        for archivo in archivos:
            if archivo.lower().endswith(extensiones_validas):
                rutas.append(os.path.join(raiz, archivo))
    rutas.sort()
    return rutas


def construir_ruta_salida(ruta_entrada_video):
    ruta_relativa = os.path.relpath(ruta_entrada_video, carpeta_entrada)
    carpeta_relativa = os.path.dirname(ruta_relativa)
    nombre_archivo = os.path.basename(ruta_relativa)
    nombre_base, extension = os.path.splitext(nombre_archivo)

    carpeta_destino = os.path.join(carpeta_salida, carpeta_relativa)
    os.makedirs(carpeta_destino, exist_ok=True)

    return os.path.join(carpeta_destino, f"{nombre_base}{SUFIJO}{extension}")


def aplicar_blur_video(ruta_entrada_video, ruta_salida_video, sigma):
    filtro = f"gblur=sigma={sigma}"

    comando = [
        'ffmpeg', '-y',
        '-i', ruta_entrada_video,
        '-vf', filtro,
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-threads', FFMPEG_THREADS,
        '-an',
        ruta_salida_video
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)

    if resultado.returncode != 0:
        return False
    if not os.path.exists(ruta_salida_video) or os.path.getsize(ruta_salida_video) == 0:
        return False
    return True


def mostrar_progreso(actual, total):
    porcentaje = int((actual / total) * 100) if total else 100
    ancho_barra = 30
    completados = int(ancho_barra * actual / total) if total else ancho_barra
    barra = "#" * completados + "-" * (ancho_barra - completados)
    print(f"\rProgreso: [{barra}] {actual}/{total} ({porcentaje:3d}%)", end="", flush=True)


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    os.makedirs(carpeta_salida, exist_ok=True)

    rutas = obtener_rutas_a_procesar(carpeta_entrada)
    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {carpeta_entrada}")
        raise SystemExit(0)

    total = len(rutas)
    print(f"Se encontraron {total} videos para aplicar blur (sigma={NIVEL_BLUR}).\n")

    exitosos = 0
    for idx, ruta in enumerate(rutas, start=1):
        ruta_salida_video = construir_ruta_salida(ruta)

        try:
            if aplicar_blur_video(ruta, ruta_salida_video, NIVEL_BLUR):
                exitosos += 1
        except Exception:
            pass

        mostrar_progreso(idx, total)

    print("\n")
    print("========================================")
    print(f"¡BLUR APLICADO! {exitosos}/{total} videos procesados.")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("========================================")