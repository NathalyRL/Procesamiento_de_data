import os
import shutil
import subprocess

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

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
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_RUIDO"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')

SUFIJO = "_07"

# =============================================================================
# INTENSIDAD DEL RUIDO BLANCO GAUSSIANO (escala 0-100 del filtro 'noise')
# 0     = sin cambios
# Rango recomendado para augmentation: 5 a 25 aprox.
#   5-10  = ruido sutil (simula sensor de cámara de gama baja/poca luz leve)
#   10-20 = ruido moderado (poca luz notoria, cámara de baja calidad)
#   >25   = empieza a degradar demasiado el detalle facial, usar con cuidado
# =============================================================================
NIVEL_RUIDO = 20 

FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "26"
FFMPEG_THREADS = "0"


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


def aplicar_ruido_video(ruta_entrada_video, ruta_salida_video, intensidad):
    # allf=t (temporal): el patrón de ruido cambia frame a frame, como en un
    # sensor real. Sin la flag 'u' (uniforme), usa distribución GAUSSIANA.
    # Sin la flag 'c' (correlate), el ruido es independiente entre canales de
    # color -- esto es justamente lo que define "ruido blanco": sin
    # correlación espacial/entre canales, distribución gaussiana.
    filtro = f"noise=alls={intensidad}:allf=t"

    comando = [
        'ffmpeg', '-y',
        '-i', ruta_entrada_video,
        '-vf', filtro,
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-threads', FFMPEG_THREADS,
        '-pix_fmt', 'yuv420p',
        '-an',
        ruta_salida_video
    ]
    resultado = subprocess.run(comando, capture_output=True, text=True)

    if resultado.returncode != 0:
        print(f"    ❌ Error: {resultado.stderr.strip()[-400:]}")
        return False
    if not os.path.exists(ruta_salida_video) or os.path.getsize(ruta_salida_video) == 0:
        print(f"    ❌ El archivo de salida no se generó o quedó vacío.")
        return False
    return True


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    os.makedirs(carpeta_salida, exist_ok=True)

    rutas = obtener_rutas_a_procesar(carpeta_entrada)
    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {carpeta_entrada}")
        raise SystemExit(0)

    total = len(rutas)
    print(f"Se encontraron {total} videos para añadir ruido blanco gaussiano "
          f"(intensidad={NIVEL_RUIDO}).\n")

    exitosos = 0
    for ruta in tqdm(rutas, desc="Añadiendo ruido", unit="video"):
        ruta_salida_video = construir_ruta_salida(ruta)

        try:
            if aplicar_ruido_video(ruta, ruta_salida_video, NIVEL_RUIDO):
                exitosos += 1
        except Exception as e:
            print(f"    ❌ Error inesperado: {e}")

    print("\n========================================")
    print(f"¡RUIDO APLICADO! {exitosos}/{total} videos procesados.")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("========================================")