import os
import shutil
import subprocess
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO\LENOVO_F_REDIMENSION_CORTES"
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO\LENOVO_F_REDIMENSION_GDER"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')


# Sufijo que se agrega al nombre del archivo de salida (antes de la extensión).
# Ej: con SUFIJO = "_rot", "01_02_03_01.mp4" -> "01_02_03_01_rot.mp4"
# Dejalo como "" (cadena vacía) si no querés agregar ningún sufijo.
SUFIJO = "_03"

# Dirección de giro: "izquierda" (90° antihorario) o "derecha" (90° horario)
DIRECCION = "derecha"

FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "18"


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Revisa la variable 'ruta_bin_ffmpeg' al inicio del script.")
        return False
    return True


def obtener_valor_transpose(direccion):
    """Convierte la dirección elegida al valor del filtro 'transpose' de FFmpeg.
    1 = 90° horario (derecha), 2 = 90° antihorario (izquierda)."""
    direccion = direccion.strip().lower()
    if direccion in ("izquierda", "left", "counterclockwise", "antihorario"):
        return 2
    if direccion in ("derecha", "right", "clockwise", "horario"):
        return 1
    raise ValueError(f"Dirección no reconocida: '{direccion}'. Usa 'izquierda' o 'derecha'.")


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


def rotar_video(ruta_entrada_video, ruta_salida_video, valor_transpose):
    comando = [
        'ffmpeg', '-y',
        '-i', ruta_entrada_video,
        '-vf', f'transpose={valor_transpose}',
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-c:a', 'copy',
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


def construir_ruta_salida(ruta_entrada_video):
    """Reconstruye la misma estructura de subcarpetas en carpeta_salida y
    agrega el SUFIJO configurado antes de la extensión."""
    ruta_relativa = os.path.relpath(ruta_entrada_video, carpeta_entrada)
    carpeta_relativa = os.path.dirname(ruta_relativa)
    nombre_archivo = os.path.basename(ruta_relativa)
    nombre_base, extension = os.path.splitext(nombre_archivo)

    carpeta_destino = os.path.join(carpeta_salida, carpeta_relativa)
    os.makedirs(carpeta_destino, exist_ok=True)

    return os.path.join(carpeta_destino, f"{nombre_base}{SUFIJO}{extension}")


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    try:
        valor_transpose = obtener_valor_transpose(DIRECCION)
    except ValueError as e:
        print(f"❌ {e}")
        raise SystemExit(1)

    os.makedirs(carpeta_salida, exist_ok=True)

    rutas = obtener_rutas_a_procesar(carpeta_entrada)
    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {carpeta_entrada}")
        raise SystemExit(0)

    total = len(rutas)
    print(f"Se encontraron {total} videos para rotar hacia la {DIRECCION}.\n")

    exitosos = 0
    for idx, ruta in enumerate(tqdm(rutas, desc="Procesando videos", unit="video"), start=1):
        nombre = os.path.basename(ruta)
        ruta_salida_video = construir_ruta_salida(ruta)

        try:
            if rotar_video(ruta, ruta_salida_video, valor_transpose):
                exitosos += 1
        except Exception as e:
            tqdm.write(f"    ❌ Error en {nombre}: {e}")

    print("\n========================================")
    print(f"¡ROTACIÓN COMPLETADA! {exitosos}/{total} videos procesados.")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("========================================")