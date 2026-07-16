import os
import shutil
import subprocess
import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_I"
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_I_H264_2"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

FFMPEG_PRESET = "ultrafast"   # velocidad máxima; sube a "veryfast" si te sobra tiempo
FFMPEG_CRF = "18"             # 18 = casi sin pérdida de calidad
FFMPEG_THREADS = "2"          # limita núcleos usados; ajusta según tu Raspberry Pi/servidor

# El destino final es 224x224, así que no tiene sentido convertir a resolución
# completa: reducir el lado mayor a este tamaño ahorra mucho tiempo/CPU sin
# perder nada relevante para la detección de rostro ni el recorte posterior.
# Poné None para no reducir nada (mantiene resolución original).
ANCHO_MAXIMO = 640

def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Revisa la variable 'ruta_bin_ffmpeg' al inicio del script.")
        return False
    return True


def verificar_unidad(ruta, intentos=3, espera=2):
    unidad = os.path.splitdrive(ruta)[0]
    if not unidad:
        return True
    for intento in range(1, intentos + 1):
        if os.path.exists(unidad + "\\"):
            return True
        print(f"⚠️ Unidad '{unidad}' no responde, esperando {espera}s (intento {intento}/{intentos})...")
        time.sleep(espera)
    print(f"❌ La unidad '{unidad}' nunca respondió.")
    return False


def obtener_rutas_a_procesar(ruta_entrada):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: La ruta '{ruta_entrada}' no existe.")
        return []
    if os.path.isfile(ruta_entrada):
        return [ruta_entrada]
    rutas = []
    for raiz, _, archivos in os.walk(ruta_entrada):
        for archivo in archivos:
            if archivo.lower().endswith(extensiones_validas):
                rutas.append(os.path.join(raiz, archivo))
    rutas.sort()
    return rutas


def detectar_codec_video(ruta_video):
    comando = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        ruta_video
    ]
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True, timeout=15)
        return resultado.stdout.strip().lower()
    except Exception as e:
        print(f"    ⚠️ No se pudo leer el códec: {e}")
        return None


def convertir_a_h264(ruta_entrada, ruta_salida):
    comando = [
        'ffmpeg', '-y',
        '-hwaccel', 'auto',
        '-i', ruta_entrada,
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-threads', FFMPEG_THREADS,
        '-c:a', 'copy',   # mantiene el audio tal cual, sin recodificar
    ]

    if ANCHO_MAXIMO:
        # Reduce el lado mayor a ANCHO_MAXIMO, manteniendo proporción, solo si
        # el video es más grande que eso (evita agrandar videos ya chicos).
        filtro = (
            f"scale='if(gt(iw,ih),min(iw,{ANCHO_MAXIMO}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{ANCHO_MAXIMO}))'"
        )
        comando += ['-vf', filtro]

    comando.append(ruta_salida)

    inicio = time.time()
    resultado = subprocess.run(comando, capture_output=True, text=True)
    duracion = time.time() - inicio

    if resultado.returncode != 0:
        print(f"    ❌ Falló la conversión: {resultado.stderr.strip()[-400:]}")
        return False
    if not os.path.exists(ruta_salida) or os.path.getsize(ruta_salida) == 0:
        print(f"    ❌ El archivo de salida no se generó o quedó vacío.")
        return False

    print(f"    ✓ Convertido en {duracion:.1f}s.")
    return True


def procesar_video(ruta_entrada, carpeta_salida):
    nombre_archivo = os.path.basename(ruta_entrada)
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

    # Reanudación: si ya existe (de una corrida anterior interrumpida), se salta
    if os.path.exists(ruta_salida) and os.path.getsize(ruta_salida) > 0:
        print(f"⏭️  Ya existe, se omite: {nombre_archivo}")
        return

    print(f"\n>>> {nombre_archivo}")
    codec = detectar_codec_video(ruta_entrada)
    print(f"    ℹ️ Códec detectado: {codec if codec else '(no se pudo determinar)'}")

    if codec in ('hevc', 'h265'):
        print("    🔄 Convirtiendo a H.264...")
        convertir_a_h264(ruta_entrada, ruta_salida)
    else:
        # Ya es H.264 (u otro códec compatible): copiar directo, sin recodificar
        print("    ✓ Ya es H.264 (u otro códec no-HEVC). Copiando sin recodificar...")
        shutil.copy2(ruta_entrada, ruta_salida)


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    if not verificar_unidad(carpeta_entrada) or not verificar_unidad(carpeta_salida):
        raise SystemExit(1)

    os.makedirs(carpeta_salida, exist_ok=True)

    rutas = obtener_rutas_a_procesar(carpeta_entrada)
    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {carpeta_entrada}")
        raise SystemExit(0)

    total = len(rutas)
    print(f"Se encontraron {total} videos para revisar/convertir.\n")

    for idx, ruta in enumerate(rutas, start=1):
        print(f"[{idx}/{total}]", end=" ")
        try:
            procesar_video(ruta, carpeta_salida)
        except Exception as e:
            print(f"❌ Error inesperado procesando {os.path.basename(ruta)}: {e}")

    print("\n========================================")
    print("¡CONVERSIÓN COMPLETADA!")
    print(f"Videos en H.264 guardados en: {carpeta_salida}")
    print("========================================")