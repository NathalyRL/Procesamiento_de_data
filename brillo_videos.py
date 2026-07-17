import os
import shutil
import subprocess

try:
    from tqdm import tqdm
except ImportError:  # fallback si tqdm no está instalado
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
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO\LENOVO_F_REDIMENSION_CORTES"
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO\LENOVO_F_REDIMENSION_BRILLO LOW"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')

SUFIJO = "_05"

# =============================================================================
# NIVEL DE BRILLO (multiplicativo, tipo exposición fotográfica)
# 1.0  = sin cambios | >1.0 = sube | <1.0 = baja
# Rango recomendado para augmentation: entre 0.7 y 1.3 aprox.
# =============================================================================
NIVEL_BRILLO = 0.7

FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "18"


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


def ajustar_brillo_video(ruta_entrada_video, ruta_salida_video, nivel):
    # colorchannelmixer con solo la diagonal (rr, gg, bb) = multiplicar cada
    # canal por 'nivel', igual que hacíamos con NumPy, pero calculado
    # internamente por FFmpeg en un solo paso, sin pipe de Python de por medio.
    filtro = f"colorchannelmixer=rr={nivel}:gg={nivel}:bb={nivel}"

    comando = [
        'ffmpeg', '-y',
        '-i', ruta_entrada_video,
        '-vf', filtro,
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-pix_fmt', 'yuv420p',   # formato de color ampliamente compatible;
                                  # sin esto, el video generado puede quedar
                                  # con un formato que muchos reproductores
                                  # y frameworks no reconocen ("códec no
                                  # compatible" al intentar abrirlo).
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
    porcentaje = (NIVEL_BRILLO - 1.0) * 100
    print(f"Se encontraron {total} videos para ajustar brillo "
          f"(nivel {NIVEL_BRILLO}, {porcentaje:+.0f}%).\n")

    exitosos = 0
    for ruta in tqdm(rutas, desc="Procesando videos", unit="video"):
        ruta_salida_video = construir_ruta_salida(ruta)

        try:
            if ajustar_brillo_video(ruta, ruta_salida_video, NIVEL_BRILLO):
                exitosos += 1
        except Exception as e:
            print(f"    ❌ Error inesperado: {e}")

    print("\n========================================")
    print(f"¡AJUSTE DE BRILLO COMPLETADO! {exitosos}/{total} videos procesados.")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("========================================")