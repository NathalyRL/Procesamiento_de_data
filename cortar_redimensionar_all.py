import cv2
import os
import re
import shutil
import subprocess
import time
import traceback

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
sufijo_corte = "_01"
video_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_I"
carpeta_salida_master = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_I_REDIMENSION_CORTES"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

TAMANO_SALIDA = 224
FRAME_SEGURO = 20
MAX_INTENTOS = 5
MARGEN_ROSTRO = 2.2

DURACION_FRAGMENTO = 5.0
DURACION_MINIMA_FINAL = 2.0

FFMPEG_PRESET = "faster"
FFMPEG_CRF = "20"


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Revisa la variable 'ruta_bin_ffmpeg' al inicio del script.")
        return False
    return True


def verificar_unidad(ruta, intentos=3, espera=2):
    """Espera a que la unidad de disco de 'ruta' esté realmente accesible.
    Evita el fallo intermitente de carpetas no creadas por timing del disco."""
    unidad = os.path.splitdrive(ruta)[0]
    if not unidad:
        return True
    print(f"🔧 Verificando unidad '{unidad}'...")
    for intento in range(1, intentos + 1):
        if os.path.exists(unidad + "\\"):
            print(f"    ✓ Unidad '{unidad}' accesible (intento {intento}).")
            return True
        print(f"    ⚠️ Unidad '{unidad}' no responde, esperando {espera}s (intento {intento}/{intentos})...")
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


def obtener_duracion_con_ffprobe(ruta_video):
    comando = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        ruta_video
    ]
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        return float(resultado.stdout.strip())
    except Exception as e:
        print(f"    ❌ ffprobe falló leyendo {os.path.basename(ruta_video)}: {e}")
        return 0.0


def detectar_recorte_rostro(cap, ancho, alto, total_frames, face_cascade):
    """Prueba FRAME_SEGURO, 2x, 3x... hasta encontrar rostro. Devuelve (x1, y1, lado, encontrado)."""
    lado_central = min(alto, ancho)
    x1_def = (ancho - lado_central) // 2
    y1_def = (alto - lado_central) // 2

    for intento in range(1, MAX_INTENTOS + 1):
        frame_objetivo = FRAME_SEGURO * intento
        if frame_objetivo >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame_deteccion = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame_deteccion, cv2.COLOR_BGR2GRAY)
        gray_pequeno = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        rostros = face_cascade.detectMultiScale(gray_pequeno, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        if len(rostros) > 0:
            fx, fy, fw, fh = rostros[0] * 2
            cx, cy = fx + fw // 2, fy + fh // 2
            tamano_cuadro = int(max(fw, fh) * MARGEN_ROSTRO)

            x1 = max(0, cx - tamano_cuadro // 2)
            y1 = max(0, cy - tamano_cuadro // 2)
            x2 = min(ancho, cx + tamano_cuadro // 2)
            y2 = min(alto, cy + tamano_cuadro // 2)

            lado = min(x2 - x1, y2 - y1)
            if intento > 1:
                print(f"    ✓ Rostro detectado en el intento {intento} (frame {frame_objetivo}).")
            return x1, y1, lado, True
        else:
            print(f"    ⚠️ Sin rostro en frame {frame_objetivo}, probando siguiente...")

    return x1_def, y1_def, lado_central, False


def procesar_video(ruta_entrada, carpeta_destino_master, face_cascade):
    nombre_archivo = os.path.basename(ruta_entrada)
    nombre_base, _ = os.path.splitext(nombre_archivo)

    print(f"\n>>> Procesando: {nombre_archivo}")

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {ruta_entrada}")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("🔍 Detectando rostro...")
    x1, y1, lado, rostro_encontrado = detectar_recorte_rostro(cap, ancho, alto, total_frames, face_cascade)
    cap.release()

    if not rostro_encontrado:
        print("⚠️ No se detectó rostro en ningún frame de prueba. Se usará recorte central por defecto.")

    duracion_total = obtener_duracion_con_ffprobe(ruta_entrada)
    if duracion_total == 0.0:
        print("    ⚠️ Se omite este video porque no se pudo leer su duración.")
        return

    carpeta_destino_video = os.path.join(carpeta_destino_master, nombre_base)
    os.makedirs(carpeta_destino_video, exist_ok=True)

    patron_temporal = os.path.join(carpeta_destino_video, f"{nombre_base}_tmp_%03d.mp4")
    filtro = f"crop={lado}:{lado}:{x1}:{y1},scale={TAMANO_SALIDA}:{TAMANO_SALIDA}"

    comando_ffmpeg = [
        'ffmpeg', '-y',
        '-i', ruta_entrada,
        '-vf', filtro,
        '-an',
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-pix_fmt', 'yuv420p',
        '-f', 'segment',
        '-segment_time', str(DURACION_FRAGMENTO),
        '-reset_timestamps', '1',
        patron_temporal
    ]

    print(f"🎬 Recortando rostro, escalando a {TAMANO_SALIDA}x{TAMANO_SALIDA} y "
          f"cortando cada {DURACION_FRAGMENTO:.0f}s...")
    resultado = subprocess.run(comando_ffmpeg, capture_output=True, text=True)

    if resultado.returncode != 0:
        print(f"    ❌ FFmpeg falló:")
        print(f"       {resultado.stderr.strip()[-500:]}")
        return

    patron_nombre = re.compile(rf"^{re.escape(nombre_base)}_tmp_(\d+)\.mp4$")
    temporales = sorted(f for f in os.listdir(carpeta_destino_video) if patron_nombre.match(f))

    if not temporales:
        print("    ❌ No se generó ningún fragmento.")
        return

    ultimo = os.path.join(carpeta_destino_video, temporales[-1])
    duracion_ultimo = obtener_duracion_con_ffprobe(ultimo)
    if duracion_ultimo < DURACION_MINIMA_FINAL and len(temporales) > 1:
        print(f"    ℹ️ Último fragmento de {duracion_ultimo:.2f}s descartado.")
        os.remove(ultimo)
        temporales.pop()

    fragmentos_generados = 0
    for idx, nombre_tmp in enumerate(temporales, start=1):
        ruta_tmp = os.path.join(carpeta_destino_video, nombre_tmp)
        nombre_final = f"{nombre_base}_{idx:02d}{sufijo_corte}.mp4"
        ruta_final = os.path.join(carpeta_destino_video, nombre_final)
        os.replace(ruta_tmp, ruta_final)
        fragmentos_generados += 1

    print(f"    ✓ {nombre_base}: {fragmentos_generados} fragmentos generados correctamente.")


if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    if not verificar_unidad(video_entrada) or not verificar_unidad(carpeta_salida_master):
        raise SystemExit(1)

    try:
        os.makedirs(carpeta_salida_master, exist_ok=True)
        if not os.path.isdir(carpeta_salida_master):
            print(f"❌ La carpeta de salida no se pudo crear: {carpeta_salida_master}")
            raise SystemExit(1)
        print(f"✓ Carpeta de salida confirmada: {carpeta_salida_master}")

        rutas = obtener_rutas_a_procesar(video_entrada)
    except Exception:
        print("❌ Error inesperado durante la configuración inicial:")
        traceback.print_exc()
        raise SystemExit(1)

    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {video_entrada}")
        raise SystemExit(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    total = len(rutas)
    print(f"🔥 Iniciando procesamiento híbrido (OpenCV + FFmpeg) de {total} videos.")

    for index, ruta in enumerate(rutas):
        print(f"[{index + 1}/{total}]", end="")
        try:
            procesar_video(ruta, carpeta_salida_master, face_cascade)
        except Exception as e:
            print(f"❌ Error inesperado procesando {os.path.basename(ruta)}: {e}")
            traceback.print_exc()

    print("\n========================================")
    print("¡PROCESO DE RECORTE Y CORTE COMPLETADO!")
    print(f"Resultados guardados en: {carpeta_salida_master}")
    print("========================================")