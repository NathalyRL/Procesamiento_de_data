# REDIMENSIONAR VIDEOS PREVIAMENTE CORTADOS POR FRASE AYUDADOS POR PITIDO
import cv2
import numpy as np
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
video_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\arreglos"
directorio_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\arreglos_redimension"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

sufijo = "_01"

# Modificar estos valores si se desea cambiar el comportamiento de la detección de rostro
FRAME_SEGURO = 15   # primer frame donde se intenta detectar el rostro
MAX_INTENTOS = 5    # probará FRAME_SEGURO, 2x, 3x, 4x, 5x antes de rendirse

TAMANO_SALIDA = 224
FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "15"

def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg",) if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        return False
    return True


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


def obtener_grupos_por_subcarpeta(ruta_entrada):
    """Agrupa los videos por la subcarpeta donde están, para poder detectar
    el rostro una sola vez por grupo y reutilizarlo en todo el resto."""
    rutas = obtener_rutas_a_procesar(ruta_entrada)
    if not rutas:
        return []

    base = ruta_entrada if os.path.isdir(ruta_entrada) else os.path.dirname(ruta_entrada)
    grupos = {}
    for ruta in rutas:
        carpeta_relativa = os.path.relpath(os.path.dirname(ruta), base)
        carpeta_key = "." if carpeta_relativa == "." else carpeta_relativa
        grupos.setdefault(carpeta_key, []).append(ruta)

    return sorted(grupos.items(), key=lambda item: item[0])


def construir_ruta_salida(ruta_entrada_video):
    base_relativa = video_entrada if os.path.isdir(video_entrada) else os.path.dirname(video_entrada)
    ruta_relativa = os.path.relpath(ruta_entrada_video, base_relativa)
    carpeta_relativa = os.path.dirname(ruta_relativa)
    nombre_archivo = os.path.basename(ruta_relativa)
    nombre_base, _ = os.path.splitext(nombre_archivo)

    carpeta_destino = os.path.join(directorio_salida, carpeta_relativa)
    os.makedirs(carpeta_destino, exist_ok=True)

    return os.path.join(carpeta_destino, f"{nombre_base}{sufijo}.mp4")


def detectar_recorte_rostro(ruta_video):
    """Abre el video, intenta detectar el rostro con reintentos, y devuelve
    (x1, y1, x2, y2, encontrado) SIN codificar nada. Se usa una sola vez por
    subcarpeta, sobre el primer clip."""
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el formato de este video: {ruta_video}")
        return None

    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Coordenadas de respaldo (Recorte central)
    x1, y1 = (ancho - min(alto, ancho)) // 2, (alto - min(alto, ancho)) // 2
    x2, y2 = x1 + min(alto, ancho), y1 + min(alto, ancho)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostro_encontrado = False

    for intento in range(1, MAX_INTENTOS + 1):
        frame_objetivo = FRAME_SEGURO * intento
        if frame_objetivo >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame_deteccion = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame_deteccion, cv2.COLOR_BGR2GRAY)
        gray_pequeño = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        rostros = face_cascade.detectMultiScale(gray_pequeño, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        if len(rostros) > 0:
            fx, fy, fw, fh = rostros[0] * 2
            cx = fx + fw // 2
            cy = fy + fh // 2
            tamaño_cuadro = int(max(fw, fh) * 2.2)

            x1 = max(0, cx - tamaño_cuadro // 2)
            y1 = max(0, cy - tamaño_cuadro // 2)
            x2 = min(ancho, cx + tamaño_cuadro // 2)
            y2 = min(alto, cy + tamaño_cuadro // 2)

            lado_real = min(x2 - x1, y2 - y1)
            x2, y2 = x1 + lado_real, y1 + lado_real
            rostro_encontrado = True
            if intento > 1:
                print(f"    ✓ Rostro detectado en el intento {intento} (frame {frame_objetivo}).")
            break
        else:
            print(f"    ⚠️ Sin rostro en frame {frame_objetivo}, probando siguiente...")

    cap.release()

    if not rostro_encontrado:
        print("    ⚠️ No se detectó rostro en ningún frame de prueba. Se usará recorte central por defecto.")

    return x1, y1, x2, y2, rostro_encontrado


def codificar_video(ruta_entrada_video, x1, y1, x2, y2):
    """Recorta/redimensiona el video usando coordenadas YA CALCULADAS
    (no vuelve a detectar el rostro)."""
    cap = cv2.VideoCapture(ruta_entrada_video)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el formato de este video: {ruta_entrada_video}")
        return

    nombre_archivo = os.path.basename(ruta_entrada_video)
    ruta_salida = construir_ruta_salida(ruta_entrada_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    comando_ffmpeg = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{TAMANO_SALIDA}x{TAMANO_SALIDA}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-pix_fmt', 'yuv420p',
        ruta_salida
    ]

    proceso = subprocess.Popen(comando_ffmpeg, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frames_procesados = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_recortado = frame[y1:y2, x1:x2]
        if frame_recortado.size == 0:
            continue
        frame_reseteado = cv2.resize(frame_recortado, (TAMANO_SALIDA, TAMANO_SALIDA))

        try:
            proceso.stdin.write(frame_reseteado.astype(np.uint8).tobytes())
            frames_procesados += 1
        except (BrokenPipeError, OSError):
            print("    ❌ FFmpeg cerró el pipe inesperadamente.")
            break

    cap.release()
    proceso.stdin.close()
    stderr_output = proceso.stderr.read().decode(errors="ignore")
    proceso.wait()

    if proceso.returncode != 0:
        print(f"    ❌ FFmpeg falló (código {proceso.returncode}): {stderr_output.strip()[-400:]}")
    elif not os.path.exists(ruta_salida) or os.path.getsize(ruta_salida) == 0:
        print(f"    ❌ El archivo de salida no se generó o quedó vacío.")


# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    grupos = obtener_grupos_por_subcarpeta(video_entrada)

    if not grupos:
        print(f"⚠️ No se encontraron videos válidos en: {video_entrada}")
    else:
        total_grupos = len(grupos)
        for indice, (nombre_carpeta, rutas_carpeta) in enumerate(grupos, start=1):
            nombre_mostrar = nombre_carpeta if nombre_carpeta != "." else "raíz"
            print(f"\n[{indice}/{total_grupos}] 📁 Subcarpeta: {nombre_mostrar} "
                  f"({len(rutas_carpeta)} clip(s))")

            # -----------------------------------------------------------
            # Detectar el rostro UNA SOLA VEZ, usando el primer clip del
            # grupo, y reutilizar ese mismo recorte para todos los demás.
            # -----------------------------------------------------------
            primer_clip = rutas_carpeta[0]
            print("    🔍 Detectando rostro en el primer clip...")
            resultado_deteccion = detectar_recorte_rostro(primer_clip)

            if resultado_deteccion is None:
                print(f"    ⚠️ Se omite toda la subcarpeta '{nombre_mostrar}' "
                      f"(no se pudo abrir el primer clip).")
                continue

            x1, y1, x2, y2, encontrado = resultado_deteccion
            if encontrado:
                print("    ✅ Se usará el recorte detectado del rostro.")
            else:
                print("    ⚠️ No se detectó rostro; se usará el recorte central.")

            # -----------------------------------------------------------
            # Codificar TODOS los clips de la subcarpeta con ese recorte,
            # incluyendo el primero (ya detectado, no se vuelve a analizar).
            # -----------------------------------------------------------
            for ruta_clip in rutas_carpeta:
                codificar_video(ruta_clip, x1, y1, x2, y2)

    print("\n========================================")
    print("¡PROCESAMIENTO COMPLETADO!")
    print(f"Resultados guardados en: {directorio_salida}")
    print("========================================")