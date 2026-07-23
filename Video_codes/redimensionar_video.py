# REDIMESIONA VIDEOS - Ingresa data cruda de video y genera un video cuadrado de 224x224 centrado en el rostro
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

sufijo = "_01"
video_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_CORTES"
directorio_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL GOPRO\CEL_GOPRO_REDIMENSION"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

FRAME_SEGURO = 25 # primer frame donde se intenta detectar el rostro
MAX_INTENTOS = 5       # probará FRAME_SEGURO, 2x, 3x, 4x, 5x antes de rendirse

TAMANO_SALIDA = 224
FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = "15"       # 15 = margen extra de calidad (prácticamente sin pérdida perceptual)


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg",) if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Revisa la variable 'ruta_bin_ffmpeg' al inicio del script.")
        return False
    return True


def obtener_rutas_a_procesar(ruta_entrada):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: La ruta '{ruta_entrada}' no existe. Verifica la dirección.")
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


def procesar_video_ia(ruta_entrada):
    if not os.path.exists(ruta_entrada):
        print(f"❌ Error: El archivo '{ruta_entrada}' no existe. Verifica la ruta.")
        return

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el formato de este video: {ruta_entrada}")
        return

    nombre_archivo = os.path.basename(ruta_entrada)
    nombre_sin_extension, _ = os.path.splitext(nombre_archivo)

    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida, exist_ok=True)

    ruta_salida = os.path.join(directorio_salida, f"{nombre_sin_extension}{sufijo}.mp4")

    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Coordenadas de respaldo (Recorte central)
    x1, y1 = (ancho - min(alto, ancho)) // 2, (alto - min(alto, ancho)) // 2
    x2, y2 = x1 + min(alto, ancho), y1 + min(alto, ancho)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostro_encontrado = False

    print(f"🔍 Analizando '{nombre_archivo}'...")

    # =====================================================================
    # Reintento de detección: prueba FRAME_SEGURO, 2x, 3x... hasta encontrar
    # rostro o quedarse sin frames del video.
    # =====================================================================
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

    if not rostro_encontrado:
        print("⚠️ No se detectó rostro en ningún frame de prueba. Se usará recorte central por defecto.")

    # Rebobinar al inicio para procesar el video completo desde el frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # =========================================================================
    # Codificación con FFmpeg (libx264) en vez de cv2.VideoWriter/mp4v,
    # que era la causa de la pérdida de calidad visible en el resultado.
    # =========================================================================
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

    print(f"🎬 Redimensionando fotogramas a {TAMANO_SALIDA}x{TAMANO_SALIDA}...")
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
    else:
        print(f"✅ ¡Listo! {frames_procesados} frames guardados en: {ruta_salida}\n")


# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    if not verificar_dependencias():
        raise SystemExit(1)

    rutas = obtener_rutas_a_procesar(video_entrada)

    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {video_entrada}")
    else:
        total = len(rutas)
        print(f"Se encontraron {total} videos para procesar.\n")
        for idx, ruta in enumerate(rutas, start=1):
            print(f"[{idx}/{total}]", end=" ")
            procesar_video_ia(ruta)