import av
import cv2
import numpy as np
import os
import time

sufijo = "_01" 
video_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_I"
directorio_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\CEL_GOPRO_REDIMENSION"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

TAMANO_SALIDA = 224
FRAME_SEGURO = 23      # primer frame donde se intenta detectar el rostro
MAX_INTENTOS = 4       # probará FRAME_SEGURO, 2x, 3x, 4x antes de rendirse
MARGEN_ROSTRO = 2.2

# Codificación de salida (liviano; ajustar THREADS según el hardware final)
PRESET = "ultrafast"
CRF = "20"
THREADS = "2"


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


def obtener_info_video(ruta_video):
    """Devuelve (ancho, alto, fps, total_frames_estimado, rotacion) usando PyAV.
    ancho/alto ya vienen ajustados según la rotación (si el video es vertical
    por metadata, se devuelven intercambiados para que coincidan con el frame
    ya rotado que se va a procesar)."""
    with av.open(ruta_video) as contenedor:
        stream = contenedor.streams.video[0]
        ancho = stream.codec_context.width
        alto = stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else 25.0

        # Metadata de rotación (típica en videos grabados con celular)
        rotacion = 0
        valor_rotate = stream.metadata.get("rotate")
        if valor_rotate is not None:
            try:
                rotacion = int(valor_rotate) % 360
            except ValueError:
                rotacion = 0

        if stream.frames and stream.frames > 0:
            total_frames = stream.frames
        elif stream.duration and stream.time_base:
            total_frames = int(float(stream.duration * stream.time_base) * fps)
        else:
            total_frames = int(float(contenedor.duration / av.time_base) * fps) if contenedor.duration else 0

    # Si la rotación es de 90 o 270, el ancho/alto visual queda intercambiado
    if rotacion in (90, 270):
        ancho, alto = alto, ancho

    return ancho, alto, fps, total_frames, rotacion


def aplicar_rotacion(frame_bgr, rotacion):
    """Rota el frame según la metadata del video, para que quede en la
    orientación correcta antes de cualquier detección o recorte."""
    if rotacion == 90:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotacion == 180:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    elif rotacion == 270:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


def detectar_recorte_rostro(ruta_video, ancho, alto, total_frames, fps, rotacion, face_cascade):
    """Prueba FRAME_SEGURO, 2x, 3x, 4x... en UNA SOLA pasada secuencial del
    video (no reabre/re-decodifica desde el inicio por cada intento, que era
    la causa de la lentitud extrema en videos de alta resolución con pocos
    keyframes). Devuelve (x1, y1, lado, encontrado)."""
    lado_central = min(alto, ancho)
    x1_def = (ancho - lado_central) // 2
    y1_def = (alto - lado_central) // 2

    objetivos = [FRAME_SEGURO * i for i in range(1, MAX_INTENTOS + 1)]
    if total_frames:
        objetivos = [f for f in objetivos if f < total_frames]
    if not objetivos:
        return x1_def, y1_def, lado_central, False

    max_objetivo = objetivos[-1]

    with av.open(ruta_video) as contenedor:
        stream = contenedor.streams.video[0]
        # Decodificación multi-hilo: acelera mucho HEVC de alta resolución
        stream.codec_context.thread_type = "AUTO"

        idx = -1
        for frame in contenedor.decode(stream):
            idx += 1
            if idx > max_objetivo:
                break
            if idx not in objetivos:
                continue

            frame_bgr = frame.to_ndarray(format="bgr24")
            frame_bgr = aplicar_rotacion(frame_bgr, rotacion)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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
                if idx != objetivos[0]:
                    print(f"    ✓ Rostro detectado en frame {idx}.")
                return x1, y1, lado, True
            else:
                print(f"    ⚠️ Sin rostro en frame {idx}, probando siguiente...")

    return x1_def, y1_def, lado_central, False


def procesar_video(ruta_entrada, face_cascade):
    nombre_archivo = os.path.basename(ruta_entrada)
    nombre_sin_extension, _ = os.path.splitext(nombre_archivo)

    print(f"\n>>> Procesando: {nombre_archivo}")

    try:
        ancho, alto, fps, total_frames, rotacion = obtener_info_video(ruta_entrada)
    except Exception as e:
        print(f"❌ No se pudo abrir el video: {e}")
        return

    if rotacion:
        print(f"    ℹ️ Rotación detectada: {rotacion}° (se corrige automáticamente)")

    print(f"🔍 Analizando '{nombre_archivo}' ({ancho}x{alto}, {fps:.1f}fps)...")
    x1, y1, lado, rostro_encontrado = detectar_recorte_rostro(
        ruta_entrada, ancho, alto, total_frames, fps, rotacion, face_cascade
    )

    if not rostro_encontrado:
        print("⚠️ No se detectó rostro en ningún frame de prueba. Se usará recorte central por defecto.")

    os.makedirs(directorio_salida, exist_ok=True)
    ruta_salida = os.path.join(directorio_salida, f"{nombre_sin_extension}{sufijo}.mp4")

    # =========================================================================
    # Decodificación secuencial completa + recorte/resize + codificación,
    # todo dentro de PyAV (sin cv2.VideoWriter, sin subprocess a ffmpeg.exe).
    # =========================================================================
    inicio = time.time()
    frames_procesados = 0
    try:
        with av.open(ruta_entrada) as entrada, av.open(ruta_salida, mode="w") as salida:
            stream_in = entrada.streams.video[0]
            stream_in.codec_context.thread_type = "AUTO"  # decodificación multi-hilo

            # Usar el framerate como Fraction (no float): PyAV lo requiere así
            # para add_stream, si no lanza 'float' object has no attribute 'numerator'.
            rate_stream = stream_in.average_rate or fps
            stream_out = salida.add_stream("libx264", rate=rate_stream)
            stream_out.width = TAMANO_SALIDA
            stream_out.height = TAMANO_SALIDA
            stream_out.pix_fmt = "yuv420p"
            stream_out.options = {"preset": PRESET, "crf": CRF, "threads": THREADS}

            for frame in entrada.decode(stream_in):
                img = frame.to_ndarray(format="bgr24")
                img = aplicar_rotacion(img, rotacion)
                recorte = img[y1:y1 + lado, x1:x1 + lado]
                if recorte.size == 0:
                    continue
                redimensionado = cv2.resize(recorte, (TAMANO_SALIDA, TAMANO_SALIDA))

                frame_salida = av.VideoFrame.from_ndarray(redimensionado, format="bgr24")
                for paquete in stream_out.encode(frame_salida):
                    salida.mux(paquete)
                frames_procesados += 1

            # Vaciar el buffer del encoder al terminar
            for paquete in stream_out.encode():
                salida.mux(paquete)
    except Exception as e:
        print(f"    ❌ Error durante la codificación: {e}")
        return

    duracion = time.time() - inicio
    print(f"✅ ¡Listo! {frames_procesados} frames en {duracion:.1f}s → {ruta_salida}")
    return rostro_encontrado


if __name__ == "__main__":
    rutas = obtener_rutas_a_procesar(video_entrada)

    videos_recorte_central = []

    if not rutas:
        print(f"⚠️ No se encontraron videos válidos en: {video_entrada}")
    else:
        total = len(rutas)
        print(f"Se encontraron {total} videos para procesar.\n")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for idx, ruta in enumerate(rutas, start=1):
            print(f"[{idx}/{total}]", end=" ")
            try:
                rostro_ok = procesar_video(ruta, face_cascade)
                if rostro_ok is False:
                    videos_recorte_central.append(os.path.basename(ruta))
            except Exception as e:
                print(f"❌ Error inesperado procesando {os.path.basename(ruta)}: {e}")

        if videos_recorte_central:
            ruta_log = os.path.join(directorio_salida, "_videos_revisar_recorte_central.txt")
            with open(ruta_log, "w", encoding="utf-8") as f:
                f.write("Videos que NO detectaron rostro y usaron recorte central:\n\n")
                f.write("\n".join(videos_recorte_central))
            print(f"\n⚠️ {len(videos_recorte_central)} video(s) usaron recorte central. "
                  f"Lista guardada en: {ruta_log}")

        print("\n========================================")
        print("¡PROCESAMIENTO COMPLETADO!")
        print(f"Videos guardados en: {directorio_salida}")
        print("========================================")