import cv2
import os

sufijo = "_01"
video_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO_F"
directorio_salida = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO_F_REDIMENSION2"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

FRAME_SEGURO = 25      # primer frame donde se intenta detectar el rostro
MAX_INTENTOS = 5       # probará FRAME_SEGURO, 2x, 3x, 4x, 5x antes de rendirse


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

    # Coordenadas de respaldo (Recorte central)
    x1, y1 = (ancho - min(alto, ancho)) // 2, (alto - min(alto, ancho)) // 2
    x2, y2 = x1 + min(alto, ancho), y1 + min(alto, ancho)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostro_encontrado = False

    print(f"🔍 Analizando '{nombre_archivo}'...")

    # =====================================================================
    # Reintento de detección: prueba FRAME_SEGURO, 2x, 3x... hasta encontrar
    # rostro o quedarse sin frames del video. Evita que un solo fotograma
    # con mala suerte (alguien cruzándose, movimiento brusco, etc.) arruine
    # la detección de todo el video.
    # =====================================================================
    for intento in range(1, MAX_INTENTOS + 1):
        frame_objetivo = FRAME_SEGURO * intento

        if frame_objetivo >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame_deteccion = cap.read()

        if not ret:
            break

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame_deteccion, cv2.COLOR_BGR2GRAY)

        # OPTIMIZACIÓN: Reducir la imagen al 50% solo para que el detector sea ultra rápido
        gray_pequeño = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        # Buscamos el rostro en la imagen pequeña (ajustamos minSize a la mitad también)
        rostros = face_cascade.detectMultiScale(gray_pequeño, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        if len(rostros) > 0:
            # Multiplicamos por 2 para devolver las coordenadas a su tamaño original
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

    # 3. REBOBINAR al inicio para procesar el video completo desde el frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Configurar la salida de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (224, 224))

    print(f"🎬 Redimensionando fotogramas a 224x224...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_recortado = frame[y1:y2, x1:x2]
        frame_reseteado = cv2.resize(frame_recortado, (224, 224))
        out.write(frame_reseteado)

    cap.release()
    out.release()
    print(f"✅ ¡Listo! Video guardado en: {ruta_salida}\n")


# =====================================================================
# EJECUCIÓN
# =====================================================================
rutas = obtener_rutas_a_procesar(video_entrada)

if not rutas:
    print(f"⚠️ No se encontraron videos válidos en: {video_entrada}")
else:
    for ruta in rutas:
        procesar_video_ia(ruta)