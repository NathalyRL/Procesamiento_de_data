# CORTAR VIDEOS REDIMENSIONADOS EN FRAGMENTOS DE 5 SEGUNDOS - NO SYNC
import os
import shutil
import subprocess

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
if ruta_bin_ffmpeg and os.path.isdir(ruta_bin_ffmpeg):
    os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg

# CONFIGURACIÓN para tus videos YA redimensionados (con keyframes cada 5s)
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO_F_REDIMENSION"
carpeta_salida_master = r"D:\Documentos\Ayudante de Investigacion\VIDEOS\LENOVO_F_REDIMENSION_CORTES"
extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')
DURACION_FRAGMENTO = 5.0
DURACION_MINIMA_FINAL = 2.0  # si el resto es menor a esto, se descarta


def verificar_dependencias():
    faltantes = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if faltantes:
        print(f"❌ No se encontró en el PATH: {', '.join(faltantes)}.")
        print("   Instala FFmpeg y asegúrate de que esté agregado al PATH de Windows.")
        return False
    return True


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
    except subprocess.CalledProcessError as e:
        print(f"❌ ffprobe falló en {os.path.basename(ruta_video)}: {e.stderr.strip()}")
        return 0.0
    except Exception as e:
        print(f"❌ Error al leer la duración de {os.path.basename(ruta_video)}: {e}")
        return 0.0


def construir_nombre_clip(nombre_base, bloque_idx):
    prefijo, separador, sufijo = nombre_base.rpartition('_')
    if separador and sufijo.isdigit():
        return f"{prefijo}_{bloque_idx:03d}_{sufijo}.mp4"
    return f"{nombre_base}_{bloque_idx:03d}.mp4"


def verificar_duracion_esperada(ruta_clip, duracion_esperada, nombre_clip):
    """Alerta si -c copy generó un clip con duración muy distinta a la esperada
    (indicio de que los keyframes no están alineados a DURACION_FRAGMENTO)."""
    duracion_real = obtener_duracion_con_ffprobe(ruta_clip)
    diferencia = abs(duracion_real - duracion_esperada)
    if diferencia > 0.5:  # más de medio segundo de desfase
        print(f"    ⚠️ {nombre_clip}: duración {duracion_real:.2f}s "
              f"(esperada ~{duracion_esperada:.2f}s). Revisa que el video de entrada "
              f"tenga keyframes cada {DURACION_FRAGMENTO:.0f}s.")


def cortar_video_puro_ffmpeg(ruta_video, carpeta_destino_master):
    nombre_archivo = os.path.basename(ruta_video)
    nombre_base, _ = os.path.splitext(nombre_archivo)

    print(f"\n>>> Fragmentando con FFmpeg (stream copy): {nombre_archivo}")

    duracion_total = obtener_duracion_con_ffprobe(ruta_video)
    if duracion_total == 0.0:
        print("    ⚠️ Se omite este video porque no se pudo leer su duración.")
        return

    carpeta_destino_video = os.path.join(carpeta_destino_master, nombre_base)
    os.makedirs(carpeta_destino_video, exist_ok=True)

    tiempo_actual = 0.0
    bloque_idx = 1
    fragmentos_generados = 0

    while tiempo_actual < duracion_total:
        tiempo_restante = duracion_total - tiempo_actual

        if tiempo_restante < DURACION_MINIMA_FINAL:
            print(f"    ℹ️ Fragmento final de {tiempo_restante:.2f}s descartado "
                  f"(menor a {DURACION_MINIMA_FINAL:.0f}s).")
            break

        nombre_clip = construir_nombre_clip(nombre_base, bloque_idx)
        ruta_final_clip = os.path.join(carpeta_destino_video, nombre_clip)

        duracion_bloque = min(DURACION_FRAGMENTO, tiempo_restante)

        # =====================================================================
        # -c copy: copia los paquetes comprimidos sin decodificar/recodificar.
        # Requiere que el video de entrada tenga keyframes en (o cerca de)
        # los puntos de corte -- por eso el script de redimensión ahora fuerza
        # un keyframe exacto cada DURACION_FRAGMENTO segundos.
        # =====================================================================
        comando = [
            'ffmpeg', '-y',
            '-ss', str(tiempo_actual),   # antes de -i: seek rápido
            '-i', ruta_video,
            '-t', str(duracion_bloque),
            '-c', 'copy',
            '-an',
            ruta_final_clip
        ]

        resultado = subprocess.run(comando, capture_output=True, text=True)

        if resultado.returncode != 0:
            print(f"    ❌ Error generando {nombre_clip}:")
            print(f"       {resultado.stderr.strip()[-500:]}")
        elif not os.path.exists(ruta_final_clip) or os.path.getsize(ruta_final_clip) == 0:
            print(f"    ❌ {nombre_clip} no se generó o quedó vacío.")
        else:
            verificar_duracion_esperada(ruta_final_clip, duracion_bloque, nombre_clip)
            fragmentos_generados += 1

        tiempo_actual += duracion_bloque
        bloque_idx += 1

    print(f"    ✓ {nombre_base}: {fragmentos_generados} fragmentos generados correctamente.")


if __name__ == "__main__":
    import time
    import traceback

    if not verificar_dependencias():
        raise SystemExit(1)

    # Diagnóstico: confirmar que el disco/unidad está realmente accesible
    unidad = os.path.splitdrive(carpeta_entrada)[0]
    print(f"🔧 Verificando unidad '{unidad}'...")
    for intento in range(1, 4):
        if os.path.exists(unidad + "\\"):
            print(f"    ✓ Unidad '{unidad}' accesible (intento {intento}).")
            break
        print(f"    ⚠️ Unidad '{unidad}' no responde todavía, esperando 2s (intento {intento}/3)...")
        time.sleep(2)
    else:
        print(f"❌ La unidad '{unidad}' nunca respondió. Verifica que el disco/unidad de red esté conectado.")
        raise SystemExit(1)

    try:
        if not os.path.isdir(carpeta_entrada):
            print(f"❌ La carpeta de entrada no existe: {carpeta_entrada}")
            raise SystemExit(1)

        os.makedirs(carpeta_salida_master, exist_ok=True)

        # Verificación inmediata: ¿realmente quedó creada?
        if not os.path.isdir(carpeta_salida_master):
            print(f"❌ os.makedirs() no lanzó error pero la carpeta NO existe: {carpeta_salida_master}")
            print("   Esto indica un problema del sistema de archivos/unidad, no del script.")
            raise SystemExit(1)
        print(f"✓ Carpeta de salida confirmada: {carpeta_salida_master}")

        archivos_video = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(extensiones_validas)]
    except Exception:
        print("❌ Error inesperado durante la configuración inicial:")
        traceback.print_exc()
        raise SystemExit(1)

    total = len(archivos_video)
    print(f"Se encontraron {total} videos para fragmentar en:\n  {carpeta_entrada}")

    if total == 0:
        print("⚠️ No se encontraron videos con extensiones válidas "
              f"{extensiones_validas} en esa carpeta.")
        raise SystemExit(0)

    for index, nombre_archivo in enumerate(archivos_video):
        ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
        print(f"[{index + 1}/{total}]", end="")
        try:
            cortar_video_puro_ffmpeg(ruta_completa, carpeta_salida_master)
        except Exception as e:
            print(f"❌ Error inesperado procesando {nombre_archivo}: {e}")
            traceback.print_exc()

    # Verificación final: ¿la carpeta sigue existiendo y tiene contenido?
    if os.path.isdir(carpeta_salida_master):
        contenido = os.listdir(carpeta_salida_master)
        print(f"\n🔍 Verificación final: {len(contenido)} elemento(s) en {carpeta_salida_master}")
    else:
        print(f"\n❌ ALERTA: la carpeta {carpeta_salida_master} desapareció durante la ejecución.")

    print("\n========================================")
    print("¡FRAGMENTACIÓN 100% FFMPEG COMPLETADA!")
    print(f"Los sub-clips organizados están en: {carpeta_salida_master}")
    print("========================================")