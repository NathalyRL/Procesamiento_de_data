"""
=============================================================================
NORMALIZACIÓN RMS DE AUDIOS
=============================================================================
Entrada : audios en carpeta_entrada (recorre subcarpetas)
Salida  : audios normalizados en carpeta_salida, replicando la estructura
          de carpetas y aplicando SUFIJO a cada carpeta y archivo.

¿Qué hace?
  Para cada archivo .wav calcula su RMS actual, luego escala todas las
  muestras por el factor (RMS_OBJETIVO / RMS_actual) de modo que el
  audio resultante tenga exactamente el nivel de energía deseado.

Parámetros clave:
  RMS_OBJETIVO : nivel de energía al que se normalizan todos los audios.
                 Valor típico: 0.1 (rango de audio float32: -1.0 a 1.0)
  SUFIJO       : sufijo que se añade a carpetas y archivos de salida (_06 para RMS.
 
¿Por qué RMS y no Peak?
  - Peak normaliza al valor máximo instantáneo. Una emoción como Enojo
    tiene picos muy altos pero también silencios; normalizar por peak
    la deja con energía promedio mucho más baja que Tristeza (sostenida).
  - RMS normaliza a la energía promedio real, que es lo que el oído
    percibe como volumen. Esto iguala el "peso energético" entre
    emociones y entre grabaciones de distintos micrófonos o sesiones.
  - Para redes neuronales que aprenden de MFCCs/GFCCs, diferencias de
    volumen entre archivos son ruido no deseado; RMS lo elimina.
=============================================================================
"""

import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# RUTAS
# =============================================================================
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_Limpio"
carpeta_salida  = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_RMS"

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
SUFIJO       = "_06"   # ← cambia aquí para versionar la salida
RMS_OBJETIVO = 0.02     # nivel de energía objetivo (float32, rango 0.0–1.0)
SR_OBJETIVO  = 44100   # frecuencia de muestreo de salida

# =============================================================================
# FUNCIONES
# =============================================================================

def calcular_rms(audio):
    """Calcula el RMS (energía promedio) de la señal."""
    return np.sqrt(np.mean(audio ** 2))

def normalizar_rms(audio, rms_objetivo=RMS_OBJETIVO):
    """
    Escala el audio para que su RMS iguale rms_objetivo.
    Si el audio es silencio (RMS = 0) lo devuelve sin cambios.
    """
    rms_actual = calcular_rms(audio)
    if rms_actual == 0:
        return audio
    factor = rms_objetivo / rms_actual
    return (audio * factor).astype(np.float32)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def normalizar_rms_masivo():
    if not os.path.exists(carpeta_entrada):
        print("❌ Error: La carpeta de entrada no existe.")
        return

    os.makedirs(carpeta_salida, exist_ok=True)

    # Recolectar archivos
    lista_tareas = []
    for root, dirs, files in os.walk(carpeta_entrada):
        for archivo in files:
            if archivo.lower().endswith('.wav'):
                lista_tareas.append((root, archivo))

    if not lista_tareas:
        print("⚠️  No se encontraron archivos .wav.")
        return

    print("=" * 62)
    print("  NORMALIZACIÓN RMS DE AUDIOS")
    print("=" * 62)
    print(f"  RMS objetivo : {RMS_OBJETIVO}")
    print(f"  Sufijo       : '{SUFIJO}'")
    print(f"  Archivos     : {len(lista_tareas)}")
    print("=" * 62)

    errores         = []
    total_procesados = 0

    for root, archivo in tqdm(lista_tareas, desc="Normalizando", unit="audio"):
        try:
            # Cargar
            ruta_entrada = os.path.join(root, archivo)
            audio, sr    = sf.read(ruta_entrada, dtype='float32')

            # Asegurar mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Normalizar RMS
            rms_antes  = calcular_rms(audio)
            audio_norm = normalizar_rms(audio)
            rms_despues = calcular_rms(audio_norm)

            # Replicar estructura de carpetas con sufijo en cada nivel
            rel_path = os.path.relpath(root, carpeta_entrada)
            if rel_path != ".":
                carpetas_con_sufijo = [f"{c}{SUFIJO}" for c in rel_path.split(os.sep) if c]
                rel_path_salida     = os.path.join(*carpetas_con_sufijo)
            else:
                rel_path_salida = ""

            carpeta_destino = os.path.join(carpeta_salida, rel_path_salida)
            os.makedirs(carpeta_destino, exist_ok=True)

            # Guardar
            nombre_base  = os.path.splitext(archivo)[0]
            nombre_salida = f"{nombre_base}{SUFIJO}.wav"
            ruta_salida  = os.path.join(carpeta_destino, nombre_salida)
            sf.write(ruta_salida, audio_norm, sr)

            total_procesados += 1

        except Exception as e:
            errores.append((archivo, str(e)))

    print(f"\n✅ Normalización terminada.")
    print(f"   Archivos procesados : {total_procesados}")
    print(f"   RMS objetivo        : {RMS_OBJETIVO}")
    print(f"   Carpeta de salida   : {carpeta_salida}")

    if errores:
        print(f"\n⚠️  Errores ({len(errores)}):")
        for nombre, err in errores:
            print(f"   • {nombre}: {err}")


# =============================================================================
if __name__ == "__main__":
    normalizar_rms_masivo()