"""
=============================================================================
MÓDULO: Limpieza y Depuración Automatizada de Audio por Umbral de RMS
=============================================================================

DESCRIPCIÓN:
    Este script realiza un escaneo recursivo en un directorio específico para
    identificar, analizar y opcionalmente eliminar archivos de audio en formato
    '.wav' que se consideren "sospechosos" debido a un bajo nivel de energía.
    
    El criterio de selección se basa en el cálculo del valor RMS (Root Mean Square).
    Si un archivo es estéreo o multicanal, se promedia a mono antes del análisis
    para garantizar la consistencia de la métrica.

FUNCIONAMIENTO / FLUJO:
    1. Escaneo exhaustivo (os.walk) de la carpeta de audios limpios.
    2. Conversión interna a tipo de dato float32 y unificación a canal mono.
    3. Extracción de métricas clave por archivo: RMS, Valor Pico y Duración (seg).
    4. Clasificación y ordenamiento ascendente según el nivel de RMS.
    5. Despliegue de un reporte estadístico general y listado de sospechosos.
    6. Acción correctiva: Muestra de control en 'Modo Seguro' o eliminación 
       física de los archivos que no superen el umbral establecido.

CONFIGURACIÓN DE VARIABLES:
    * carpeta (str): Ruta absoluta del directorio raíz que contiene los audios.
    * UMBRAL_RMS_BAJO (float): Límite inferior de RMS. Valores menores se marcan 
      como silencios o audios defectuosos.
    * MODO_SEGURO (bool): 
        - True: (Simulación) Solo lista los archivos que serían borrados.
        - False: (Ejecución) Elimina permanentemente los archivos sospechosos.

PROTOCOLO DE OPERACIÓN SEGURA:
    Para evitar la pérdida accidental de datos, siga estrictamente estos pasos:
    
    * PASO 1 — Ejecute el script con MODO_SEGURO = True y revise la lista de 
               archivos sospechosos desplegada en la pantalla.
    * PASO 2 — Si tras la revisión confirma que los archivos listados son correctos 
               para eliminar, cambie a MODO_SEGURO = False y vuelva a ejecutarlo.
               
    Nota: Este flujo garantiza que ningún archivo sea eliminado del sistema sin 
          haber sido validado previamente en el reporte previo.

DEPENDENCIAS:
    - os: Navegación por el sistema de archivos y borrado de elementos.
    - numpy (np): Operaciones matemáticas vectorizadas (raíz, media, absoluto, pico).
    - soundfile (sf): Lectura robusta de señales de audio y tasas de muestreo.
=============================================================================
"""

import numpy as np
import soundfile as sf
import os

carpeta = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_Limpio"

# =============================================================================
# UMBRAL — archivos con RMS menor a esto se consideran sospechosos
# =============================================================================
UMBRAL_RMS_BAJO = 0.0017

# =============================================================================
# MODO SEGURO — True: solo muestra qué eliminaría sin borrar nada
#               False: elimina los archivos sospechosos
# =============================================================================
MODO_SEGURO = False

# =============================================================================
resultados     = []
archivos_error = []

for root, dirs, files in os.walk(carpeta):
    for archivo in files:
        if archivo.lower().endswith('.wav'):
            ruta = os.path.join(root, archivo)
            try:
                audio, sr = sf.read(ruta, dtype='float32')
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                rms          = np.sqrt(np.mean(audio ** 2))
                duracion_seg = len(audio) / sr
                pico         = np.max(np.abs(audio))

                resultados.append({
                    "ruta"        : ruta,
                    "archivo"     : archivo,
                    "rms"         : rms,
                    "pico"        : pico,
                    "duracion_seg": duracion_seg,
                })
            except Exception as e:
                archivos_error.append((archivo, str(e)))

# Ordenar por RMS ascendente
resultados.sort(key=lambda x: x["rms"])

# =============================================================================
# REPORTE PREVIO
# =============================================================================
todos_rms    = [r["rms"] for r in resultados]
sospechosos  = [r for r in resultados if r["rms"] < UMBRAL_RMS_BAJO]

print("=" * 70)
print("  ANÁLISIS DE RMS")
print("=" * 70)
print(f"  Archivos analizados : {len(resultados)}")
print(f"  RMS mínimo          : {min(todos_rms):.4f}")
print(f"  RMS máximo          : {max(todos_rms):.4f}")
print(f"  RMS promedio        : {np.mean(todos_rms):.4f}")
print(f"  RMS mediana         : {np.median(todos_rms):.4f}")
print(f"  Umbral sospechosos  : {UMBRAL_RMS_BAJO}")
print(f"  Sospechosos found   : {len(sospechosos)}")

# =============================================================================
# LISTADO DE SOSPECHOSOS
# =============================================================================
print("\n" + "=" * 70)
print(f"  ARCHIVOS SOSPECHOSOS  (RMS < {UMBRAL_RMS_BAJO})")
print("=" * 70)

if not sospechosos:
    print("  ✅ Ningún archivo por debajo del umbral.")
else:
    print(f"  {'Archivo':<40} {'RMS':>8}  {'Pico':>8}  {'Duración':>10}")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*10}")
    for r in sospechosos:
        print(f"  {r['archivo']:<40} {r['rms']:>8.4f}  {r['pico']:>8.4f}  {r['duracion_seg']:>8.2f} s")

# =============================================================================
# ELIMINACIÓN
# =============================================================================
print("\n" + "=" * 70)
if MODO_SEGURO:
    print("  MODO SEGURO ACTIVO — no se eliminó ningún archivo")
    print("  Cambia MODO_SEGURO = False para eliminar los sospechosos")
else:
    print("  ELIMINANDO ARCHIVOS SOSPECHOSOS...")
    print("=" * 70)
    eliminados = []
    errores_eliminacion = []

    for r in sospechosos:
        try:
            os.remove(r["ruta"])
            eliminados.append(r["archivo"])
            print(f"  🗑️  Eliminado: {r['archivo']}  (RMS: {r['rms']:.4f})")
        except Exception as e:
            errores_eliminacion.append((r["archivo"], str(e)))
            print(f"  ❌  Error al eliminar {r['archivo']}: {e}")

    print("\n" + "=" * 70)
    print(f"  ✅ Eliminados : {len(eliminados)}")
    print(f"  ❌ Errores    : {len(errores_eliminacion)}")

print("=" * 70)

if archivos_error:
    print(f"\n⚠️  Errores de lectura ({len(archivos_error)}):")
    for nombre, err in archivos_error:
        print(f"   • {nombre}: {err}")