"""
=============================================================================
EXTRACCIÓN DE MFCCs PARA RECONOCIMIENTO DE EMOCIONES EN VOZ
=============================================================================
Pipeline aplicado (Liu 2018, Tabla 1):
  1. Pre-énfasis          α = 0.97
  2. Framing              25 ms con salto de 10 ms
  3. Ventana de Hamming
  4. FFT (n=2048)
  5. Banco de filtros Mel (n_mels=40, fmin=80, fmax=8000)
  6. Compresión logarítmica
  7. DCT → 13 MFCCs base
  8. Deltas (Δ) y dobles deltas (ΔΔ) → vector final de 39 coeficientes
  9. Normalización CMN (Cepstral Mean Normalization) por archivo

Representación contextual (Liu 2018, Fig. 1):
  Para cada frame t se concatenan los 39 coeficientes de los frames
  t-9 … t … t+9 (ventana de 19 frames), generando un vector de 741
  dimensiones por frame.

Formato de nombre de archivo esperado:
  01_08_05_1_007_01.wav
  └─ segmento [1] = código de emoción → "08" = enojo

Mapa de emociones:
  00=neutral  01=calmado   02=emocionado  03=alegre  04=sorprendido
  05=nervioso 06=tristeza  07=miedo       08=enojo

Dependencias:
    pip install librosa numpy scipy soundfile tqdm
=============================================================================
"""

import os
import re
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# RUTAS
# =============================================================================
carpeta_limpios = r"D:\Documentos\Ayudante de Investigacion\Codigos\Pruebas limpio"
carpeta_mfcc    = r"D:\Documentos\Ayudante de Investigacion\Codigos\MFCCs"

# =============================================================================
# CONFIGURACIÓN DE NOMBRES
# =============================================================================
SUFIJO_SALIDA = "_07"

# =============================================================================
# MAPA DE EMOCIONES — segundo segmento del nombre de archivo
# Formato: 01_08_05_1_007_01  →  posición [1] = "08" → "enojo"
# =============================================================================
MAPA_EMOCIONES = {
    "00": "neutral",
    "01": "calmado",
    "02": "emocionado",
    "03": "alegre",
    "04": "sorprendido",
    "05": "nervioso",
    "06": "tristeza",
    "07": "miedo",
    "08": "enojo",
}

# =============================================================================
# PARÁMETROS DEL PIPELINE MFCC (Liu 2018, Tabla 1)
# =============================================================================
SR              = 44100
PRE_ENFASIS     = 0.97
N_MFCC          = 13
N_MELS          = 40
N_FFT           = 2048
HOP_LENGTH      = int(SR * 0.010)   # Salto de frame: 10 ms
WIN_LENGTH      = int(SR * 0.025)   # Longitud de frame: 25 ms
FMIN            = 80
FMAX            = 8000
CONTEXTO_FRAMES = 9
CMN_ACTIVO      = True

# =============================================================================
# FUNCIONES
# =============================================================================

def aplicar_pre_enfasis(audio, alpha=PRE_ENFASIS):
    """H(z) = 1 - alpha·z⁻¹  (Liu 2018, Tabla 1, primer paso)."""
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])


def extraer_mfcc_base(audio, sr):
    """
    Pasos 2–8 del pipeline (Liu 2018, Tabla 1):
      Framing → Hamming → FFT → Mel Bank → log → DCT → Δ → ΔΔ

    Retorna array de forma (n_frames, N_MFCC * 3):
      columnas 0:13  = MFCCs base
      columnas 13:26 = deltas  (Δ)
      columnas 26:39 = dobles deltas (ΔΔ)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window='hamming', n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=True,
    )
    delta1   = librosa.feature.delta(mfcc, order=1)
    delta2   = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta1, delta2], axis=0).T
    return features.astype(np.float32)


def aplicar_cmn(features):
    """Cepstral Mean Normalization: resta la media de cada coeficiente."""
    return (features - features.mean(axis=0)).astype(np.float32)


def representacion_contextual(features, k=CONTEXTO_FRAMES):
    """
    Representación contextual de Liu 2018 (Fig. 1).
    Entrada : (T, D)           donde D = N_MFCC * 3 = 39
    Salida  : (T, D * (2k+1)) donde 2k+1 = 19  →  39*19 = 741
    """
    if k == 0:
        return features

    T, D      = features.shape
    ventana   = 2 * k + 1
    resultado = np.zeros((T, D * ventana), dtype=np.float32)

    for t in range(T):
        partes = []
        for offset in range(-k, k + 1):
            idx = t + offset
            partes.append(features[idx] if 0 <= idx < T else np.zeros(D, dtype=np.float32))
        resultado[t] = np.concatenate(partes)

    return resultado


def inferir_emocion(nombre_archivo):
    """
    Extrae la emoción del segundo segmento del nombre de archivo.

    Formato esperado: 01_08_05_1_007_01
      partes[1] = "08" → "enojo"

    Si el segmento no está en el mapa devuelve "desconocida".
    """
    partes = nombre_archivo.split("_")
    try:
        codigo  = partes[1]
        emocion = MAPA_EMOCIONES.get(codigo, "desconocida")
    except IndexError:
        emocion = "desconocida"
    return emocion


def procesar_archivo(ruta_wav):
    """
    Pipeline completo para un archivo:
      1. Carga del audio
      2. Pre-énfasis
      3. MFCCs base + Δ + ΔΔ
      4. CMN (opcional)
      5. Representación contextual
    Retorna (features_ctx, emocion).
    """
    audio, sr = sf.read(ruta_wav, dtype='float32')

    # Asegurar mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resamplear si la fs no coincide
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    audio        = aplicar_pre_enfasis(audio)
    features     = extraer_mfcc_base(audio, SR)
    if CMN_ACTIVO:
        features = aplicar_cmn(features)
    features_ctx = representacion_contextual(features)

    nombre_base  = os.path.splitext(os.path.basename(ruta_wav))[0]
    emocion      = inferir_emocion(nombre_base)

    return features_ctx, emocion


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def extraer_mfcc_masivo():
    if not os.path.exists(carpeta_limpios):
        print("❌ Error: La carpeta de audios limpios no existe.")
        return

    os.makedirs(carpeta_mfcc, exist_ok=True)

    dim_vec = N_MFCC * 3 * (2 * CONTEXTO_FRAMES + 1)

    print("=" * 62)
    print("  EXTRACCIÓN DE MFCCs — CORPUS DE EMOCIONES EN VOZ")
    print("=" * 62)
    print(f"  Pipeline  : pre-énfasis → Mel({N_MELS}) → {N_MFCC} MFCC → Δ → ΔΔ")
    print(f"  Ventana   : {int(WIN_LENGTH/SR*1000)} ms  |  Salto: {int(HOP_LENGTH/SR*1000)} ms  |  FFT: {N_FFT}")
    print(f"  Banda Mel : {FMIN}–{FMAX} Hz")
    print(f"  CMN       : {'SÍ' if CMN_ACTIVO else 'NO'}")
    print(f"  Contexto  : ±{CONTEXTO_FRAMES} frames → vector de {dim_vec} dimensiones")
    print(f"  Formato   : .npz comprimido  |  Sufijo: '{SUFIJO_SALIDA}'")
    print("=" * 62)

    # Recolectar archivos
    lista_tareas = []
    for root, dirs, files in os.walk(carpeta_limpios):
        for archivo in files:
            if archivo.lower().endswith('.wav'):
                lista_tareas.append((root, archivo))

    if not lista_tareas:
        print("⚠️  No se encontraron archivos .wav en la carpeta de limpios.")
        return

    errores          = []
    total_frames     = 0
    emociones_vistas = set()

    for root, archivo in tqdm(lista_tareas, desc="Extrayendo", unit="audio"):
        try:
            ruta_wav     = os.path.join(root, archivo)
            features_ctx, emocion = procesar_archivo(ruta_wav)
            nombre_base  = os.path.splitext(archivo)[0]
            n_frames     = features_ctx.shape[0]

            # Replicar estructura de carpetas con sufijo en cada nivel
            rel_path = os.path.relpath(root, carpeta_limpios)
            if rel_path != ".":
                carpetas_con_sufijo = [f"{c}{SUFIJO_SALIDA}" for c in rel_path.split(os.sep) if c]
                rel_path_salida     = os.path.join(*carpetas_con_sufijo)
            else:
                rel_path_salida = ""

            carpeta_destino = os.path.join(carpeta_mfcc, rel_path_salida)
            os.makedirs(carpeta_destino, exist_ok=True)

            # Guardar .npz — solo frames y emoción
            nombre_npz = f"{nombre_base}{SUFIJO_SALIDA}.npz"
            ruta_npz   = os.path.join(carpeta_destino, nombre_npz)

            np.savez_compressed(
                ruta_npz,
                frames  = features_ctx,       # float32 (n_frames, 741)
                emocion = np.array([emocion]), # str
            )

            total_frames += n_frames
            emociones_vistas.add(emocion)

        except Exception as e:
            errores.append((archivo, str(e)))

    if total_frames > 0:
        print(f"\n✅ Extracción terminada.")
        print(f"   Archivos procesados : {len(lista_tareas) - len(errores)}")
        print(f"   NPZs generados      : {len(lista_tareas) - len(errores)}  → en {carpeta_mfcc}")
        print(f"   Total frames        : {total_frames}")
        print(f"   Emociones vistas    : {sorted(emociones_vistas)}")
    else:
        print("⚠️  No se generaron features.")

    if errores:
        print(f"\n⚠️  Errores ({len(errores)}):")
        for nombre, err in errores:
            print(f"   • {nombre}: {err}")


# =============================================================================
if __name__ == "__main__":
    extraer_mfcc_masivo()