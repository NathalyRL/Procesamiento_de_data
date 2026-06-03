"""
=============================================================================
EXTRACCIÓN DE MFCCs PARA RECONOCIMIENTO DE EMOCIONES EN VOZ
=============================================================================
Entrada : audios limpios generados por limpiar_voces_emocionales.py
Salida  : archivos .npy con MFCCs + deltas + dobles deltas por audio,
          más un archivo resumen corpus_mfcc.npz con todos los vectores
          y sus etiquetas de emoción e intensidad.

Pipeline aplicado (Liu 2018, Tabla 1 — idéntico al paper):
  1. Pre-énfasis          α = 0.97
  2. Framing              25 ms con salto de 10 ms
  3. Ventana de Hamming
  4. FFT (n=512)
  5. Banco de filtros Mel (n_mels=40, fmin=80, fmax=8000)
  6. Compresión logarítmica
  7. DCT → 13 MFCCs base
  8. Deltas (Δ)  y dobles deltas (ΔΔ) → vector final de 39 coeficientes
  9. Normalización CMN (Cepstral Mean Normalization) por archivo

Representación contextual (Liu 2018, Fig. 1):
  Para cada frame t se concatenan los 39 coeficientes de los frames
  t-9 … t … t+9 (ventana de 19 frames), generando un vector de 741
  dimensiones por frame. Esta es exactamente la representación usada
  por Liu como entrada a sus redes FCNN, LSTM y LSTM+A.

Estructura de carpetas esperada en carpeta_limpios:
  carpeta_limpios/
    Enojo_01/
      hombre_01_enojo_normal_01.wav
      mujer_03_enojo_fuerte_02.wav
      ...
    Alegre_01/
      ...

  El nombre del archivo debe contener:
    - género  : 'hombre' o 'mujer'
    - emoción : nombre de la carpeta padre (ej. Enojo, Alegre)
    - intensidad: 'normal' o 'fuerte' (si aplica; si no, se asigna 'normal')

Salida (carpeta_mfcc/corpus_mfcc.csv):
  Una fila por frame. Columnas:
    archivo   — nombre del archivo de audio origen
    emocion   — etiqueta de emoción (ej. enojo, alegre)
    intensidad— etiqueta de intensidad (normal / fuerte)
    frame     — índice del frame dentro del archivo
    mfcc_0 … mfcc_740 — los 741 coeficientes contextuales

AJUSTE RÁPIDO:
  • Más coeficientes base : subir N_MFCC (ej: 20). El vector pasa a N_MFCC×3×19.
  • Ventana contextual     : cambiar CONTEXTO_FRAMES (9 → 5 para menos dimensiones)
  • Sin representación ctx : poner CONTEXTO_FRAMES = 0 → vector de 39 por frame
  • Sin CMN               : poner CMN_ACTIVO = False

Dependencias:
    pip install librosa numpy scipy soundfile tqdm
=============================================================================
"""

import os
import re
import csv
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# RUTAS
# =============================================================================
carpeta_limpios = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_Limpios"
carpeta_mfcc    = r"D:\Documentos\Ayudante de Investigacion\Codigos\MFCCs"
ARCHIVO_CSV     = os.path.join(carpeta_mfcc, "corpus_mfcc.csv")

# =============================================================================
# CONFIGURACIÓN DE NOMBRES
# =============================================================================
SUFIJO_SALIDA   = "_07"                                          
NOMBRE_BASE_CSV = "corpus_mfcc"                                  
ARCHIVO_CSV     = os.path.join(                                  
    carpeta_mfcc, f"{NOMBRE_BASE_CSV}{SUFIJO_SALIDA}.csv"        
)  

# =============================================================================
# PARÁMETROS DEL PIPELINE MFCC (Liu 2018, Tabla 1)
# =============================================================================
SR             = 44100    # Frecuencia de muestreo — igual que en la limpieza
PRE_ENFASIS    = 0.97     # Coeficiente de pre-énfasis
N_MFCC         = 13       # Coeficientes base (13 = estándar; Liu usa 13)
N_MELS         = 40       # Filtros Mel en el banco
N_FFT          = 512      # Tamaño de FFT
HOP_LENGTH     = int(SR * 0.010)   # Salto de frame: 10 ms
WIN_LENGTH     = int(SR * 0.025)   # Longitud de frame: 25 ms
FMIN           = 80       # Frecuencia mínima del banco Mel (= HPF limpieza)
FMAX           = 8000     # Frecuencia máxima del banco Mel

# Representación contextual (Liu 2018, Fig. 1)
# Frame t usa contexto [t-CONTEXTO_FRAMES .. t .. t+CONTEXTO_FRAMES]
# Con 9 → ventana de 19 frames → vector de 13×3×19 = 741 dimensiones
CONTEXTO_FRAMES = 9

# Normalización cepstral por archivo (resta la media de cada coeficiente)
CMN_ACTIVO = True

# =============================================================================
# ETIQUETAS DE EMOCIÓN — ajusta según tus nombres de carpeta
# =============================================================================
EMOCIONES_VALIDAS = {
    "neutral", "calmado", "emocionado", "alegre",
    "sorprendido", "nervioso", "tristeza", "miedo", "enojo"
}

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
      columnas 0:13   = MFCCs base
      columnas 13:26  = deltas  (Δ)
      columnas 26:39  = dobles deltas (ΔΔ)
    """
    # Banco de filtros Mel + log + DCT → MFCCs base (shape: n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(
        y         = audio,
        sr        = sr,
        n_mfcc    = N_MFCC,
        n_fft     = N_FFT,
        hop_length= HOP_LENGTH,
        win_length= WIN_LENGTH,
        window    = 'hamming',
        n_mels    = N_MELS,
        fmin      = FMIN,
        fmax      = FMAX,
        center    = True,
    )

    # Deltas y dobles deltas
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Concatenar → (n_mfcc*3, n_frames) → transponer → (n_frames, n_mfcc*3)
    features = np.concatenate([mfcc, delta1, delta2], axis=0).T
    return features.astype(np.float32)


def aplicar_cmn(features):
    """
    Cepstral Mean Normalization (CMN): resta la media de cada coeficiente
    calculada sobre todos los frames del archivo. Reduce el efecto del canal
    y del micrófono, haciendo los MFCCs más robustos entre grabaciones.
    """
    return (features - features.mean(axis=0)).astype(np.float32)


def representacion_contextual(features, k=CONTEXTO_FRAMES):
    """
    Construye la representación contextual de Liu 2018 (Fig. 1).

    Para cada frame t (0 ≤ t < T) concatena los vectores de
    [t-k, t-k+1, ..., t, ..., t+k-1, t+k], rellenando con ceros
    los frames fuera de rango (padding).

    Entrada : (T, D)              donde D = N_MFCC * 3 = 39
    Salida  : (T, D * (2k+1))    donde 2k+1 = 19  →  39*19 = 741
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
            if 0 <= idx < T:
                partes.append(features[idx])
            else:
                partes.append(np.zeros(D, dtype=np.float32))
        resultado[t] = np.concatenate(partes)

    return resultado


def inferir_etiquetas(nombre_archivo, carpeta_padre):
    """
    Extrae emoción e intensidad desde el nombre de carpeta y de archivo.

    Emoción   : nombre de la carpeta padre (ej. 'Enojo_01' → 'enojo')
    Intensidad: busca 'normal' o 'fuerte' en el nombre del archivo;
                si no encuentra ninguno, asigna 'normal' por defecto.
    """
    emocion   = re.sub(r'_\d+$', '', carpeta_padre).lower().strip()
    nombre_l  = nombre_archivo.lower()
    if 'fuerte' in nombre_l or 'strong' in nombre_l or 'high' in nombre_l:
        intensidad = 'fuerte'
    else:
        intensidad = 'normal'
    return emocion, intensidad


def procesar_archivo(ruta_wav, carpeta_padre):
    """
    Pipeline completo para un archivo:
      1. Carga del audio limpio
      2. Pre-énfasis
      3. MFCCs base + Δ + ΔΔ
      4. CMN (opcional)
      5. Representación contextual
    Retorna (features_ctx, emocion, intensidad).
    """
    audio, sr = sf.read(ruta_wav, dtype='float32')

    # Asegurar mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resamplear si la fs no coincide (no debería ocurrir, pero por seguridad)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    # 1. Pre-énfasis
    audio = aplicar_pre_enfasis(audio)

    # 2–8. MFCCs + deltas
    features = extraer_mfcc_base(audio, SR)

    # 9. CMN
    if CMN_ACTIVO:
        features = aplicar_cmn(features)

    # 10. Representación contextual (Liu 2018, Fig. 1)
    features_ctx = representacion_contextual(features)

    # Etiquetas
    nombre_base  = os.path.splitext(os.path.basename(ruta_wav))[0]
    emocion, intensidad = inferir_etiquetas(nombre_base, carpeta_padre)

    return features_ctx, emocion, intensidad


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
    print("=" * 62)

    # Recolectar archivos
    lista_tareas = []
    for root, dirs, files in os.walk(carpeta_limpios):
        carpeta_padre = os.path.basename(root)
        for archivo in files:
            if archivo.lower().endswith('.wav'):
                lista_tareas.append((root, carpeta_padre, archivo))

    if not lista_tareas:
        print("⚠️  No se encontraron archivos .wav en la carpeta de limpios.")
        return

    # Construir cabecera del CSV
    cols_mfcc = [f"mfcc_{i}" for i in range(dim_vec)]        
    cabecera  = ["frame"] + cols_mfcc   

    errores          = []
    total_frames     = 0
    emociones_vistas = set()

    os.makedirs(carpeta_mfcc, exist_ok=True)

    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as f_csv:
        escritor = csv.writer(f_csv)
        escritor.writerow(cabecera)

        for root, carpeta_padre, archivo in tqdm(lista_tareas, desc="Extrayendo", unit="audio"):
            try:
                ruta_wav    = os.path.join(root, archivo)
                features_ctx, emocion, intensidad = procesar_archivo(ruta_wav, carpeta_padre)
                nombre_base = os.path.splitext(archivo)[0]
                n_frames    = features_ctx.shape[0]

                for idx in range(n_frames):
                    fila = [idx] + features_ctx[idx].tolist()
                    escritor.writerow(fila)

                total_frames += n_frames
                emociones_vistas.add(emocion)

            except Exception as e:
                errores.append((archivo, str(e)))

    if total_frames > 0:
        tam_mb = os.path.getsize(ARCHIVO_CSV) / (1024 * 1024)
        print(f"\n✅ Extracción terminada.")
        print(f"   Archivos procesados : {len(lista_tareas) - len(errores)}")
        print(f"   Total frames        : {total_frames}")
        print(f"   Columnas por fila   : {len(cabecera)}  (4 etiquetas + {dim_vec} coeficientes)")
        print(f"   Emociones           : {sorted(emociones_vistas)}")
        print(f"   CSV generado        : {ARCHIVO_CSV}")
        print(f"   Tamaño del CSV      : {tam_mb:.1f} MB")
    else:
        print("⚠️  No se generaron features.")

    if errores:
        print(f"\n⚠️  Errores ({len(errores)}):")
        for nombre, err in errores:
            print(f"   • {nombre}: {err}")


# =============================================================================
if __name__ == "__main__":
    extraer_mfcc_masivo()