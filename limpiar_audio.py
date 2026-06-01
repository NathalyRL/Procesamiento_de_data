"""
=============================================================================
LIMPIEZA DE AUDIO PARA RECONOCIMIENTO DE EMOCIONES EN VOZ
=============================================================================
Basado en:
  - Liu, G.K. (2018): GFCCs para reconocimiento de emociones en habla
  - Dimitrova-Grekow et al. (2019): Frecuencia fundamental (F0) en emociones

Emociones: Neutral, Calmado, Emocionado, Alegre, Sorprendido,
           Nervioso, Tristeza, Miedo, Enojo

Dependencias:
    pip install scipy soundfile numpy pydub tqdm
=============================================================================

PIPELINE (en orden de ejecución):
  1.  Carga y conversión     → mono, 44100 Hz, float32
  2.  Notch 50 Hz            → elimina interferencia de red eléctrica
  3.  HPF 80 Hz              → elimina graves (DC, vibraciones)
  4.  LPF 10 kHz             → elimina ruido ultrasónico
  5.  SS por bandas          → sustracción espectral diferenciada por rango
  6.  Wiener género-adaptivo → SNR local con protección de banda femenina
  7.  Supresor de música     → detecta y atenúa fondo musical por estabilidad temporal
  8.  VAD espectral          → detecta frames sin voz humana y los suprime
  9.  Noise Gate             → silencia residuos bajo umbral de energía
  10. Pre-énfasis            → compensa caída de energía en altas frecuencias
  11. Normalización RMS      → nivela volumen a -23 dBFS

AJUSTE RÁPIDO SI HAY PROBLEMAS:
  • Voz femenina enojada metálica  → subir SS_ALFA_MEDIA (ej: 1.4) o bajar WIENER_SNR_FLOOR
  • Música persiste                → subir MUSICA_ESTABILIDAD_UMBRAL (ej: 0.85)
  • Voz suena recortada            → bajar VAD_UMBRAL_SFM, subir VAD_CONTEXTO_MS
  • Zumbido residual               → subir GATE_UMBRAL_DB (ej: -40)
=============================================================================
"""

import os
import numpy as np
import soundfile as sf
from scipy import signal
from pydub import AudioSegment
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN DE FFMPEG
# =============================================================================
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg
AudioSegment.converter = os.path.join(ruta_bin_ffmpeg, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(ruta_bin_ffmpeg, "ffprobe.exe")

# =============================================================================
# RUTAS
# =============================================================================
carpeta_recortes = r"D:\Documentos\Ayudante de Investigacion\Codigos\Pruebas"
carpeta_final    = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_Limpios"

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
SUFIJO             = "_01"
FRECUENCIA_TRABAJO = 44100

# =============================================================================
# PARÁMETROS — FILTROS CLÁSICOS
# =============================================================================
HPF_FREQ_HZ       = 80
LPF_FREQ_HZ       = 10000
NOTCH_FREQ_HZ     = 50
NOTCH_Q           = 30
FILTRO_ORDEN      = 4
PRE_ENFASIS_ALPHA = 0.97
OBJETIVO_DBFS     = -23.0

# =============================================================================
# PARÁMETROS — SUSTRACCIÓN ESPECTRAL POR BANDAS
# =============================================================================
#
#  CORRECCIÓN VOCALIZACIÓN FEMENINA ENOJADA:
#  La voz femenina enojada concentra energía en 250–500 Hz (F0 elevado)
#  con armónicos fuertes hasta 3 kHz. El alfa anterior (2.5) en esa banda
#  era demasiado agresivo y "secaba" o metalizaba la voz.
#
#  Solución: reducir alfa de la banda media a 1.5 (conservador).
#  El supresor de música (etapa 7) se encarga del fondo musical en ese rango
#  usando un criterio distinto (estabilidad temporal), sin tocar la voz.
#
#  Cada tupla: (hz_inicio, hz_fin, alfa)

REDUCIR_RUIDO = True
SS_BANDAS = [
    (80,    300,  1.2),    # Baja  — protege F0 hombre (≥85 Hz) y mujer (≥165 Hz)
    (300,  3000,  1.5),    # Media — BAJADO de 2.5 → protege armónicos de enojo femenino
    (3000, 10000, 1.6),    # Alta  — sibilantes y fricativas emocionales
]
SS_BETA          = 0.05
SS_SUAVIZADO_BIN = 3

# =============================================================================
# PARÁMETROS — FILTRO DE WIENER GÉNERO-ADAPTIVO
# =============================================================================
#
#  El Wiener estándar usa la misma ganancia para todos los bins, lo que
#  aplana el espectro y produce el efecto "metálico" en la voz femenina.
#
#  Solución: proteger la banda vocal femenina (165–4000 Hz) aplicando
#  un floor de ganancia mínima WIENER_FLOOR_BANDA_VOZ > 0. Esto garantiza
#  que esa banda nunca quede completamente atenuada, aunque la SNR sea baja
#  (que ocurre en segmentos de enojo donde la energía se distribuye más).
#
#  WIENER_SNR_FLOOR: SNR mínima asumida antes de calcular la ganancia.
#  Subir (ej: 0.5) si la voz suena muy procesada; bajar (ej: 0.0) si
#  persiste ruido de fondo suave.

WIENER_ACTIVO             = True
WIENER_SUAVIZADO_TEMPORAL = 5
WIENER_SNR_FLOOR          = 0.3     # SNR mínima garantizada (evita ganancia=0 en voz)
WIENER_FLOOR_BANDA_VOZ    = 0.35    # Ganancia mínima en banda vocal (165–4000 Hz)
WIENER_BANDA_VOZ_HZ       = (165, 4000)  # Rango protegido para voz femenina

# =============================================================================
# PARÁMETROS — SUPRESOR DE MÚSICA
# =============================================================================
#
#  PROBLEMA: la música tiene estructura armónica similar a la voz, por lo
#  que el VAD por SFM no la distingue. La diferencia clave es la
#  ESTABILIDAD TEMPORAL: la música de fondo cambia poco entre frames
#  consecutivos (melodías, acordes sostenidos), mientras que la voz
#  emocional cambia rápido (especialmente Enojo, Sorprendido, Alegre).
#
#  Algoritmo (por band de frecuencia, frame a frame):
#    1. Calcula la correlación de magnitud entre frame actual y anterior.
#    2. Si la correlación supera MUSICA_ESTABILIDAD_UMBRAL durante
#       MUSICA_FRAMES_CONSECUTIVOS frames seguidos → se considera música.
#    3. Los bins identificados como música se atenúan con MUSICA_ATENUACION_DB.
#    4. Se protege la banda vocal (WIENER_BANDA_VOZ_HZ) con una atenuación
#       máxima menor para no dañar la voz aunque haya música en ese rango.
#
#  Valores de referencia:
#    MUSICA_ESTABILIDAD_UMBRAL = 0.80 → más agresivo (elimina más música)
#    MUSICA_ESTABILIDAD_UMBRAL = 0.92 → más conservador (menos riesgo de cortar voz)

MUSICA_ACTIVO               = True
MUSICA_ESTABILIDAD_UMBRAL   = 0.87   # Correlación mínima para considerar "música"
MUSICA_FRAMES_CONSECUTIVOS  = 4      # Frames seguidos estables para confirmar música
MUSICA_ATENUACION_DB        = -18.0  # dB de atenuación sobre bins de música
MUSICA_ATENUACION_BANDA_VOZ = -8.0   # dB de atenuación máxima en banda vocal protegida

# =============================================================================
# PARÁMETROS — VAD ESPECTRAL
# =============================================================================
#
#  Criterio 1 — SFM: voz tiene espectro con picos armónicos (SFM bajo).
#  Criterio 2 — Centroide: voz humana tiene centroide en rango conocido.
#  Frame activo si cumple AL MENOS UNO (AND sería demasiado estricto).

VAD_ACTIVO           = True
VAD_FRAME_MS         = 20
VAD_UMBRAL_SFM       = 0.50    # Subido de 0.45 — más permisivo para proteger enojo femenino
VAD_CENTROIDE_MIN    = 100
VAD_CENTROIDE_MAX    = 4500
VAD_GANANCIA_NOVOZ   = 0.02
VAD_CONTEXTO_MS      = 100     # Subido de 80 ms — más margen para ataques de enojo

# =============================================================================
# PARÁMETROS — NOISE GATE
# =============================================================================
GATE_UMBRAL_DB  = -45.0
GATE_FRAME_MS   = 20
GATE_RELEASE_MS = 60


# =============================================================================
# FUNCIONES
# =============================================================================

def disenar_filtros(fs):
    nyq = fs / 2.0
    b_hp,    a_hp    = signal.butter(FILTRO_ORDEN, HPF_FREQ_HZ  / nyq, btype='high')
    b_lp,    a_lp    = signal.butter(FILTRO_ORDEN, LPF_FREQ_HZ  / nyq, btype='low')
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ_HZ / nyq, NOTCH_Q)
    return (b_hp, a_hp), (b_lp, a_lp), (b_notch, a_notch)


def reduccion_ruido_por_bandas(audio, fs, duracion_ruido_ms=50):
    """
    Sustracción espectral con alfa diferenciado por banda.
    Banda media reducida a 1.5 para proteger armónicos de voz femenina enojada.
    """
    n_ruido = int(fs * duracion_ruido_ms / 1000)
    n_fft   = 1024
    hop     = n_fft // 2
    freqs   = np.fft.rfftfreq(n_fft, d=1.0/fs)

    seg_ruido    = audio[:n_ruido] if len(audio) > n_ruido else audio
    _, _, Zr     = signal.stft(seg_ruido, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    perfil_ruido = np.mean(np.abs(Zr), axis=1)

    k = SS_SUAVIZADO_BIN
    perfil_ruido = np.convolve(perfil_ruido, np.ones(k)/k, mode='same')

    alfa_por_bin = np.ones(len(freqs))
    for (f_ini, f_fin, alfa_banda) in SS_BANDAS:
        mascara = (freqs >= f_ini) & (freqs < f_fin)
        alfa_por_bin[mascara] = alfa_banda
    alfa_por_bin = alfa_por_bin[:, np.newaxis]
    perfil_2d    = perfil_ruido[:, np.newaxis]

    _, _, Z = signal.stft(audio, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag  = np.abs(Z)
    fase = np.angle(Z)

    mag_limpia = np.maximum(mag - alfa_por_bin * perfil_2d, SS_BETA * perfil_2d)

    _, audio_limpio = signal.istft(
        mag_limpia * np.exp(1j * fase), fs=fs, nperseg=n_fft, noverlap=n_fft - hop
    )
    if len(audio_limpio) > len(audio):
        audio_limpio = audio_limpio[:len(audio)]
    elif len(audio_limpio) < len(audio):
        audio_limpio = np.pad(audio_limpio, (0, len(audio) - len(audio_limpio)))

    return audio_limpio.astype(np.float32)


def aplicar_wiener_genero_adaptivo(audio, fs, duracion_ruido_ms=50):
    """
    Filtro de Wiener con protección de banda vocal femenina.

    Diferencia respecto al Wiener estándar:
      1. SNR mínima garantizada (WIENER_SNR_FLOOR): evita que la ganancia
         caiga a cero en bins de baja SNR que sí pueden contener voz
         (ocurre en segmentos de enojo con espectro muy distribuido).
      2. Floor de ganancia por banda vocal (WIENER_FLOOR_BANDA_VOZ):
         en la banda 165–4000 Hz, la ganancia nunca baja de ese valor,
         aunque la SNR sea baja. Previene el efecto "metálico/seco"
         en la voz femenina sin sacrificar la supresión fuera de esa banda.
    """
    n_ruido = int(fs * duracion_ruido_ms / 1000)
    n_fft   = 1024
    hop     = n_fft // 2
    freqs   = np.fft.rfftfreq(n_fft, d=1.0/fs)

    # Máscara de banda vocal protegida
    f_min_voz, f_max_voz = WIENER_BANDA_VOZ_HZ
    mascara_voz = (freqs >= f_min_voz) & (freqs <= f_max_voz)

    seg_ruido      = audio[:n_ruido] if len(audio) > n_ruido else audio
    _, _, Zr       = signal.stft(seg_ruido, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    pot_ruido      = np.mean(np.abs(Zr)**2, axis=1, keepdims=True) + 1e-12

    _, _, Z   = signal.stft(audio, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    pot_señal = np.abs(Z)**2

    # SNR local con floor mínimo garantizado
    snr = np.maximum(pot_señal / pot_ruido - 1.0, WIENER_SNR_FLOOR)

    # Ganancia de Wiener
    ganancia = snr / (snr + 1.0)

    # Aplicar floor de banda vocal: nunca atenuar más de (1 - WIENER_FLOOR_BANDA_VOZ)
    # en el rango 165–4000 Hz para proteger la voz femenina enojada
    ganancia[mascara_voz, :] = np.maximum(
        ganancia[mascara_voz, :], WIENER_FLOOR_BANDA_VOZ
    )

    # Suavizado temporal (evita fluctuaciones audibles entre frames)
    k = WIENER_SUAVIZADO_TEMPORAL
    if k > 1:
        kernel   = np.ones((1, k)) / k
        ganancia = signal.fftconvolve(ganancia, kernel, mode='same')
        ganancia = np.clip(ganancia, 0.0, 1.0)

    _, audio_limpio = signal.istft(
        ganancia * Z, fs=fs, nperseg=n_fft, noverlap=n_fft - hop
    )
    if len(audio_limpio) > len(audio):
        audio_limpio = audio_limpio[:len(audio)]
    elif len(audio_limpio) < len(audio):
        audio_limpio = np.pad(audio_limpio, (0, len(audio) - len(audio_limpio)))

    return audio_limpio.astype(np.float32)


def suprimir_musica(audio, fs):
    """
    Supresión de música de fondo por estabilidad temporal del espectro.

    Fundamento:
      La música de fondo (melodías, acordes sostenidos) tiene magnitud
      espectral muy estable entre frames consecutivos: un bin en 440 Hz
      (La) permanece fuerte durante muchos frames seguidos.
      La voz emocional —especialmente Enojo, Sorprendido, Alegre— cambia
      rápidamente: su espectro evoluciona frame a frame.

    Algoritmo:
      1. Para cada frame t y bin f, calcula correlación de Pearson entre
         el vector de magnitudes [t-N..t] y [t-N+1..t+1] (ventana deslizante).
      2. Si la correlación supera MUSICA_ESTABILIDAD_UMBRAL durante
         MUSICA_FRAMES_CONSECUTIVOS frames → ese bin se considera música.
      3. Se construye una máscara suave de atenuación (en dB) y se aplica.
      4. La banda vocal (165–4000 Hz) tiene una atenuación máxima menor
         (MUSICA_ATENUACION_BANDA_VOZ) para no cortar voz que coincida
         en frecuencia con la música.
    """
    n_fft = 1024
    hop   = n_fft // 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)

    f_min_voz, f_max_voz = WIENER_BANDA_VOZ_HZ
    mascara_voz = (freqs >= f_min_voz) & (freqs <= f_max_voz)

    _, _, Z = signal.stft(audio, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag  = np.abs(Z)          # shape: (n_bins, n_frames)
    fase = np.angle(Z)
    n_bins, n_frames = mag.shape

    N = MUSICA_FRAMES_CONSECUTIVOS  # ventana de correlación

    # Calcular correlación temporal entre frames consecutivos por bin
    # corr[f, t] = correlación entre mag[f, t-N:t] y mag[f, t-N+1:t+1]
    corr_matrix = np.zeros_like(mag)
    for t in range(N, n_frames):
        ventana_a = mag[:, t-N:t]      # (n_bins, N)
        ventana_b = mag[:, t-N+1:t+1]  # (n_bins, N)

        # Correlación de Pearson por bin (fila a fila)
        med_a = np.mean(ventana_a, axis=1, keepdims=True)
        med_b = np.mean(ventana_b, axis=1, keepdims=True)
        num   = np.sum((ventana_a - med_a) * (ventana_b - med_b), axis=1)
        den   = (np.sqrt(np.sum((ventana_a - med_a)**2, axis=1)) *
                 np.sqrt(np.sum((ventana_b - med_b)**2, axis=1)) + 1e-12)
        corr_matrix[:, t] = np.clip(num / den, 0.0, 1.0)

    # Detectar bins con correlación alta sostenida (= música)
    es_musica = corr_matrix >= MUSICA_ESTABILIDAD_UMBRAL  # bool (n_bins, n_frames)

    # Construcción de máscara de atenuación lineal
    atten_musica   = 10 ** (MUSICA_ATENUACION_DB       / 20.0)  # factor < 1
    atten_voz      = 10 ** (MUSICA_ATENUACION_BANDA_VOZ / 20.0) # factor < 1, menos agresivo

    mascara_ganancia = np.ones_like(mag)

    # Zona fuera de la banda vocal: atenuación completa si es música
    mascara_ganancia[~mascara_voz, :] = np.where(
        es_musica[~mascara_voz, :], atten_musica, 1.0
    )
    # Banda vocal: atenuación reducida para proteger voz femenina
    mascara_ganancia[mascara_voz, :] = np.where(
        es_musica[mascara_voz, :], atten_voz, 1.0
    )

    # Suavizar la máscara temporal para evitar artefactos de transición
    k_smooth = 3
    kernel   = np.ones((1, k_smooth)) / k_smooth
    mascara_ganancia = signal.fftconvolve(mascara_ganancia, kernel, mode='same')
    mascara_ganancia = np.clip(mascara_ganancia, atten_musica, 1.0)

    _, audio_limpio = signal.istft(
        mascara_ganancia * mag * np.exp(1j * fase), fs=fs,
        nperseg=n_fft, noverlap=n_fft - hop
    )
    if len(audio_limpio) > len(audio):
        audio_limpio = audio_limpio[:len(audio)]
    elif len(audio_limpio) < len(audio):
        audio_limpio = np.pad(audio_limpio, (0, len(audio) - len(audio_limpio)))

    return audio_limpio.astype(np.float32)


def aplicar_vad_espectral(audio, fs):
    """
    VAD de dos criterios: SFM + centroide espectral.
    Umbral SFM subido a 0.50 y contexto a 100 ms para proteger voz
    femenina enojada (espectro más distribuido que voces suaves).
    """
    n_frame   = int(fs * VAD_FRAME_MS    / 1000)
    n_ctx     = int(fs * VAD_CONTEXTO_MS / 1000)
    n_ctx_fr  = max(1, n_ctx // n_frame)
    n         = len(audio)
    n_frames  = max(1, int(np.ceil(n / n_frame)))
    n_fft_vad = 512
    es_voz    = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        ini   = i * n_frame
        fin   = min(ini + n_frame, n)
        frame = audio[ini:fin]
        if len(frame) < 16:
            continue
        nf    = min(n_fft_vad, len(frame))
        spec  = np.maximum(np.abs(np.fft.rfft(frame, n=nf)), 1e-12)
        freqs = np.fft.rfftfreq(nf, d=1.0/fs)

        # SFM
        sfm = np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12)

        # Centroide
        centroide = np.sum(freqs * spec) / (np.sum(spec) + 1e-12)

        es_voz[i] = (sfm < VAD_UMBRAL_SFM) or \
                    (VAD_CENTROIDE_MIN <= centroide <= VAD_CENTROIDE_MAX)

    # Dilatar contexto
    es_voz_exp = es_voz.copy()
    for i in range(n_frames):
        if es_voz[i]:
            es_voz_exp[max(0, i-n_ctx_fr):min(n_frames, i+n_ctx_fr+1)] = True

    curva = np.where(
        np.repeat(es_voz_exp, n_frame)[:n], 1.0, VAD_GANANCIA_NOVOZ
    )
    if len(curva) < n:
        curva = np.pad(curva, (0, n - len(curva)), constant_values=VAD_GANANCIA_NOVOZ)

    n_smooth = max(3, int(fs * 0.010 / n_frame) * n_frame)
    ventana  = np.hanning(n_smooth * 2 + 1)
    ventana /= ventana.sum()
    curva = np.clip(np.convolve(curva, ventana, mode='same'),
                    VAD_GANANCIA_NOVOZ, 1.0)

    return (audio * curva).astype(np.float32)


def aplicar_noise_gate(audio, fs):
    """Puerta de ruido con apertura/cierre suavizado (Hanning)."""
    n_frame   = int(fs * GATE_FRAME_MS   / 1000)
    n_release = int(fs * GATE_RELEASE_MS / 1000)
    n         = len(audio)
    n_frames  = max(1, int(np.ceil(n / n_frame)))
    ganancias = np.zeros(n_frames)

    for i in range(n_frames):
        ini   = i * n_frame
        fin   = min(ini + n_frame, n)
        frame = audio[ini:fin]
        rms   = np.sqrt(np.mean(frame**2) + 1e-12)
        ganancias[i] = 1.0 if (20 * np.log10(rms)) >= GATE_UMBRAL_DB else 0.0

    curva = np.repeat(ganancias, n_frame)[:n]
    if len(curva) < n:
        curva = np.pad(curva, (0, n - len(curva)), constant_values=curva[-1])

    if n_release > 1:
        ventana = np.hanning(n_release * 2 + 1)
        ventana /= ventana.sum()
        curva = np.clip(np.convolve(curva, ventana, mode='same'), 0.0, 1.0)

    return (audio * curva).astype(np.float32)


def aplicar_pre_enfasis(audio, alpha=PRE_ENFASIS_ALPHA):
    """H(z) = 1 - alpha·z⁻¹  — pipeline estándar MFCC/GFCC (Liu 2018)."""
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])


def normalizar_rms(audio, objetivo_dbfs=OBJETIVO_DBFS):
    """Normalización RMS robusta para emociones intensas."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-9:
        return audio
    ganancia = min(10 ** (objetivo_dbfs / 20.0) / rms, 10.0)
    return np.clip(audio * ganancia, -0.999, 0.999).astype(np.float32)


def procesar_audio(ruta_entrada, ruta_salida, filtros):
    """
    Pipeline completo de 11 etapas.
    """
    (b_hp, a_hp), (b_lp, a_lp), (b_notch, a_notch) = filtros

    # 1. Carga
    seg = (AudioSegment.from_wav(ruta_entrada)
           .set_frame_rate(FRECUENCIA_TRABAJO)
           .set_channels(1)
           .set_sample_width(2))
    muestras = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    # 2. Notch 50 Hz
    muestras = signal.filtfilt(b_notch, a_notch, muestras)

    # 3. HPF
    muestras = signal.filtfilt(b_hp, a_hp, muestras)

    # 4. LPF
    muestras = signal.filtfilt(b_lp, a_lp, muestras)

    # 5. Sustracción espectral por bandas (alfa conservador en banda media)
    if REDUCIR_RUIDO:
        muestras = reduccion_ruido_por_bandas(muestras, FRECUENCIA_TRABAJO)

    # 6. Wiener género-adaptivo (floor de ganancia en banda vocal femenina)
    if WIENER_ACTIVO:
        muestras = aplicar_wiener_genero_adaptivo(muestras, FRECUENCIA_TRABAJO)

    # 7. Supresor de música (estabilidad temporal del espectro)
    if MUSICA_ACTIVO:
        muestras = suprimir_musica(muestras, FRECUENCIA_TRABAJO)

    # 8. VAD espectral (SFM + centroide)
    if VAD_ACTIVO:
        muestras = aplicar_vad_espectral(muestras, FRECUENCIA_TRABAJO)

    # 9. Noise Gate
    muestras = aplicar_noise_gate(muestras, FRECUENCIA_TRABAJO)

    # 10. Pre-énfasis
    muestras = aplicar_pre_enfasis(muestras)

    # 11. Normalización RMS
    muestras = normalizar_rms(muestras)

    sf.write(ruta_salida, (muestras * 32767).astype(np.int16),
             FRECUENCIA_TRABAJO, subtype='PCM_16')


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def limpiar_voces_masivo():
    if not os.path.exists(carpeta_recortes):
        print("❌ Error: La carpeta de recortes no existe.")
        return

    print("=" * 68)
    print("  LIMPIEZA DE AUDIO PARA RECONOCIMIENTO DE EMOCIONES EN VOZ")
    print("=" * 68)
    print(f"  HPF ord.{FILTRO_ORDEN} @ {HPF_FREQ_HZ} Hz | LPF @ {LPF_FREQ_HZ} Hz | Notch @ {NOTCH_FREQ_HZ} Hz")
    print(f"  SS bandas     : {'SÍ' if REDUCIR_RUIDO else 'NO'}  β={SS_BETA}  alfa={[b[2] for b in SS_BANDAS]}")
    print(f"  Wiener adapt. : {'SÍ' if WIENER_ACTIVO else 'NO'}  floor_voz={WIENER_FLOOR_BANDA_VOZ}  SNR_floor={WIENER_SNR_FLOOR}")
    print(f"  Supresor mús. : {'SÍ' if MUSICA_ACTIVO else 'NO'}  umbral={MUSICA_ESTABILIDAD_UMBRAL}  atten={MUSICA_ATENUACION_DB} dB")
    print(f"  VAD espectral : {'SÍ' if VAD_ACTIVO else 'NO'}  SFM<{VAD_UMBRAL_SFM}  ctx={VAD_CONTEXTO_MS} ms")
    print(f"  Noise Gate    : umbral={GATE_UMBRAL_DB} dBFS  release={GATE_RELEASE_MS} ms")
    print(f"  Pre-énfasis α={PRE_ENFASIS_ALPHA} | RMS objetivo={OBJETIVO_DBFS} dBFS")
    print("=" * 68)

    filtros      = disenar_filtros(FRECUENCIA_TRABAJO)
    lista_tareas = []

    for root, dirs, files in os.walk(carpeta_recortes):
        for archivo in files:
            if archivo.lower().endswith(".wav"):
                lista_tareas.append((root, archivo))

    if not lista_tareas:
        print("⚠️  No se encontraron archivos .wav.")
        return

    errores = []

    for root, archivo in tqdm(lista_tareas, desc="Procesando", unit="audio"):
        try:
            nombre_base = os.path.splitext(archivo)[0]
            rel_path    = os.path.relpath(root, carpeta_recortes)

            if rel_path != ".":
                partes     = rel_path.split(os.sep)
                rel_salida = os.path.join(*[f"{p}{SUFIJO}" for p in partes if p])
            else:
                rel_salida = ""

            out_dir = os.path.join(carpeta_final, rel_salida)
            os.makedirs(out_dir, exist_ok=True)

            procesar_audio(
                os.path.join(root, archivo),
                os.path.join(out_dir, f"{nombre_base}{SUFIJO}.wav"),
                filtros
            )

        except Exception as e:
            errores.append((archivo, str(e)))

    print(f"\n✅ Procesamiento terminado.")
    print(f"   Procesados : {len(lista_tareas) - len(errores)}")
    print(f"   Errores    : {len(errores)}")
    print(f"   Salida     : {carpeta_final}")

    if errores:
        print("\n⚠️  Archivos con error:")
        for nombre, err in errores:
            print(f"   • {nombre}: {err}")


# =============================================================================
if __name__ == "__main__":
    limpiar_voces_masivo()