"""
ppg_clean_openbci.py
====================
Pipeline de limpieza de señal PPG grabada con OpenBCI Cyton,
sensor de pulso conectado al pin D11 (Analog Channel 0).

Uso:
    Ejecución directa desde el bloque __main__ con rutas fijas.

Requisitos:
    pip install numpy pandas scipy matplotlib
"""

import os
import sys
import argparse
import random  
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# 👈 MODIFICADO: Importamos directo para permitir ventanas emergentes (pop-ups)
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#  PARÁMETROS (ajustar si cambia el hardware)
# ──────────────────────────────────────────────
FS_NOMINAL    = 250.0   # Hz — frecuencia de muestreo declarada del Cyton
LOWCUT        = 0.5     # Hz — corte inferior del pasa-banda (≈ 30 lpm)
HIGHCUT       = 5.0     # Hz — corte superior del pasa-banda (≈ 300 lpm)
FILTER_ORDER  = 4       # orden del filtro Butterworth
WARMUP_S      = 2.0     # segundos a descartar al inicio por inestabilidad óptica
GAP_THRESH_S  = 0.5     # salto de tiempo mínimo para considerarse "gap real"
OUTLIER_STD   = 4       # umbral de z-score para detectar outliers
OUTLIER_WIN   = 250     # ventana móvil (muestras) para el z-score

# VARIABLE DEL SUFIJO
SUFIJO_SALIDA = "_00"  


# ──────────────────────────────────────────────
#  PASOS DEL PIPELINE
# ──────────────────────────────────────────────
def load_openbci_txt(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, skiprows=4, sep=',', engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df

def extract_ppg_raw(df: pd.DataFrame):
    t   = df['Timestamp'].to_numpy(dtype=float)
    ppg = df['Analog Channel 0'].to_numpy(dtype=float)
    return t, ppg

def trim_warmup(t, ppg, warmup_s: float = WARMUP_S):
    mask = (t - t[0]) >= warmup_s
    return t[mask], ppg[mask]

def detect_gaps(t, threshold_s: float = GAP_THRESH_S):
    dt = np.diff(t)
    idx = np.where(dt > threshold_s)[0]
    return [(t[i], t[i+1], dt[i]) for i in idx]

def resample_uniform(t, ppg, fs: float = FS_NOMINAL):
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    f_interp  = interp1d(t, ppg, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    return t_uniform, f_interp(t_uniform)

def remove_outliers(ppg, n_std: int = OUTLIER_STD, window: int = OUTLIER_WIN):
    ppg = ppg.copy()
    s   = pd.Series(ppg)
    mu  = s.rolling(window, center=True, min_periods=1).mean()
    sd  = s.rolling(window, center=True, min_periods=1).std()
    z   = (s - mu) / sd.replace(0, np.nan)
    mask = z.abs().gt(n_std).to_numpy()

    n_out = int(mask.sum())
    if n_out > 0:
        valid = ~mask
        ppg[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(valid),
            ppg[valid]
        )
    return ppg, n_out

def bandpass_filter(ppg, fs: float = FS_NOMINAL,
                    lowcut: float = LOWCUT, highcut: float = HIGHCUT,
                    order: int = FILTER_ORDER):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, ppg)


# ──────────────────────────────────────────────
#  PIPELINE COMPLETO
# ──────────────────────────────────────────────
def clean_ppg_pipeline(filepath: str, verbose: bool = True):
    df_raw = load_openbci_txt(filepath)
    t, ppg = extract_ppg_raw(df_raw)
    n_raw  = len(ppg)

    t, ppg = trim_warmup(t, ppg)
    gaps   = detect_gaps(t)

    t_uni, ppg_uni   = resample_uniform(t, ppg)
    ppg_no_out, n_out = remove_outliers(ppg_uni)
    ppg_clean         = bandpass_filter(ppg_no_out)

    duration_min = (t_uni[-1] - t_uni[0]) / 60.0
    fs_real      = len(t) / (t[-1] - t[0])

    report = {
        'archivo'              : os.path.basename(filepath),
        'muestras_originales'  : n_raw,
        'fs_efectiva_hz'       : round(fs_real, 2),
        'duracion_min'         : round(duration_min, 2),
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Archivo : {report['archivo']}")
        print(f"  Muestras originales       : {report['muestras_originales']}")
        print(f"  Fs efectiva               : {report['fs_efectiva_hz']} Hz")
        print(f"  Duración útil             : {report['duracion_min']} min")

    out = pd.DataFrame({
        'time_s'    : t_uni - t_uni[0],
        'ppg_raw'   : ppg_uni,
        'ppg_clean' : ppg_clean,
    })
    return out, report


# ──────────────────────────────────────────────
#  GRAFICO MODIFICADO
# ──────────────────────────────────────────────
def show_comparison_plot(df_out: pd.DataFrame, fs: float, filename: str):
    """
    Extrae un segmento continuo de 2 segundos aleatorios 
    y abre una ventana interactiva para visualizar la comparativa.
    """
    segundos = 20  
    samples_Ns = int(fs * segundos)  
    total_samples = len(df_out)
    
    if total_samples <= samples_Ns:
        print(f"  [Aviso] El archivo es demasiado corto para extraer {segundos} segundos de gráfico.")
        return

    start_idx = random.randint(0, total_samples - samples_Ns - 1)
    end_idx = start_idx + samples_Ns
    
    segment = df_out.iloc[start_idx:end_idx]
    time_x = segment['time_s'] - segment['time_s'].iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Subplot superior: Data Original
    ax1.plot(time_x, segment['ppg_raw'], color='#7f8c8d', alpha=0.8, linewidth=1.5, label='Original (Con Offset/Ruido)')
    ax1.set_title(f"Comparativa PPG - {filename}", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cuentas ADC")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Subplot inferior: Data Limpia
    ax2.plot(time_x, segment['ppg_clean'], color='#16a085', linewidth=2, label='Limpia (Filtrada 0.5 - 5.0 Hz)')
    ax2.set_xlabel("Tiempo (segundos)")
    ax2.set_ylabel("Amplitud Filtrada")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 👇 CAMBIADO: Muestra la figura interactiva en pantalla y pausa el script hasta que la cierres
    plt.show()


# ──────────────────────────────────────────────
#  PROCESAMIENTO POR LOTES
# ──────────────────────────────────────────────
def process_folder(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = []

    if os.path.isfile(input_path):
        files_to_process.append((input_path, os.path.basename(input_path)))
    else:
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if f.endswith('.txt'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, input_path)
                    files_to_process.append((full_path, rel_path))

    if not files_to_process:
        print(f"No se encontraron archivos .txt en: {input_path}")
        return

    print(f"\nProcesando {len(files_to_process)} archivo(s) manteniendo estructura...\n")

    for fp, rel_path in files_to_process:
        try:
            out, report = clean_ppg_pipeline(fp, verbose=True)
            
            rel_dir = os.path.dirname(rel_path)
            stem = os.path.splitext(os.path.basename(rel_path))[0]
            
            target_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(target_dir, exist_ok=True)
            
            # 1. Guardar el CSV final EXCLUSIVAMENTE con la data limpia
            out_csv = os.path.join(target_dir, f"{stem}{SUFIJO_SALIDA}.csv")
            out[['time_s', 'ppg_clean']].to_csv(out_csv, index=False)
            print(f"  [OK] Guardado CSV limpio en: {out_csv}")
            
            # 2. 👇 MODIFICADO: Llama a la función para mostrar la ventana gráfica
            print(f"  [Mostrando Gráfica] Cierra la ventana del gráfico para continuar...")
            show_comparison_plot(out, FS_NOMINAL, report['archivo'])
            
        except Exception as e:
            print(f"  ERROR procesando {fp}: {e}")

    print(f"\n{'═'*50}")
    print(f"  Proceso finalizado con éxito.")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Pones aquí la ruta de la carpeta que tiene tus archivos .txt
    ruta_entrada = r"D:\Documentos\Ayudante de Investigacion\OPENBCI\05_04_00.txt"
    
    # 2. Pones aquí la ruta de la carpeta donde quieres que se guarden
    ruta_salida = r"D:\Documentos\Ayudante de Investigacion\Fisio_limpio"
        
    # El script se ejecuta automáticamente con estas rutas al darle a "Play"
    process_folder(ruta_entrada, ruta_salida)