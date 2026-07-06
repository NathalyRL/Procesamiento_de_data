"""
ppg_clean_openbci.py
====================
Pipeline avanzado de limpieza de señal PPG grabada con OpenBCI Cyton.
Resuelve problemas de telemetría Bluetooth, clipping de latidos,
artefactos por movimiento masivo y genera máscaras de calidad científica.

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
from scipy.ndimage import label  
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#  PARÁMETROS (Calibrados para evitar falsos rechazos en sístoles)
# ──────────────────────────────────────────────
FS_NOMINAL    = 250.0   
LOWCUT        = 0.5     
HIGHCUT       = 5.0     
FILTER_ORDER  = 4       
WARMUP_S      = 2.0     
GAP_THRESH_S  = 0.5     
OUTLIER_STD   = 4       
OUTLIER_WIN   = 250     

# --- AJUSTE DE PARÁMETROS CRUCIALES PARA SEÑALES CON CLIPPING ---
# Aumentamos el tiempo mínimo de saturación. Un latido plano no dura más de 100ms (25 muestras).
# Si pasa de 35 muestras (140ms), entonces sí es un error de desconexión real.
MIN_SAT_DURATION   = 55   # ¡CORREGIDO! De 15 a 35 muestras para proteger los picos del pulso

# Umbrales físicos absolutos del OpenBCI Cyton para el canal analógico
LOW_THRESH         = 5    # Margen inferior
HIGH_THRESH        = 920  # Margen superior donde ocurre el clipping natural

# Desviación estándar móvil tolerante para que no rechace pendientes pronunciadas del pulso
ROLLING_STD_THRESH = 350  # ¡CORREGIDO! Mayor tolerancia al movimiento normal
MAX_BIOLOGICAL_DIFF = 300 # ¡CORREGIDO! Evita que los flancos de subida rápidos activen la máscara

# VARIABLE DEL SUFIJO Y DURACIÓN DEL GRÁFICO
SUFIJO_SALIDA = "_00"  
DURACION_GRAFICO_S = 150 


# ──────────────────────────────────────────────
#  PASOS DEL PIPELINE
# ──────────────────────────────────────────────

def load_openbci_txt(filepath: str) -> pd.DataFrame:
    """Carga el archivo ignorando metadatos dinámicamente usando el caracter '%'."""
    df = pd.read_csv(filepath, comment='%', sep=',', engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df

def extract_ppg_raw(df: pd.DataFrame):
    """
    Usa el 'Sample Index' de hardware para reconstruir el tiempo real,
    detectando pérdidas de Bluetooth y rellenándolas localmente.
    """
    df = df.reset_index(drop=True)
    sample_indices = df['Sample Index'].to_numpy(dtype=int)
    ppg_raw        = df['Analog Channel 0'].to_numpy(dtype=float)
    
    diffs = np.diff(sample_indices) % 256
    
    cumulative_indices = np.zeros(len(sample_indices), dtype=int)
    cumulative_indices[1:] = np.cumsum(diffs)
    
    total_expected_samples = cumulative_indices[-1] + 1
    full_range = np.arange(0, total_expected_samples)
    
    ppg_fixed = np.full(total_expected_samples, np.nan)
    ppg_fixed[cumulative_indices] = ppg_raw
    
    dropped_packets = total_expected_samples - len(df)
    if dropped_packets > 0:
        pct = (dropped_packets / total_expected_samples) * 100
        print(f"  [Info Telemetría] {dropped_packets} paquetes caídos interpolados ({pct:.2f}%).")
        
    s = pd.Series(ppg_fixed)
    ppg_interpolated = s.interpolate(method='linear').to_numpy()
    
    t_corrected = full_range / FS_NOMINAL
    return t_corrected, ppg_interpolated

def trim_warmup(t, ppg, warmup_s: float = WARMUP_S):
    """Elimina los primeros segundos de estabilización del hardware."""
    mask = (t - t[0]) >= warmup_s
    return t[mask], ppg[mask]

def detect_and_mask_artifacts(ppg, low_thresh=10, high_thresh=925):
    """
    Detecta saturaciones largas, caos por movimiento brusco y saltos electromagnéticos.
    Retorna la señal corregida para el filtro y la Máscara de Calidad.
    """
    ppg_fixed = ppg.copy()
    valid_mask = np.ones(len(ppg), dtype=bool)
    
    # CRITERIO 1: Bloques planos continuos en extremos físicos (Saturaciones)
    hard_mask = (ppg_fixed <= low_thresh) | (ppg_fixed >= high_thresh)
    labeled_mask, num_features = label(hard_mask)
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i)
        if np.sum(component_mask) >= MIN_SAT_DURATION:
            valid_mask[component_mask] = False
            
    # CRITERIO 2: Desviación estándar móvil (Movimientos caóticos)
    s = pd.Series(ppg_fixed)
    r_std = s.rolling(int(FS_NOMINAL), center=True, min_periods=1).std().to_numpy()
    motion_mask = r_std > ROLLING_STD_THRESH
    valid_mask[motion_mask] = False
    
    # CRITERIO 3: Diferencias consecutivas biológicamente imposibles (Spikes)
    signal_diffs = np.zeros(len(ppg_fixed))
    signal_diffs[1:] = np.abs(np.diff(ppg_fixed))
    spike_mask = signal_diffs > MAX_BIOLOGICAL_DIFF
    valid_mask[spike_mask] = False
    
    # BLINDAJE DE SEGURIDAD: Evita colapsar la señal a NaN si todo se marca erróneamente
    invalid_idx = np.flatnonzero(~valid_mask)
    valid_idx = np.flatnonzero(valid_mask)
    
    if len(invalid_idx) > 0 and len(valid_idx) > 0:
        ppg_fixed[~valid_mask] = np.interp(invalid_idx, valid_idx, ppg[valid_idx])
        
    return ppg_fixed, valid_mask

def resample_uniform(t, ppg, fs: float = FS_NOMINAL):
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    f_interp  = interp1d(t, ppg, kind='linear', bounds_error=False, fill_value='extrapolate')
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
        ppg[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(valid), ppg[valid])
    return ppg, n_out

def bandpass_filter(ppg, fs: float = FS_NOMINAL, lowcut: float = LOWCUT, highcut: float = HIGHCUT, order: int = FILTER_ORDER):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, ppg)


# ──────────────────────────────────────────────
#  PIPELINE CENTRAL
# ──────────────────────────────────────────────
def clean_ppg_pipeline(filepath: str, verbose: bool = True):
    df_raw = load_openbci_txt(filepath)
    t, ppg = extract_ppg_raw(df_raw)
    n_raw  = len(df_raw)

    t, ppg = trim_warmup(t, ppg)
    ppg_smoothed, valid_mask = detect_and_mask_artifacts(ppg)

    t_uni, ppg_uni = resample_uniform(t, ppg_smoothed)
    f_mask = interp1d(t, valid_mask.astype(float), kind='nearest', bounds_error=False, fill_value=0)
    valid_mask_uni = f_mask(t_uni).astype(bool)
    
    ppg_no_out, n_out = remove_outliers(ppg_uni)
    ppg_clean         = bandpass_filter(ppg_no_out)

    duration_min = (t_uni[-1] - t_uni[0]) / 60.0
    fs_real = len(df_raw) / (df_raw['Timestamp'].iloc[-1] - df_raw['Timestamp'].iloc[0]) if 'Timestamp' in df_raw.columns else FS_NOMINAL
    pct_valido = (valid_mask_uni.sum() / len(valid_mask_uni)) * 100 if len(valid_mask_uni) > 0 else 0.0

    report = {
        'archivo'              : os.path.basename(filepath),
        'muestras_originales'  : n_raw,
        'fs_efectiva_hz'       : round(fs_real, 2),
        'duracion_min'         : round(duration_min, 2),
        'porcentaje_valido'    : round(pct_valido, 2)
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Archivo : {report['archivo']}")
        print(f"  Muestras leídas           : {report['muestras_originales']}")
        print(f"  Fs estimada PC (Aprox)    : {report['fs_efectiva_hz']} Hz")
        print(f"  Duración útil             : {report['duracion_min']} min")
        print(f"  Calidad (Data Válida)     : {report['porcentaje_valido']}%")

    out = pd.DataFrame({
        'time_s'      : t_uni - t_uni[0],
        'ppg_raw'     : ppg_uni,
        'ppg_clean'   : ppg_clean,
        'valid_sample': valid_mask_uni  
    })
    return out, report


# ──────────────────────────────────────────────
#  VISUALIZACIÓN DE CONTROL (CORREGIDA)
# ──────────────────────────────────────────────
def show_comparison_plot(df_out: pd.DataFrame, fs: float, filename: str):
    samples_total = int(fs * DURACION_GRAFICO_S)  
    total_samples = len(df_out)
    
    if total_samples <= samples_total:
        print("  [Aviso] El archivo es demasiado corto para extraer el gráfico aleatorio.")
        return

    start_idx = random.randint(0, total_samples - samples_total - 1)
    end_idx = start_idx + samples_total
    
    segment = df_out.iloc[start_idx:end_idx]
    time_x = segment['time_s'] - segment['time_s'].iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(time_x, segment['ppg_raw'], color='#7f8c8d', alpha=0.8, linewidth=1.5, label='Original Reconstruida')
    ax1.set_title(f"Comparativa PPG ({DURACION_GRAFICO_S}s aleatorios) - {filename}", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cuentas ADC")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    ax2.plot(time_x, segment['ppg_clean'], color='#16a085', linewidth=2, label='Limpia (Filtrada 0.5 - 5.0 Hz)')
    
    # Uso correcto del operador bitwise ~ de Pandas/Numpy
    invalid_segments = ~segment['valid_sample']
    if invalid_segments.any():
        ax2.fill_between(time_x, segment['ppg_clean'].min(), segment['ppg_clean'].max(), 
                         where=invalid_segments, color='#e74c3c', alpha=0.3, label='ZONA RECHAZADA (Artefacto)')
        
    ax2.set_xlabel("Tiempo (segundos)")
    ax2.set_ylabel("Amplitud Filtrada")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
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
            
            out_csv = os.path.join(target_dir, f"{stem}{SUFIJO_SALIDA}.csv")
            
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write(f"Muestras originales       : {report['muestras_originales']}\n")
                f.write(f"Fs efectiva               : {report['fs_efectiva_hz']} Hz\n")
                f.write(f"Duracion util             : {report['duracion_min']} min\n")
                f.write(f"Porcentaje data valida    : {report['porcentaje_valido']} %\n")
                f.write("──────────────────────────────────────────────────\n") 
            
            out[['time_s', 'ppg_clean', 'valid_sample']].to_csv(out_csv, mode='a', index=False)
            print(f"  [OK] Guardado CSV limpio con metadatos en: {out_csv}")
            
            print(f"  [Mostrando Gráfica] Cierra la ventana del gráfico para continuar...")
            show_comparison_plot(out, FS_NOMINAL, report['archivo'])
            
        except Exception as e:
            print(f"  ERROR procesando {fp}: {e}")

    print(f"\n{'═'*50}")
    print(f"  Proceso finalizado con éxito.")


if __name__ == '__main__':
    ruta_entrada = r"D:\Documentos\Ayudante de Investigacion\OPENBCI\21_08_00.txt"
    ruta_salida = r"D:\Documentos\Ayudante de Investigacion\Fisio_limpio"      
    process_folder(ruta_entrada, ruta_salida)