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

import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#  PARÁMETROS (ajustar si cambia el hardware)
# ──────────────────────────────────────────────
FS_NOMINAL    = 250.0   
LOWCUT        = 0.5     
HIGHCUT       = 5.0     
FILTER_ORDER  = 4       
WARMUP_S      = 2.0     
GAP_THRESH_S  = 0.5     
OUTLIER_STD   = 4       
OUTLIER_WIN   = 250     

# VARIABLE DEL SUFIJO Y DURACIÓN DEL GRÁFICO
SUFIJO_SALIDA = "_00"  
DURACION_GRAFICO_S = 150 # Cambia este valor para ver más o menos segundos en la gráfica


# ──────────────────────────────────────────────
#  PASOS DEL PIPELINE
# ──────────────────────────────────────────────
def load_openbci_txt(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, skiprows=4, sep=',', engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df

def extract_ppg_raw(df: pd.DataFrame):
    # Asegurar limpieza de nombres de columnas
    df.columns = df.columns.str.strip()
    
    # 1. Extraer los arrays nativos de OpenBCI
    sample_indices = df['Sample Index'].to_numpy(dtype=int)
    ppg_raw        = df['Analog Channel 0'].to_numpy(dtype=float)
    
    # 2. Calcular los saltos entre muestras usando aritmética modular (% 256)
    # Esto corrige automáticamente el reinicio cuando el contador pasa de 255 a 0
    diffs = np.diff(sample_indices) % 256
    
    # 3. Reconstruir un índice de muestras continuo acumulativo (sin topes)
    cumulative_indices = [0]
    for d in diffs:
        # d debería ser 1 si no hay pérdidas. Si d > 1, acumula el salto real.
        cumulative_indices.append(cumulative_indices[-1] + d)
    cumulative_indices = np.array(cumulative_indices)
    
    # 4. Crear la estructura de la línea de tiempo total ideal
    total_muestras_esperadas = cumulative_indices[-1] + 1
    full_range = np.arange(0, total_muestras_esperadas)
    
    # Creamos un contenedor del tamaño perfecto lleno de NaNs (huecos vacíos)
    ppg_fixed = np.full(total_muestras_esperadas, np.nan)
    
    # Ubicamos cada dato que sí llegó en su "asiento" correspondiente
    ppg_fixed[cumulative_indices] = ppg_raw
    
    # 5. CONTABILIZAR PÉRDIDAS (Opcional, para tu reporte)
    paquetes_perdidos = total_muestras_esperadas - len(df)
    if paquetes_perdidos > 0:
        porcentaje = (paquetes_perdidos / total_muestras_esperadas) * 100
        print(f"  [Info Bluetooth] Se detectaron {paquetes_perdidos} paquetes perdidos ({porcentaje:.2f}%). Reconstruyendo...")

    # 6. RESOLVER LOS HUECOS: Interpolación lineal local de los NaNs
    # Convierte los NaNs en una transición suave para que no rompa el filtro pasabanda
    s = pd.Series(ppg_fixed)
    ppg_interpolada = s.interpolate(method='linear').to_numpy()
    
    # 7. Generar el tiempo ideal real basado en la frecuencia nominal (250 Hz)
    t_corregido = full_range / FS_NOMINAL
    
    return t_corregido, ppg_interpolada

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
#  FUNCION PARA MOSTRAR GRÁFICO
# ──────────────────────────────────────────────
def show_comparison_plot(df_out: pd.DataFrame, fs: float, filename: str):
    samples_total = int(fs * DURACION_GRAFICO_S)  
    total_samples = len(df_out)
    
    if total_samples <= samples_total:
        print("  [Aviso] El archivo es demasiado corto para extraer el gráfico.")
        return

    start_idx = random.randint(0, total_samples - samples_total - 1)
    end_idx = start_idx + samples_total
    
    segment = df_out.iloc[start_idx:end_idx]
    time_x = segment['time_s'] - segment['time_s'].iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(time_x, segment['ppg_raw'], color='#7f8c8d', alpha=0.8, linewidth=1.5, label='Original (Con Offset/Ruido)')
    ax1.set_title(f"Comparativa PPG ({DURACION_GRAFICO_S} segundos aleatorios) - {filename}", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cuentas ADC")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    ax2.plot(time_x, segment['ppg_clean'], color='#16a085', linewidth=2, label='Limpia (Filtrada 0.5 - 5.0 Hz)')
    ax2.set_xlabel("Tiempo (segundos)")
    ax2.set_ylabel("Amplitud Filtrada")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
#  PROCESAMIENTO POR LOTES (MODIFICADO)
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
            
            # 👇 MODIFICACIÓN CRUCIAL: Escribir el encabezado de texto con la información antes de meter la tabla
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write(f"Muestras originales       : {report['muestras_originales']}\n")
                f.write(f"Fs efectiva               : {report['fs_efectiva_hz']} Hz\n")
                f.write(f"Duracion util             : {report['duracion_min']} min\n")
                f.write("──────────────────────────────────────────────────\n") # Línea divisoria
            
            # 👇 Guardar la data limpia anexándola abajo (mode='a' de "append")
            out[['time_s', 'ppg_clean']].to_csv(out_csv, mode='a', index=False)
            print(f"  [OK] Guardado CSV limpio con metadatos en: {out_csv}")
            
            # Mostrar la ventana gráfica
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
    ruta_entrada = r"D:\Documentos\Ayudante de Investigacion\OPENBCI\23_08_00.txt"
    
    # 2. Pones aquí la ruta de la carpeta donde quieres que se guarden
    ruta_salida = r"D:\Documentos\Ayudante de Investigacion\Fisio_limpio"      
    process_folder(ruta_entrada, ruta_salida)