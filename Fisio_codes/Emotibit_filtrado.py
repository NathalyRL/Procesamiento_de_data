import pandas as pd
import numpy as np
import os
import glob
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import butter, filtfilt

# =============================================================================
# 0. CONFIGURACIÓN DE RUTAS
# =============================================================================

carpeta_datos = r"d:\Documentos\Ayudante de Investigacion\Emotibit_test"
carpeta_salida = r"d:\Documentos\Ayudante de Investigacion\Emotibit_Limpios2"

# Sufijos para cada archivo de salida. Se generan DOS csv por archivo de
# entrada (uno de EDA a su fs nativa ~15 Hz, otro de PPG a su fs nativa
# ~25 Hz), en vez de forzarlos a compartir una sola línea de tiempo.
sufijo_eda = "_01"
sufijo_ppg = "_02"

# Si es True, abre una ventana con la comparación "crudo vs. limpio" de
# cada señal, para verificar visualmente el filtrado antes de confiar en
# el CSV generado. La ventana debe cerrarse para que el script continúe.
GENERAR_GRAFICAS = True # Cambia a True si quieres ver las gráficas de verificación

# Tags fisiológicos que nos interesan
TAGS_INTERES = ["EA", "EL", "PG", "PR", "PI"]

# =============================================================================
# 1. PARSEO ROBUSTO DEL CSV CRUDO (fila por fila, DataLength variable)
# =============================================================================
# IMPORTANTE: el DataLength (columna 3) NO es constante por sensor.
# Un mismo TypeTag (ej. AX, EA) puede traer paquetes con 1, 2, 3, 4 o más
# valores según el momento de transmisión. Un pd.read_csv con columnas
# fijas (Value1, Value2, Value3) descarta silenciosamente cualquier fila
# que traiga más valores de los esperados. Por eso parseamos manualmente,
# leyendo exactamente DataLength valores por fila.
def parse_emotibit_csv(filepath, tags_interes):
    """
    Devuelve:
        data[tag] = {"timestamps": [...], "values": [...]}  (values = lista de listas, un paquete por entrada)
        t0_global = timestamp mínimo de TODO el archivo (para alinear señales)
    """
    data = defaultdict(lambda: {"timestamps": [], "values": []})
    t0_global = None

    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                timestamp = float(row[0])
                data_length = int(row[2])
                type_tag = row[3].strip()
            except (ValueError, IndexError):
                continue

            if t0_global is None or timestamp < t0_global:
                t0_global = timestamp

            if type_tag not in tags_interes:
                continue

            values_raw = row[6:6 + data_length]
            values = []
            ok = True
            for v in values_raw:
                v = v.strip()
                try:
                    values.append(float(v))
                except ValueError:
                    ok = False
                    break
            if not ok or len(values) == 0:
                continue

            data[type_tag]["timestamps"].append(timestamp)
            data[type_tag]["values"].append(values)  # lista de valores del paquete

    return data, t0_global


def expandir_muestras(entradas_paquete, fs_hint=None):
    """
    Convierte [(timestamp_paquete, [v1, v2, ...]), ...] en dos arrays planos
    (timestamps, values), interpolando el timestamp de cada muestra dentro
    del paquete si se conoce fs. El EmotiBit Timestamp corresponde a la
    última muestra del paquete.
    """
    all_ts, all_vals = [], []
    for ts, vals in entradas_paquete:
        n = len(vals)
        if fs_hint and n > 1:
            dt_ms = 1000.0 / fs_hint
            sample_ts = [ts - (n - 1 - i) * dt_ms for i in range(n)]
        else:
            sample_ts = [ts] * n
        all_ts.extend(sample_ts)
        all_vals.extend(vals)
    return np.array(all_ts), np.array(all_vals)


def calcular_fs_real(timestamps, n_muestras_total):
    if len(timestamps) < 2:
        return None
    duracion_s = (timestamps[-1] - timestamps[0]) / 1000.0
    if duracion_s <= 0:
        return None
    return n_muestras_total / duracion_s


# =============================================================================
# 2. FUNCIONES DE FILTRADO DIGITAL
# =============================================================================

def filtro_pasa_bajas(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)


def filtro_pasa_banda(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band', analog=False)
    return filtfilt(b, a, data)


# =============================================================================
# 3. PROCESAMIENTO POR SEÑAL
# =============================================================================

def procesar_tag(data, tag, t0_global, tipo_filtro, **filtro_kwargs):
    """
    tipo_filtro: 'pasa_bajas' o 'pasa_banda'
    Retorna (DataFrame con Time_Sec/Value/Clean_Value, fs_estimada) o (vacío, None).
    """
    if tag not in data or len(data[tag]["timestamps"]) == 0:
        return pd.DataFrame(), None

    paquetes = list(zip(data[tag]["timestamps"], data[tag]["values"]))
    paquetes.sort(key=lambda p: p[0])

    n_muestras_total = sum(len(v) for _, v in paquetes)
    ts_paquetes = [p[0] for p in paquetes]
    fs_estimada = calcular_fs_real(ts_paquetes, n_muestras_total)
    if fs_estimada is None or fs_estimada <= 0:
        return pd.DataFrame(), None

    ts, vals = expandir_muestras(paquetes, fs_hint=fs_estimada)

    orden = np.argsort(ts)
    ts, vals = ts[orden], vals[orden]

    if len(vals) < 10:  # muy pocos datos para filtrar de forma confiable
        return pd.DataFrame(), fs_estimada

    if tipo_filtro == "pasa_bajas":
        clean = filtro_pasa_bajas(vals, fs=fs_estimada, **filtro_kwargs)
    else:
        clean = filtro_pasa_banda(vals, fs=fs_estimada, **filtro_kwargs)

    df_res = pd.DataFrame({
        "Time_Sec": (ts - t0_global) / 1000.0,
        "Value": vals,
        "Clean_Value": clean,
    })
    return df_res, fs_estimada


# =============================================================================
# 4. VISUALIZACIÓN: crudo vs. limpio, para revisar antes de confiar en el CSV
# =============================================================================

def graficar_crudo_vs_limpio(señales, nombre_archivo):
    """
    señales: lista de (nombre_tag, dataframe) con columnas Time_Sec/Value/Clean_Value.
    Abre una figura en pantalla con un subplot por señal (no la guarda en disco).
    Crudo y limpio se grafican con EJES Y INDEPENDIENTES (twinx): comparten el
    eje de tiempo, pero cada uno tiene su propia escala. Esto es necesario porque
    el crudo (ej. PPG ~0-160000) y el limpio filtrado (ej. PPG ~-50 a 50) tienen
    órdenes de magnitud muy distintos; si comparten el mismo eje Y, el limpio se
    ve aplastado como una línea plana.

    NOTA: plt.show() bloquea la ejecución hasta que cierres la ventana de la
    figura. Si procesas varios archivos en el bucle, tendrás que cerrar cada
    figura para que continúe con el siguiente archivo.
    """
    señales_validas = [(nombre, df) for nombre, df in señales if not df.empty]
    if not señales_validas:
        print("    ⚠️  No hay señales para graficar.")
        return

    n = len(señales_validas)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax_crudo, (nombre, df) in zip(axes, señales_validas):
        ax_limpio = ax_crudo.twinx()

        l1, = ax_crudo.plot(df["Time_Sec"], df["Value"], color="gray", alpha=0.5,
                             linewidth=0.7, label="Crudo")
        l2, = ax_limpio.plot(df["Time_Sec"], df["Clean_Value"], color="tab:blue",
                              linewidth=0.8, label="Limpio (filtrado)")

        ax_crudo.set_title(nombre)
        ax_crudo.set_xlabel("Tiempo (s)")
        ax_crudo.set_ylabel("Amplitud cruda", color="gray")
        ax_limpio.set_ylabel("Amplitud limpia", color="tab:blue")
        ax_crudo.tick_params(axis='y', labelcolor="gray")
        ax_limpio.tick_params(axis='y', labelcolor="tab:blue")
        ax_crudo.grid(True, alpha=0.3)
        ax_crudo.legend(handles=[l1, l2], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =============================================================================
# 5. BUCLE DE PROCESAMIENTO PRINCIPAL
# =============================================================================

if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)
    print(f"📁 La carpeta de salida no existía, se ha creado automáticamente: {carpeta_salida}")

patron_busqueda = os.path.join(carpeta_datos, "*.csv")
archivos_a_procesar = glob.glob(patron_busqueda)

if not archivos_a_procesar:
    print(f"❌ No se encontraron archivos .csv para procesar en la carpeta de origen: {carpeta_datos}")

for ruta_completa_entrada in archivos_a_procesar:
    nombre_archivo = os.path.basename(ruta_completa_entrada)
    print(f"\n--- Leyendo desde: {ruta_completa_entrada} ---")

    data, t0_global = parse_emotibit_csv(ruta_completa_entrada, TAGS_INTERES)
    if t0_global is None:
        print(f"❌ Archivo vacío o ilegible: '{nombre_archivo}'. Saltando...")
        continue

    # EDA: pasa-bajas. Nota: ya NO se descartan valores negativos;
    # EL (nivel tónico) es legítimamente negativo en escala cruda.
    eda_ea, fs_ea = procesar_tag(data, "EA", t0_global, "pasa_bajas", cutoff=1.0)
    eda_el, fs_el = procesar_tag(data, "EL", t0_global, "pasa_bajas", cutoff=0.05)

    # PPG: pasa-banda 0.7-3.5 Hz (banda de frecuencia cardiaca)
    ppg_pg, fs_pg = procesar_tag(data, "PG", t0_global, "pasa_banda", lowcut=0.7, highcut=3.5)
    ppg_pi, fs_pi = procesar_tag(data, "PI", t0_global, "pasa_banda", lowcut=0.7, highcut=3.5)
    ppg_pr, fs_pr = procesar_tag(data, "PR", t0_global, "pasa_banda", lowcut=0.7, highcut=3.5)

    for nombre_señal, fs_val in [("EA", fs_ea), ("EL", fs_el), ("PG", fs_pg),
                                  ("PI", fs_pi), ("PR", fs_pr)]:
        if fs_val:
            print(f"    fs calculada para {nombre_señal}: {fs_val:.2f} Hz")

    if ppg_pg.empty:
        print(f"❌ Error: Sin datos de PPG Verde en '{nombre_archivo}'. Saltando archivo...")
        continue

    if GENERAR_GRAFICAS:
        graficar_crudo_vs_limpio(
            [("EDA Fásica (EA)", eda_ea), ("EDA Tónica (EL)", eda_el),
             ("PPG Verde (PG)", ppg_pg), ("PPG Infrarrojo (PI)", ppg_pi), ("PPG Rojo (PR)", ppg_pr)],
            nombre_archivo
        )

    nombre_base, extension = os.path.splitext(nombre_archivo)

    def guardar_csv(df_base, columnas_extra, fs_dict, ruta_salida):
        """
        df_base: DataFrame ancla (con Time_Sec ya renombrado a su primera columna de señal).
        columnas_extra: lista de (dataframe, nombre_columna) a fusionar sobre df_base
                         vía merge_asof (todas dentro del mismo grupo EDA o PPG,
                         por lo que su fs nativa es casi idéntica y el desfase es mínimo).
        fs_dict: {nombre_columna: fs} para escribir como comentario en el CSV.
        """
        df_out = df_base
        for df_signal, nombre_col in columnas_extra:
            if not df_signal.empty:
                df_out = pd.merge_asof(
                    df_out,
                    df_signal[['Time_Sec', 'Clean_Value']].rename(columns={'Clean_Value': nombre_col}),
                    on='Time_Sec', direction='nearest'
                )
            else:
                print(f"    ⚠️  Sin datos suficientes para {nombre_col}, se omite del CSV de salida.")

        with open(ruta_salida, "w", newline="") as f:
            for columna, fs_val in fs_dict.items():
                if columna in df_out.columns and fs_val is not None:
                    f.write(f"# fs_{columna}_Hz={fs_val:.4f}\n")
            df_out.to_csv(f, index=False)

    # --- CSV de EDA, a su fs nativa (~15 Hz) ---
    if not eda_ea.empty:
        eda_df_base = eda_ea[['Time_Sec', 'Clean_Value']].rename(columns={'Clean_Value': 'EDA_Phasic'})
        ruta_eda = os.path.join(carpeta_salida, f"{nombre_base}{sufijo_eda}{extension}")
        guardar_csv(eda_df_base, [(eda_el, 'EDA_Tonic')],
                    {"EDA_Phasic": fs_ea, "EDA_Tonic": fs_el}, ruta_eda)
        print(f"✔️ EDA guardado en: '{ruta_eda}'!")
    else:
        print(f"❌ Sin datos de EDA Fásica en '{nombre_archivo}'. No se genera CSV de EDA.")

    # --- CSV de PPG, a su fs nativa (~25 Hz) ---
    ppg_df_base = ppg_pg[['Time_Sec', 'Clean_Value']].rename(columns={'Clean_Value': 'PPG_Green'})
    ruta_ppg = os.path.join(carpeta_salida, f"{nombre_base}{sufijo_ppg}{extension}")
    guardar_csv(ppg_df_base, [(ppg_pi, 'PPG_Infrared'), (ppg_pr, 'PPG_Red')],
                {"PPG_Green": fs_pg, "PPG_Infrared": fs_pi, "PPG_Red": fs_pr}, ruta_ppg)
    print(f"✔️ PPG guardado en: '{ruta_ppg}'!")

print("\n🎉 ¡Lote automático terminado! Todos los archivos procesados están en tu carpeta de destino.")