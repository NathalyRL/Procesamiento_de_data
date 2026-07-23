import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo CSV de EmotiBit
archivo_csv = r"D:\Documentos\Ayudante de Investigacion\EMOTIBIT\07_07_01.csv"  # Cambia esto por el nombre real de tu archivo CSV
col_names = ['EmotiBitTimestamp', 'PacketNumber', 'DataLength', 'TypeTag', 'Version', 'Reliability', 'Value1', 'Value2', 'Value3']
df = pd.read_csv(archivo_csv, names=col_names, header=None, on_bad_lines='skip')

# --- FUNCIÓN REUTILIZABLE PARA PROCESAR PPG (Aplanar matrices de 3 valores) ---
def procesar_ppg(df_filtrado):
    if df_filtrado.empty:
        return pd.DataFrame(columns=['Time_Sec', 'Value'])
    
    registros = []
    # El muestreo de EmotiBit envía 3 datos secuenciales por cada timestamp registrado
    # Estimamos el intervalo promedio entre sub-muestras (aprox. 10ms para ~100Hz o 40ms para ~25Hz)
    dt = 0.010  
    
    for _, row in df_filtrado.iterrows():
        t_base = row['EmotiBitTimestamp'] / 1000.0
        for i, val_col in enumerate(['Value1', 'Value2', 'Value3']):
            val = pd.to_numeric(row[val_col], errors='coerce')
            if not pd.isna(val):
                # Reconstruir el tiempo lineal de cada sub-muestra
                registros.append({'Time_Sec': t_base + (i * dt), 'Value': val})
                
    df_res = pd.DataFrame(registros)
    if not df_res.empty:
        df_res['Time_Sec'] = df_res['Time_Sec'] - df_res['Time_Sec'].iloc[0]
    return df_res

# --- FUNCIÓN REUTILIZABLE PARA PROCESAR EDA (Un solo valor por fila) ---
def procesar_eda(df_filtrado):
    if df_filtrado.empty:
        return pd.DataFrame(columns=['Time_Sec', 'Value'])
    df_res = df_filtrado.copy()
    df_res['Value'] = pd.to_numeric(df_res['Value1'], errors='coerce')
    df_res['Time_Sec'] = (df_res['EmotiBitTimestamp'] - df_res['EmotiBitTimestamp'].iloc[0]) / 1000.0
    return df_res[['Time_Sec', 'Value']].dropna()

# 2. Extracción y procesamiento de cada canal específico
eda_ea = procesar_eda(df[df['TypeTag'] == 'EA']) # Electrodermal Activity (Fásico)
eda_el = procesar_eda(df[df['TypeTag'] == 'EL']) # Electrodermal Level (Tónico)

ppg_pi = procesar_ppg(df[df['TypeTag'] == 'PI']) # Infrarrojo
ppg_pr = procesar_ppg(df[df['TypeTag'] == 'PR']) # Rojo
ppg_pg = procesar_ppg(df[df['TypeTag'] == 'PG']) # Verde

# 3. Configuración y despliegue de las 5 gráficas
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=False)

# Configuración de mapeo para iterar el dibujado
canales = [
    {'df': eda_ea, 'label': 'EDA - EA (Phasic)', 'color': 'teal', 'ylabel': 'uS'},
    {'df': eda_el, 'label': 'EDA - EL (Tonic)', 'color': 'darkcyan', 'ylabel': 'uS'},
    {'df': ppg_pi, 'label': 'PPG - PI (Infrared)', 'color': 'crimson', 'ylabel': 'A.U.'},
    {'df': ppg_pr, 'label': 'PPG - PR (Red)', 'color': 'orangered', 'ylabel': 'A.U.'},
    {'df': ppg_pg, 'label': 'PPG - PG (Green)', 'color': 'forestgreen', 'ylabel': 'A.U.'}
]

for i, canal in enumerate(canales):
    ax = axes[i]
    target_df = canal['df']
    if not target_df.empty:
        ax.plot(target_df['Time_Sec'], target_df['Value'], color=canal['color'], label=canal['label'], alpha=0.9)
        ax.set_ylabel(canal['ylabel'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'Sin datos para {canal["label"]}', ha='center', va='center')

# Ajustes estéticos finales
axes[-1].set_xlabel('Tiempo transcurrido (segundos)')
fig.suptitle('Visualización de Señales Autónomas de EmotiBit', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()