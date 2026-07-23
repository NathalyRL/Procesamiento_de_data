"""
Visualización de los CSV ya limpios/filtrados generados por limpiar_emotibit.py
(archivos con sufijo _EDA y _PPG).

También puedes editar las rutas por defecto más abajo y correr el script
sin argumentos.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Rutas por defecto (edítalas si prefieres no usar argumentos de consola)
# ---------------------------------------------------------
ruta_eda_default = r"D:\Documentos\Ayudante de Investigacion\Emotibit_Limpios\EDA\23_04_01_01.csv"
ruta_ppg_default = r"D:\Documentos\Ayudante de Investigacion\Emotibit_Limpios\PPG\23_04_01_02.csv"


def leer_csv_con_metadata(filepath):
    """
    Lee un CSV generado por limpiar_emotibit.py, que trae líneas de metadata
    al inicio con el formato: # fs_NombreColumna_Hz=valor

    Devuelve (DataFrame, dict_fs) donde dict_fs = {"NombreColumna": fs_valor}
    """
    fs_info = {}
    with open(filepath, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea.startswith("#"):
                break
            if "fs_" in linea and "_Hz=" in linea:
                contenido = linea.lstrip("#").strip()
                nombre_col, valor = contenido.split("_Hz=")
                nombre_col = nombre_col.replace("fs_", "")
                try:
                    fs_info[nombre_col] = float(valor)
                except ValueError:
                    pass

    df = pd.read_csv(filepath, comment="#")
    return df, fs_info


def graficar_csv(df, fs_info, titulo_general):
    """Un subplot por columna de señal (todas menos Time_Sec)."""
    columnas_señal = [c for c in df.columns if c != "Time_Sec"]
    if not columnas_señal:
        print(f"⚠️  '{titulo_general}': no se encontraron columnas de señal para graficar.")
        return

    n = len(columnas_señal)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for i, (ax, col) in enumerate(zip(axes, columnas_señal)):
        ax.plot(df["Time_Sec"], df[col], linewidth=0.8, color=colors[i % len(colors)])
        fs_val = fs_info.get(col)
        titulo = f"{col}" + (f"  (fs = {fs_val:.2f} Hz)" if fs_val else "")
        ax.set_title(titulo)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True, alpha=0.3)

    fig.suptitle(titulo_general, fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) >= 3:
        ruta_eda = sys.argv[1]
        ruta_ppg = sys.argv[2]
    else:
        ruta_eda = ruta_eda_default
        ruta_ppg = ruta_ppg_default

    print(f"Leyendo EDA: {ruta_eda}")
    df_eda, fs_eda = leer_csv_con_metadata(ruta_eda)
    print(f"  -> {len(df_eda)} filas, fs detectadas: {fs_eda}")
    graficar_csv(df_eda, fs_eda, "EDA (limpio)")

    print(f"\nLeyendo PPG: {ruta_ppg}")
    df_ppg, fs_ppg = leer_csv_con_metadata(ruta_ppg)
    print(f"  -> {len(df_ppg)} filas, fs detectadas: {fs_ppg}")
    graficar_csv(df_ppg, fs_ppg, "PPG (limpio)")


if __name__ == "__main__":
    main()