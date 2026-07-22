import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos saltando metadatos
ruta_archivo = r"D:\Documentos\Ayudante de Investigacion\OPENBCI\21_08_00.txt"  # Cambia esto por el nombre real de tu archivo
df = pd.read_csv(ruta_archivo, comment='%')

# 2. Limpiar espacios ocultos en los nombres de las columnas
df.columns = df.columns.str.strip()

# 3. SOLUCIÓN: Reiniciar el índice y usar el número de fila continuo (0, 1, 2, 3...)
# Cada fila representa 1/250 de segundo.
df = df.reset_index(drop=True)
df['Tiempo (segundos)'] = df.index / 250.0

# 4. Graficar la señal respecto al tiempo continuo corregido
plt.figure(figsize=(14, 6))

# Graficamos con una línea continua sin marcadores 'o' para que no sature si la data es larga
plt.plot(df['Tiempo (segundos)'], df['Analog Channel 0'], 
         color='#ff1a1a', linestyle='-', linewidth=1, label='Canal Analógico 0 (D11)')

# Personalización de la gráfica
plt.title('Señal del Sensor de Pulso vs Tiempo Continuo (250 Hz)', fontsize=14, fontweight='bold')
plt.xlabel('Tiempo (segundos)', fontsize=12)
plt.ylabel('Valor Analógico', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Mostrar la gráfica estirada correctamente a lo largo del tiempo
plt.tight_layout()
plt.show()