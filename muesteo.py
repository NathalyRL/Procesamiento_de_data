import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# Configuración
file_path = r"D:\Documentos\Ayudante de Investigacion\Codigos\01_05_00.txt"
fs = 250.0  # Frecuencia de muestreo definida por OpenBCI

try:
    # 1. Carga de datos
    df = pd.read_csv(file_path, comment='%', skipinitialspace=True)
    signal = df['Analog Channel 0'].values
    
    # Creamos un eje de tiempo en segundos basado en el índice y la frecuencia de muestreo
    # Esto asegura que visualicemos los 250Hz correctamente
    tiempo_segundos = np.arange(len(signal)) / fs

    # 2. Procesamiento para Frecuencia (FFT)
    # Restamos la media para eliminar el componente DC (el valor base de la señal)
    signal_detrended = signal - np.mean(signal)
    
    n = len(signal_detrended)
    yf = fft(signal_detrended)
    xf = fftfreq(n, 1/fs)

    # Solo nos interesa la mitad positiva del espectro
    xf_final = xf[:n//2]
    yf_final = 2.0/n * np.abs(yf[0:n//2])

    # 3. Visualización
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Gráfica en el Tiempo
    ax1.plot(tiempo_segundos, signal, color='red', linewidth=0.8)
    ax1.set_title('Señal en el Tiempo (Sensor de Pulso - D11)')
    ax1.set_xlabel('Tiempo (segundos)')
    ax1.set_ylabel('Amplitud (0-1023)')
    ax1.grid(True, alpha=0.3)
    # Hacemos zoom a los primeros 5 segundos para ver los puntos de muestreo
    ax1.set_xlim(0, 5) 

    # Gráfica en la Frecuencia (FFT)
    ax2.plot(xf_final, yf_final, color='blue')
    ax2.set_title('Análisis de Frecuencia (FFT)')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Magnitud')
    ax2.set_xlim(0, 10)  # Limitamos a 10Hz porque el corazón late entre 1 y 3 Hz (60-180 BPM)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Verificación de muestreo en consola
    print(f"Número total de muestras: {len(df)}")
    print(f"Duración estimada: {len(df)/fs:.2f} segundos")

except Exception as e:
    print(f"Error al procesar el archivo: {e}")