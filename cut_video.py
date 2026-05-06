import librosa
import numpy as np
from pydub import AudioSegment
import os

# --- CONFIGURACIÓN MASIVA ---
# Carpeta donde están tus 50 archivos .wav originales
carpeta_entrada = r"D:\Documentos\Ayudante de Investigacion\Codigos\Archivo_wav_convertidos"
# Carpeta donde se guardarán todos los recortes
carpeta_salida_master = r"D:\Documentos\Ayudante de Investigacion\Codigos\Cortes"

frecuencia_pitido = 1200 
umbral_sensibilidad = 0.7 

def procesar_archivo(ruta_wav, carpeta_destino, beep_freq, threshold):
    nombre_base = os.path.splitext(os.path.basename(ruta_wav))[0]
    
    print(f"\n>>> Analizando: {nombre_base}.wav")
    
    # Carga y detección de pitidos
    y, sr = librosa.load(ruta_wav)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    idx = (np.abs(freqs - beep_freq)).argmin()
    
    energia = S[idx]
    energia = energia / (np.max(energia) + 1e-9)
    
    frames = np.where(energia > threshold)[0]
    tiempos = librosa.frames_to_time(frames, sr=sr)
    
    pitidos_finales = []
    if len(tiempos) > 0:
        pitidos_finales.append(tiempos[0])
        for t in tiempos:
            if t - pitidos_finales[-1] > 2.0:
                pitidos_finales.append(t)
    
    print(f"    Se encontraron {len(pitidos_finales)} pitidos.")

    # Recorte
    audio = AudioSegment.from_wav(ruta_wav)
    
    # Opcional: Si quieres que cada audio original tenga su propia subcarpeta, 
    # descomenta las siguientes dos líneas:
    carpeta_destino = os.path.join(carpeta_destino, nombre_base)
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)

    for i, tiempo_pitido in enumerate(pitidos_finales):
        inicio_ms = int(tiempo_pitido * 1000) + 300
        
        if i < len(pitidos_finales) - 1:
            fin_ms = int(pitidos_finales[i+1] * 1000) - 50 
        else:
            fin_ms = inicio_ms + 4590
            
        segmento = audio[inicio_ms:fin_ms]
        
        # Uso :03d por si tienes más de 99 clips, para que el orden alfabético se mantenga
        nombre_clip = f"{nombre_base}_{i+1:03d}.wav"
        ruta_final = os.path.join(carpeta_destino, nombre_clip)
        
        segmento.export(ruta_final, format="wav")

    print(f"    ✓ {nombre_base} procesado con éxito.")

if __name__ == "__main__":
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida_master):
        os.makedirs(carpeta_salida_master)

    # Listar todos los archivos en la carpeta de entrada
    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(".wav")]
    
    total = len(archivos)
    print(f"Se encontraron {total} archivos para procesar.")

    for index, nombre_archivo in enumerate(archivos):
        ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
        print(f"[{index + 1}/{total}]", end="")
        procesar_archivo(ruta_completa, carpeta_salida_master, frecuencia_pitido, umbral_sensibilidad)

    print("\n========================================")
    print("¡PROCESAMIENTO MASIVO COMPLETADO!")
    print(f"Los clips están en: {carpeta_salida_master}")
    print("========================================")