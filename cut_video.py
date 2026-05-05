import librosa
import numpy as np
from pydub import AudioSegment
import os

# --- CONFIGURACIÓN DEL ARCHIVO ÚNICO ---
archivo_entrada = "02_00_05.wav"  # Pon aquí el nombre de tu archivo
carpeta_salida = r"D:\Documentos\Ayudante de Investigacion\Codigos\test_cortes"
frecuencia_pitido = 1200  # Basado en el pico que vimos en Audacity
umbral_sensibilidad = 0.7 

def procesar_archivo_unico(ruta_wav, salida, beep_freq, threshold):
    # 1. Extraer nombre base para el formato: Nombre_001.wav
    nombre_base = os.path.splitext(os.path.basename(ruta_wav))[0]
    
    print(f"Analizando: {nombre_base}...")
    
    # 2. Detección de pitidos con Librosa
    y, sr = librosa.load(ruta_wav)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    idx = (np.abs(freqs - beep_freq)).argmin()
    
    # Normalizamos la energía del pitido
    energia = S[idx]
    energia = energia / (np.max(energia) + 1e-9)
    
    # Buscamos los tiempos donde suena el pitido
    frames = np.where(energia > threshold)[0]
    tiempos = librosa.frames_to_time(frames, sr=sr)
    
    pitidos_finales = []
    if len(tiempos) > 0:
        pitidos_finales.append(tiempos[0])
        for t in tiempos:
            if t - pitidos_finales[-1] > 2.0: # Evita detectar el mismo pitido
                pitidos_finales.append(t)
    
    print(f"Se encontraron {len(pitidos_finales)} pitidos.")

    # 3. Recorte con Pydub
    audio = AudioSegment.from_wav(ruta_wav)
    if not os.path.exists(salida):
        os.makedirs(salida)

    for i, tiempo_pitido in enumerate(pitidos_finales):
        # El clip empieza 300ms después del inicio del pitido (para que no se oiga)
        inicio_ms = int(tiempo_pitido * 1000) + 300
        
        # Lógica de fin de clip
        if i < len(pitidos_finales) - 1:
            # Si hay otro pitido después, cortamos justo antes del siguiente pitido
            fin_ms = int(pitidos_finales[i+1] * 1000) - 50 
        else:
            # Si es el ÚLTIMO, forzamos duración de 4.59 segundos (4590 ms)
            fin_ms = inicio_ms + 4590
            
        # Realizar el corte
        segmento = audio[inicio_ms:fin_ms]
        
        # Nombre de archivo solicitado: NombreOriginal_Numero.wav
        nombre_clip = f"{nombre_base}_{i+1:02d}.wav"
        ruta_final = os.path.join(salida, nombre_clip)
        
        segmento.export(ruta_final, format="wav")
        print(f" -> Exportado: {nombre_clip} (Duración: {len(segmento)/1000:.2f}s)")

if __name__ == "__main__":
    if os.path.exists(archivo_entrada):
        procesar_archivo_unico(archivo_entrada, carpeta_salida, frecuencia_pitido, umbral_sensibilidad)
    else:
        print(f"Error: No se encuentra el archivo '{archivo_entrada}'")