import librosa
import numpy as np
from pydub import AudioSegment
import os

def extract_segments(input_file, output_folder, beep_freq=1000, threshold=0.3):
    """
    Analiza un archivo .wav y lo corta basándose en la frecuencia del pitido.
    """
    print(f"--- Iniciando análisis de: {input_file} ---")
    
    # 1. Cargar el audio para análisis (librosa es mejor para detectar frecuencias)
    y, sr = librosa.load(input_file)
    
    # Transformada de Fourier para ver frecuencias
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Encontrar el índice de la frecuencia del pitido
    target_idx = (np.abs(freqs - beep_freq)).argmin()
    beep_energy = S[target_idx]
    
    # Normalizar energía entre 0 y 1
    beep_energy = beep_energy / (np.max(beep_energy) + 1e-9)
    
    # Encontrar frames donde la energía supera el umbral
    frames = np.where(beep_energy > threshold)[0]
    times = librosa.frames_to_time(frames, sr=sr)
    
    # Agrupar detecciones para no detectar el mismo pitido muchas veces
    # Como tus frases duran ~5s, un margen de 2s es seguro.
    final_beeps = []
    if len(times) > 0:
        final_beeps.append(times[0])
        for t in times:
            if t - final_beeps[-1] > 2.0: # Salto de 2 segundos mínimo entre pitidos
                final_beeps.append(t)

    print(f"Se detectaron {len(final_beeps)} pitidos.")

    # 2. Cargar con pydub para el corte de alta precisión
    audio = AudioSegment.from_wav(input_file)
    
    # Creamos la lista de cortes (incluyendo el final del audio)
    # Si la frase empieza DESPUÉS del primer pitido, ignoramos el fragmento 0 antes del primer pitido
    timestamps_ms = [t * 1000 for t in final_beeps]
    timestamps_ms.append(len(audio)) 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. Guardar fragmentos
    for i in range(len(timestamps_ms) - 1):
        start = timestamps_ms[i]
        end = timestamps_ms[i+1]
        
        # AJUSTE FINO (Padding):
        # Le sumamos 300ms al inicio para saltarnos el sonido del propio pitido
        # Le restamos 100ms al final para no pillar el inicio del siguiente pitido
        clip = audio[start + 300 : end - 100]
        
        if len(clip) > 1000: # Solo guardar si dura más de 1 segundo
            nombre_archivo = f"clip_{i+1:03d}.wav" # clip_001.wav, clip_002.wav...
            clip.export(os.path.join(output_folder, nombre_archivo), format="wav")
            print(f"Exportado: {nombre_archivo} | Duración: {len(clip)/1000:.2f}s")

# --- CONFIGURACIÓN DEL TEST ---
archivo_test = "02_00_05.wav" # <-- CAMBIA ESTO
carpeta_destino = r"D:\Documentos\Ayudante de Investigacion\Codigos\test_cortes" # <-- CAMBIA ESTO

extract_segments(archivo_test, carpeta_destino, beep_freq=1200, threshold=0.4)