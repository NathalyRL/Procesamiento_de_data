import numpy as np
import soundfile as sf
import os

carpeta = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_Limpio"

rms_valores = []

for root, dirs, files in os.walk(carpeta):
    for archivo in files:
        if archivo.lower().endswith('.wav'):
            audio, sr = sf.read(os.path.join(root, archivo), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                rms_valores.append(rms)

print(f"Archivos analizados : {len(rms_valores)}")
print(f"RMS mínimo          : {min(rms_valores):.4f}")
print(f"RMS máximo          : {max(rms_valores):.4f}")
print(f"RMS promedio        : {np.mean(rms_valores):.4f}")  # ← usa este valor
print(f"RMS mediana         : {np.median(rms_valores):.4f}")