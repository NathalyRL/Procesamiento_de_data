import os
import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import io

# --- CONFIGURACIÓN ---
carpeta_recortes = r"D:\Documentos\Ayudante de Investigacion\Codigos\Pruebas"
carpeta_final = r"D:\Documentos\Ayudante de Investigacion\Codigos\Pruebas_ruido_trafico"
ruta_trafico = r"D:\Documentos\Ayudante de Investigacion\Codigos\Efectos de ruido\coches_atrapados_tráfico_y_tocando_la_bocina,_en_la_distancia_con_un_leve.mp3"

# RESPUESTA AL IMPULSO (IR)
# Debes conseguir un archivo .wav de un "Impulse Response" de una calle o plaza.
# Si no tienes uno, el script saltará la convolución y solo mezclará.
ruta_ir_calle = r"D:\Documentos\Ayudante de Investigacion\Codigos\Efectos de ruido\1st_baptist_nashville_balcony.wav" #Respuesta al impulso del ambiente de una calle.

# AJUSTES
SNR_DB = -15  # Relación señal-ruido (qué tan fuerte está el tráfico)
VOL_VOZ = -3  # Bajar un poco la voz para evitar saturación

def pydub_to_np(audio):
    """Convierte AudioSegment a array de numpy para convolución"""
    return np.array(audio.get_array_of_samples(), dtype=np.float32)

def np_to_pydub(data, original_segment):
    """Convierte array de numpy de vuelta a AudioSegment"""
    data = np.clip(data, -32768, 32767).astype(np.int16)
    return AudioSegment(
        data.tobytes(), 
        frame_rate=original_segment.frame_rate,
        sample_width=original_segment.sample_width, 
        channels=original_segment.channels
    )

def procesar_realismo():
    # 1. Cargar ambiente y IR
    ambiente = AudioSegment.from_file(ruta_trafico)
    ir_segment = None
    if os.path.exists(ruta_ir_calle):
        ir_segment = AudioSegment.from_file(ruta_ir_calle).set_channels(1)
        print("✅ Respuesta al Impulso cargada.")

    for root, dirs, files in os.walk(carpeta_recortes):
        for f in files:
            if f.endswith(".wav"):
                ruta_v = os.path.join(root, f)
                voz = AudioSegment.from_wav(ruta_v).set_channels(1)
                
                # --- PASO 1: CONVOLUCIÓN (Acústica) ---
                if ir_segment:
                    # Convertir a numpy para procesar matemáticamente
                    voz_np = pydub_to_np(voz)
                    ir_np = pydub_to_np(ir_segment)
                    
                    # Convolución rápida usando FFT
                    voz_convolved_np = fftconvolve(voz_np, ir_np, mode='full')
                    
                    # Volver a pydub y recortar al tamaño original
                    voz = np_to_pydub(voz_convolved_np, voz)[:len(voz)]
                
                # --- PASO 2: ECUALIZACIÓN RÁPIDA (Filtro de calle) ---
                # Quitamos graves (proximity effect del micro de estudio)
                voz = voz.high_pass_filter(200).apply_gain(VOL_VOZ)

                # --- PASO 3: MEZCLA ADITIVA (Tráfico) ---
                # Mezclamos el tráfico con el volumen ajustado
                mezcla = voz.overlay(ambiente + SNR_DB, loop=True)

                # --- GUARDAR ---
                rel_path = os.path.relpath(root, carpeta_recortes)
                out_dir = os.path.join(carpeta_final, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                mezcla.export(os.path.join(out_dir, f), format="wav")
                print(f"Procesado: {f}")

if __name__ == "__main__":
    procesar_realismo()