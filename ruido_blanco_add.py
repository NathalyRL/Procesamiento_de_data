import os
import numpy as np
import random
from scipy.signal import fftconvolve
from pydub import AudioSegment

# --- CONFIGURACIÓN DE MOTOR (FFMPEG) ---
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg
AudioSegment.converter = os.path.join(ruta_bin_ffmpeg, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(ruta_bin_ffmpeg, "ffprobe.exe")

# --- RUTAS ---
carpeta_recortes = r"D:\Documentos\Ayudante de Investigacion\Codigos\Cortes"
carpeta_final    = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_con_ruido_calle"
ruta_trafico     = r"D:\Documentos\Ayudante de Investigacion\Codigos\Efectos de ruido\coches_atrapados_tráfico_y_tocando_la_bocina,_en_la_distancia_con_un_leve.mp3"
ruta_ir_calle    = r"D:\Documentos\Ayudante de Investigacion\Codigos\RI_ruidos\test_outdoor.wav"

# --- AJUSTES DE MEZCLA ---
FRECUENCIA_TRABAJO = 44100
# Si no escuchas casi el tráfico, sube este valor (ej. -10 o -5)
# Si el tráfico tapa la voz, baja este valor (ej. -25)
NIVEL_TRAFICO_DB = -8  
VOL_VOZ_POST_CONV = -1

def pydub_to_np(audio):
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return data / (2**(8 * audio.sample_width - 1))

def np_to_pydub(data, frame_rate):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.8 # Normalizamos a un nivel sólido
    data_int = (data * 32767).astype(np.int16)
    return AudioSegment(data_int.tobytes(), frame_rate=frame_rate, sample_width=2, channels=1)

def generar_comparativa():
    if not os.path.exists(ruta_ir_calle) or not os.path.exists(ruta_trafico):
        print("❌ Error: Archivos base no encontrados.")
        return

    # Cargamos el ambiente una sola vez (lo mantenemos en estéreo para el final)
    print("⚙️ Cargando y preparando ambiente...")
    ambiente_master = AudioSegment.from_file(ruta_trafico).set_frame_rate(FRECUENCIA_TRABAJO).set_channels(2)
    ir_segment = AudioSegment.from_file(ruta_ir_calle).set_frame_rate(FRECUENCIA_TRABAJO).set_channels(1)
    #ir_segment = ir_segment[170:]
    ir_np = pydub_to_np(ir_segment)

    total_procesados = 0

    for root, dirs, files in os.walk(carpeta_recortes):
        # Filtrar solo archivos .wav
        archivos_wav = [archivo for archivo in files if archivo.lower().endswith(".wav")]
        
        if not archivos_wav:
            continue

        # --- TRUCO DE CARPETAS _02 ---
        # Obtenemos la ruta relativa (ej: "Sujeto_01\Sesion_A")
        rel_path = os.path.relpath(root, carpeta_recortes)
        
        if rel_path != ".":
            # Rompemos la ruta en carpetas individuales, les agregamos _02 y las volvemos a unir
            carpetas_individuales = rel_path.split(os.sep)
            carpetas_02 = [f"{carpeta}_02" for carpeta in carpetas_individuales if carpeta]
            rel_path_02 = os.path.join(*carpetas_02)
        else:
            rel_path_02 = ""

        # Definimos el destino final respetando la nueva estructura _02
        out_dir = os.path.join(carpeta_final, rel_path_02)
        os.makedirs(out_dir, exist_ok=True)

        for f in archivos_wav:
            nombre_base = os.path.splitext(f)[0]
            voz_original = AudioSegment.from_wav(os.path.join(root, f)).set_frame_rate(FRECUENCIA_TRABAJO).set_channels(1)
            
            # 1. CONVOLUCIÓN
            voz_np = pydub_to_np(voz_original)
            conv_np = fftconvolve(voz_np, ir_np, mode='full')
            voz_conv = np_to_pydub(conv_np, FRECUENCIA_TRABAJO)[:len(voz_original)]
            
            # 2. PREPARAR VOZ
            voz_procesada = voz_conv.high_pass_filter(250).apply_gain(VOL_VOZ_POST_CONV).set_channels(2)

            # 3. SELECCIÓN ALEATORIA DE RUIDO
            if len(ambiente_master) > len(voz_procesada):
                start_limit = int(len(ambiente_master) - len(voz_procesada) - 1)
                start_time = random.randint(0, max(0, start_limit))
                pedazo_trafico = ambiente_master[start_time : start_time + len(voz_procesada)]
            else:
                pedazo_trafico = ambiente_master

            # 4. MEZCLA FINAL
            mezcla_final = voz_procesada.overlay(pedazo_trafico.apply_gain(NIVEL_TRAFICO_DB))

            # 5. GUARDAR CON SUFIJO _02
            ruta_salida_audio = os.path.join(out_dir, f"{nombre_base}_02.wav")
            mezcla_final.export(ruta_salida_audio, format="wav")
            
            total_procesados += 1
            print(f"✔ Procesado en espejo: {rel_path_02} -> {nombre_base}_02.wav")

    print(f"\n🚀 ¡Procesamiento masivo terminado con éxito!")
    print(f"📦 Total de audios generados: {total_procesados}")
    print(f"📂 Revisa los resultados en: {carpeta_final}")

if __name__ == "__main__":
    generar_comparativa()