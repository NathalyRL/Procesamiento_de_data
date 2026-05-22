import os
import numpy as np
from pydub import AudioSegment

# --- CONFIGURACIÓN DE MOTOR (FFMPEG) ---
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg
AudioSegment.converter = os.path.join(ruta_bin_ffmpeg, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(ruta_bin_ffmpeg, "ffprobe.exe")

# --- RUTAS ---
carpeta_recortes = r"D:\Documentos\Ayudante de Investigacion\Codigos\Cortes"
carpeta_final    = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_con_ruido_blanco"

# --- CONFIGURACIÓN DEL SUFIJO DINÁMICO ---
SUFIJO = "_03"

# --- AJUSTES DE RUIDO BLANCO (SNR) ---
FRECUENCIA_TRABAJO = 44100
SNR_DESEADO_DB = 15  #Valor de dB

def pydub_to_np(audio):
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return data / (2**(8 * audio.sample_width - 1))

def np_to_pydub(data, frame_rate):
    # Control de clipping/saturación por si la suma del ruido supera el límite
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data = data / max_val * 0.95
        
    data_int = (data * 32767).astype(np.int16)
    return AudioSegment(data_int.tobytes(), frame_rate=frame_rate, sample_width=2, channels=1)

def generar_ruido_blanco():
    if not os.path.exists(carpeta_recortes):
        print("❌ Error: La carpeta de recortes original no existe.")
        return

    print(f"⚙️ Iniciando generación de Ruido Blanco Gaussiano (SNR: {SNR_DESEADO_DB} dB)...")
    total_procesados = 0

    for root, dirs, files in os.walk(carpeta_recortes):
        archivos_wav = [archivo for archivo in files if archivo.lower().endswith(".wav")]
        
        if not archivos_wav:
            continue

        # --- REPLICAR ESTRUCTURA DE CARPETAS _02 ---
        rel_path = os.path.relpath(root, carpeta_recortes)
        if rel_path != ".":
            carpetas_individuales = rel_path.split(os.sep)
            carpetas_con_sufijo = [f"{carpeta}{SUFIJO}" for carpeta in carpetas_individuales if carpeta]
            rel_path_salida = os.path.join(*carpetas_con_sufijo)
        else:
            rel_path_salida = ""

        out_dir = os.path.join(carpeta_final, rel_path_salida)
        os.makedirs(out_dir, exist_ok=True)

        for f in archivos_wav:
            nombre_base = os.path.splitext(f)[0]
            
            # Cargar voz original en mono
            voz_original = AudioSegment.from_wav(os.path.join(root, f)).set_frame_rate(FRECUENCIA_TRABAJO).set_channels(1)
            voz_np = pydub_to_np(voz_original)
            
            # --- MATEMÁTICAS DEL RUIDO GAUSSIANO (AWGN) ---
            # 1. Calcular la potencia de la señal (RMS)
            potencia_senal = np.mean(voz_np ** 2)
            
            # 2. Calcular la potencia que debe tener el ruido según el SNR deseado
            # Fórmula: Potencia_Ruido = Potencia_Señal / (10 ^ (SNR / 10))
            potencia_ruido = potencia_senal / (10 ** (SNR_DESEADO_DB / 10.0))
            
            # 3. Generar la distribución normal (Gaussiana) usando la desviación estándar (raíz de la potencia)
            desviacion_estandar = np.sqrt(potencia_ruido)
            ruido_gaussiano = np.random.normal(0, desviacion_estandar, voz_np.shape)
            
            # 4. Sumar el ruido directamente a la señal
            audio_contaminado_np = voz_np + ruido_gaussiano
            
            # Convertir de vuelta a AudioSegment
            audio_final = np_to_pydub(audio_contaminado_np, FRECUENCIA_TRABAJO)

            # --- GUARDAR CON SUFIJO  ---
            ruta_salida_audio = os.path.join(out_dir, f"{nombre_base}{SUFIJO}.wav")
            audio_final.export(ruta_salida_audio, format="wav")
            
            total_procesados += 1
            print(f"✔ Gaussiano aplicado: {rel_path_salida} -> {nombre_base}{SUFIJO}.wav")

    print(f"\n🚀 ¡Procesamiento masivo de ruido blanco terminado!")
    print(f"📦 Total de audios generados: {total_procesados}")
    print(f"📂 Carpeta de destino: {carpeta_final}")

if __name__ == "__main__":
    generar_ruido_blanco()