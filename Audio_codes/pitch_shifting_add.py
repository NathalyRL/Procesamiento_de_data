import os
from pydub import AudioSegment

# --- CONFIGURACIÓN DE MOTOR (FFMPEG) ---
ruta_bin_ffmpeg = r"D:\Documentos\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin"
os.environ["PATH"] += os.pathsep + ruta_bin_ffmpeg
AudioSegment.converter = os.path.join(ruta_bin_ffmpeg, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(ruta_bin_ffmpeg, "ffprobe.exe")

# --- RUTAS ---
carpeta_recortes = r"D:\Documentos\Ayudante de Investigacion\Codigos\Cortes_partes2"
carpeta_final    = r"D:\Documentos\Ayudante de Investigacion\Codigos\Audios_con_pitch_agudo2"

# --- CONFIGURACIÓN DE SUFIJO ---
SUFIJO = "_04"

# --- AJUSTES DE PITCH SHIFTING ---
FRECUENCIA_TRABAJO = 44100
# El cambio se mide en semitonos:
# Valores positivos agudizan la voz (ej: 2.0, 3.5) -> Voz más fina / rápida
# Valores negativos gravean la voz (ej: -2.0, -4.0) -> Voz más gruesa / lenta
PITCH_SHIFT_SEMITONES = 2.0 

def cambiar_pitch(audio, semitonos):
    """Modifica el tono del audio usando el método de velocidad/muestreo constante"""
    if semitonos == 0:
        return audio
    
    # Fórmula matemática para calcular el factor de cambio según los semitonos
    factor = 2.0 ** (semitonos / 12.0)
    
    # Calculamos la nueva tasa de muestreo ficticia
    nuevo_frame_rate = int(audio.frame_rate * factor)
    
    # Forzamos a Pydub a leer los mismos datos con la nueva tasa (cambia el tono)
    audio_modificado = audio._spawn(audio.raw_data, overrides={'frame_rate': nuevo_frame_rate})
    
    # Normalizamos el audio de vuelta a nuestra frecuencia de trabajo estándar
    return audio_modificado.set_frame_rate(FRECUENCIA_TRABAJO)

def procesar_pitch_masivo():
    if not os.path.exists(carpeta_recortes):
        print("❌ Error: La carpeta de recortes original no existe.")
        return

    print(f"⚙️ Iniciando procesamiento de Pitch Shifting ({PITCH_SHIFT_SEMITONES} semitonos)...")
    print(f"🏷️ Aplicando sufijo configurado: {SUFIJO}")
    total_procesados = 0

    for root, dirs, files in os.walk(carpeta_recortes):
        archivos_wav = [archivo for archivo in files if archivo.lower().endswith(".wav")]
        
        if not archivos_wav:
            continue

        # --- REPLICAR ESTRUCTURA DE CARPETAS CON SUFIJO VARIABLE ---
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
            
            # Cargar voz original
            voz_original = AudioSegment.from_wav(os.path.join(root, f)).set_frame_rate(FRECUENCIA_TRABAJO).set_channels(1)
            
            # Aplicar la transformación de tono
            voz_modificada = cambiar_pitch(voz_original, PITCH_SHIFT_SEMITONES)

            # --- GUARDAR CON SUFIJO ---
            ruta_salida_audio = os.path.join(out_dir, f"{nombre_base}{SUFIJO}.wav")
            voz_modificada.export(ruta_salida_audio, format="wav")
            
            total_procesados += 1
            print(f"✔ Pitch aplicado: {rel_path_salida} -> {nombre_base}{SUFIJO}.wav")

    print(f"\n🚀 ¡Procesamiento masivo de Pitch Shifting terminado!")
    print(f"📦 Total de audios generados: {total_procesados}")
    print(f"📂 Carpeta de destino: {carpeta_final}")

if __name__ == "__main__":
    procesar_pitch_masivo()