import numpy as np

# =============================================================================
# RUTA AL ARCHIVO .npz QUE QUIERES INSPECCIONAR
# =============================================================================
ruta_npz = r"D:\Documentos\Ayudante de Investigacion\Codigos\AUDIOS\MFCCs\01_04_05_1_01_07\01_04_05_1_006_01_07.npz"

# =============================================================================
# CARGAR E INSPECCIONAR
# =============================================================================
data = np.load(ruta_npz, allow_pickle=True)

print("=" * 50)
print("  CONTENIDO DEL ARCHIVO .npz")
print("=" * 50)

print(f"\n  Arrays guardados : {list(data.keys())}")

frames  = data["frames"]
emocion = str(data["emocion"][0])

print(f"\n  emocion    : {emocion}")

print(f"\n  frames:")
print(f"    Shape    : {frames.shape}  → ({frames.shape[0]} frames, {frames.shape[1]} coeficientes)")
print(f"    Dtype    : {frames.dtype}")
print(f"    Min      : {frames.min():.4f}")
print(f"    Max      : {frames.max():.4f}")
print(f"    Media    : {frames.mean():.4f}")

print(f"\n  Primeros 3 frames (primeros 10 coeficientes c/u):")
for i in range(min(3, frames.shape[0])):
    print(f"    frame {i}: {frames[i, :10]}")

print("\n" + "=" * 50)