import numpy as np, pandas as pd
df = pd.read_csv(r'D:\Documentos\Ayudante de Investigacion\OPENBCI\17_03_00.txt', skiprows=4, sep=',', engine='python')
df.columns = [c.strip() for c in df.columns]
ppg = df['Analog Channel 0'].to_numpy()
diff = np.diff(ppg)
pct_igual = 100 * (diff == 0).sum() / len(diff)
print(f"Muestras idénticas consecutivas: {pct_igual:.1f}%")
# Si sale >50% → confirmado sample-and-hold