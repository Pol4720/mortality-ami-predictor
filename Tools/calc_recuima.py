import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

df = pd.read_csv('../data/recuima-020425.csv', sep=';', low_memory=False)
df['mortality'] = (df['estado_vital'] == 'fallecido').astype(int)
df['filtrado_glomerular'] = pd.to_numeric(df['filtrado_glomerular'].astype(str).str.replace(',', '.'), errors='coerce')

comp = df['complicaciones'].fillna('').str.upper()
df['fv_tv'] = comp.str.contains('FV|TV', regex=True, na=False).astype(int)
df['bav'] = comp.str.contains('BAV3|BAV2:1', regex=True, na=False).astype(int)

# ECG-N: contar derivaciones AFECTADAS (con valor != 0), NO sumar valores
ecg = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','d1','d2','d3','avl','avf','avr']
df['ecg_n'] = (df[ecg].fillna(0) != 0).sum(axis=1)

# RECUIMA con pesos de tesis: FG=3pts, FV/TV=2pts, demás=1pt (max=10)
df['recuima'] = ((df['edad']>70).astype(int) +                           # 1 pt
                 (df['presion_arterial_sistolica']<100).astype(int) +    # 1 pt
                 (df['filtrado_glomerular']<60).astype(int) * 3 +        # 3 pts (FG)
                 (df['ecg_n']>7).astype(int) +                           # 1 pt
                 (df['indice_killip']=='IV').astype(int) +               # 1 pt
                 df['fv_tv'] * 2 +                                        # 2 pts
                 df['bav'])                                               # 1 pt

valid = df.dropna(subset=['mortality','recuima'])
y = valid['mortality'].values
scores = valid['recuima'].values

print(f"N validos: {len(valid)}")

# Bootstrap para IC
np.random.seed(42)
aurocs = []
for _ in range(1000):
    idx = np.random.choice(len(y), len(y), replace=True)
    aurocs.append(roc_auc_score(y[idx], scores[idx]))

auroc = roc_auc_score(y, scores)
ci_low = np.percentile(aurocs, 2.5)
ci_high = np.percentile(aurocs, 97.5)
print(f'RECUIMA AUROC: {auroc:.3f} [{ci_low:.3f}-{ci_high:.3f}]')

# Metricas con umbral >= 3
y_pred = (scores >= 3).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
sens = tp/(tp+fn)*100
spec = tn/(tn+fp)*100
ppv = tp/(tp+fp)*100
npv = tn/(tn+fn)*100
acc = (tp+tn)/len(y)*100
lr_pos = (tp/(tp+fn))/(fp/(tn+fp))
lr_neg = (fn/(tp+fn))/(tn/(tn+fp))

print(f'Sensibilidad: {sens:.2f}%')
print(f'Especificidad: {spec:.2f}%')
print(f'VPP: {ppv:.2f}%')
print(f'VPN: {npv:.2f}%')
print(f'Exactitud: {acc:.2f}%')
print(f'LR+: {lr_pos:.2f}')
print(f'LR-: {lr_neg:.2f}')
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')

# Componentes individuales
print("\n--- COMPONENTES ---")
print(f"Edad > 70: {(df['edad']>70).sum()}")
print(f"TAS < 100: {(df['presion_arterial_sistolica']<100).sum()}")
print(f"FG < 60: {(df['filtrado_glomerular']<60).sum()}")
print(f"ECG > 7 derivaciones: {(df['ecg_n']>7).sum()}")
print(f"Killip IV: {(df['indice_killip']=='IV').sum()}")
print(f"FV/TV: {df['fv_tv'].sum()}")
print(f"BAV alto grado: {df['bav'].sum()}")
