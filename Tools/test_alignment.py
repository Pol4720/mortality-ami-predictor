"""Test de alineación de target con eliminación de duplicados."""
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from src.cleaning import DataCleaner, CleaningConfig

# Crear dataset con duplicados
np.random.seed(42)

df = pd.DataFrame({
    'edad': [60, 70, 80, 60, 70, 90] + [np.random.randint(50, 90) for _ in range(94)],
    'presion': [120, 130, 140, 120, 130, 150] + [np.random.randint(100, 180) for _ in range(94)],
    'mortality': [0, 0, 1, 0, 0, 1] + [np.random.choice([0, 1], p=[0.9, 0.1]) for _ in range(94)]
})

print('=== DATASET ORIGINAL ===')
print(df.head(10))
print(f'Shape: {df.shape}')
print(f'Mortalidad: {df["mortality"].sum()} muertes')

# Aplicar limpieza
config = CleaningConfig(
    drop_duplicates=True,
    numeric_imputation='none',
    categorical_imputation='none',
    outlier_method='none'
)

cleaner = DataCleaner(config)
df_clean = cleaner.fit_transform(df, target_column='mortality')

print('\n=== DATASET LIMPIO ===')
print(df_clean.head(10))
print(f'Shape: {df_clean.shape}')
print(f'Mortalidad: {df_clean["mortality"].sum()} muertes')

# Verificar correlaciones
corr_orig, _ = pointbiserialr(df['mortality'], df['edad'])
corr_clean, _ = pointbiserialr(df_clean['mortality'], df_clean['edad'])

print('\n=== VERIFICACIÓN DE ALINEACIÓN ===')
print(f'Correlación edad-mortality original: {corr_orig:.4f}')
print(f'Correlación edad-mortality limpio: {corr_clean:.4f}')

if abs(corr_orig - corr_clean) < 0.15:
    print('✅ Correlaciones similares - alineación correcta!')
else:
    print('❌ ERROR: Correlaciones muy diferentes - problema de alineación!')
