#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse Complicaciones Column for Clinical Scales (RECUIMA, GRACE, TIMI)
======================================================================

This script parses the 'complicaciones' column from the RECUIMA dataset
to extract individual clinical variables needed for risk score calculations.

Variables extracted for RECUIMA scale:
- FV (Fibrilación Ventricular) - 2 points
- TV (Taquicardia Ventricular) - 2 points (combined with FV)
- BAV alto grado (BAV 2do Mobitz II, BAV 3er grado) - 1 point

Additional variables extracted:
- shock_cardiogenico
- fallo_bomba (heart failure)
- PCR (paro cardiorrespiratorio)
- EAP (edema agudo de pulmón)
- Congestión pulmonar
- Hemorragia
- Embolia pulmonar
- Reinfarto

Author: ML Team
Date: 2026-01-07
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# MAPPING DEFINITIONS FOR COMPLICACIONES TOKENS
# =============================================================================

# Tokens that indicate Ventricular Fibrillation (FV)
FV_TOKENS = {
    'fv',           # Fibrilación ventricular
    'pcr-fv',       # PCR con FV
}

# Tokens that indicate Ventricular Tachycardia (TV)
TV_TOKENS = {
    'tv',           # Taquicardia ventricular
}

# Tokens that indicate High-Grade AV Block (BAV alto grado)
# Includes: BAV 2nd degree Mobitz II, BAV 3rd degree (complete)
BAV_ALTO_GRADO_TOKENS = {
    'bav3',         # Bloqueo AV de 3er grado (completo)
    'bav2:1',       # Bloqueo AV 2:1 (considerado alto grado)
    'bav2',         # Bloqueo AV de 2do grado (puede ser Mobitz II)
    'pcr-bav-pericar',  # PCR con BAV
    'bav',          # BAV genérico (considerar alto grado por contexto)
}

# BAV de bajo grado (1er grado) - NO cuenta para RECUIMA
BAV_BAJO_GRADO_TOKENS = {
    'bav1',         # Bloqueo AV de 1er grado
}

# Shock cardiogénico
SHOCK_TOKENS = {
    'shock',        # Shock genérico
    'shock_c',      # Shock cardiogénico
}

# Fallo de bomba / Insuficiencia cardíaca
FALLO_BOMBA_TOKENS = {
    'fallo_bomba',  # Fallo de bomba explícito
    'congpulm',     # Congestión pulmonar (signo de IC)
    'eap',          # Edema agudo de pulmón
}

# Paro cardiorrespiratorio
PCR_TOKENS = {
    'pcr',          # Paro cardiorrespiratorio
    'pcr-fv',       # PCR con FV
    'pcr-bav-pericar',  # PCR con BAV
    'asist',        # Asistolia
}

# Arritmias (otras)
OTRAS_ARRITMIAS_TOKENS = {
    'farvr',        # Fibrilación auricular con RVR
    'farva',        # Fibrilación auricular
    'farvl',        # Flutter auricular
    'arritmia',     # Arritmia genérica
    'ts',           # Taquicardia sinusal
    'bs',           # Bradicardia sinusal
}

# Complicaciones mecánicas
COMPLICACIONES_MECANICAS_TOKENS = {
    'rupt p lib',   # Ruptura de pared libre
    'rupt tiv',     # Ruptura tabique interventricular
    'rupt muscp',   # Ruptura músculo papilar
    'ruptura',      # Ruptura genérica
    'aneurap',      # Aneurisma
    'pseudoaneur',  # Pseudoaneurisma
    'taponam',      # Taponamiento cardíaco
    'insmit',       # Insuficiencia mitral
    'i.mitral',     # Insuficiencia mitral
}

# Otras complicaciones importantes
OTRAS_COMPLICACIONES = {
    'hemorragia': 'hemorragia',
    'trombo': 'trombosis',
    'embolia pulmonar': 'embolia_pulmonar',
    'reinfarto': 'reinfarto',
    'isquemia recurrente': 'isquemia_recurrente',
    'dem': 'disociacion_electromecanica',
    'irc': 'insuficiencia_renal',
    'imavd': 'infarto_vd',
    'infarto vd': 'infarto_vd',
    'imabivent': 'infarto_biventricular',
    'epistenocardica': 'pericarditis_epistenocardica',
    'sind dressler': 'sindrome_dressler',
    'cvp': 'contracciones_ventriculares_prematuras',
    'contrespon': 'no_reperfusion',
    'angor_r': 'angina_refractaria',
    'hipotension': 'hipotension',
    'dm descomp': 'diabetes_descompensada',
    'psiquiatrica': 'complicacion_psiquiatrica',
}


def parse_complicaciones_string(complicaciones_str: str) -> Dict[str, bool]:
    """
    Parse a single complicaciones string and extract binary variables.
    
    Args:
        complicaciones_str: String containing comma/semicolon separated complications
        
    Returns:
        Dictionary with binary flags for each complication type
    """
    result = {
        # RECUIMA scale variables
        'comp_fv': False,               # Fibrilación ventricular
        'comp_tv': False,               # Taquicardia ventricular  
        'comp_fv_tv': False,            # FV o TV (combined for RECUIMA)
        'comp_bav_alto_grado': False,   # BAV de alto grado
        'comp_bav_bajo_grado': False,   # BAV de bajo grado (1er grado)
        
        # Other important complications
        'comp_shock': False,            # Shock cardiogénico
        'comp_fallo_bomba': False,      # Fallo de bomba/IC
        'comp_pcr': False,              # Paro cardiorrespiratorio
        'comp_otras_arritmias': False,  # Otras arritmias
        'comp_mecanicas': False,        # Complicaciones mecánicas
        
        # Specific complications
        'comp_hemorragia': False,
        'comp_embolia_pulmonar': False,
        'comp_reinfarto': False,
        'comp_isquemia_recurrente': False,
        'comp_infarto_vd': False,
    }
    
    if pd.isna(complicaciones_str) or not str(complicaciones_str).strip():
        return result
    
    # Normalize and split
    text = str(complicaciones_str).lower().strip()
    tokens = re.split(r'[;,/|]+', text)
    tokens = [t.strip() for t in tokens if t.strip()]
    
    for token in tokens:
        # FV
        if token in FV_TOKENS:
            result['comp_fv'] = True
            result['comp_fv_tv'] = True
        
        # TV
        if token in TV_TOKENS:
            result['comp_tv'] = True
            result['comp_fv_tv'] = True
        
        # BAV alto grado
        if token in BAV_ALTO_GRADO_TOKENS:
            result['comp_bav_alto_grado'] = True
        
        # BAV bajo grado
        if token in BAV_BAJO_GRADO_TOKENS:
            result['comp_bav_bajo_grado'] = True
        
        # Shock
        if token in SHOCK_TOKENS:
            result['comp_shock'] = True
        
        # Fallo de bomba
        if token in FALLO_BOMBA_TOKENS:
            result['comp_fallo_bomba'] = True
        
        # PCR
        if token in PCR_TOKENS:
            result['comp_pcr'] = True
        
        # Otras arritmias
        if token in OTRAS_ARRITMIAS_TOKENS:
            result['comp_otras_arritmias'] = True
        
        # Complicaciones mecánicas
        if token in COMPLICACIONES_MECANICAS_TOKENS:
            result['comp_mecanicas'] = True
        
        # Specific complications
        if token == 'hemorragia':
            result['comp_hemorragia'] = True
        if token == 'embolia pulmonar':
            result['comp_embolia_pulmonar'] = True
        if token == 'reinfarto':
            result['comp_reinfarto'] = True
        if token == 'isquemia recurrente':
            result['comp_isquemia_recurrente'] = True
        if token in ('imavd', 'infarto vd'):
            result['comp_infarto_vd'] = True
    
    return result


def create_recuima_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived variables specifically for RECUIMA scale calculation.
    
    RECUIMA variables needed (from the scale):
    1. Edad > 70 años (1 pt) - from 'edad'
    2. TAS < 100 mmHg (1 pt) - from 'presion_arterial_sistolica'
    3. TFG < 60 ml/min/1.73m² (3 pts) - from 'filtrado_glomerular'
    4. > 7 derivaciones ECG afectadas (1 pt) - computed from V1-V6, D1-D3, aVL, aVF, aVR
    5. Killip-Kimball IV (1 pt) - from 'indice_killip'
    6. FV/TV (2 pts) - from parsed 'complicaciones' -> 'comp_fv_tv'
    7. BAV de alto grado (1 pt) - from parsed 'complicaciones' -> 'comp_bav_alto_grado'
    
    Args:
        df: DataFrame with parsed complications and original variables
        
    Returns:
        DataFrame with additional RECUIMA-specific derived variables
    """
    df = df.copy()
    
    # 1. Edad > 70 años
    if 'edad' in df.columns:
        df['recuima_edad_gt70'] = (df['edad'] > 70).astype(int)
    
    # 2. TAS < 100 mmHg
    tas_cols = ['presion_arterial_sistolica', 'tas', 'pas']
    for col in tas_cols:
        if col in df.columns:
            df['recuima_tas_lt100'] = (df[col] < 100).astype(int)
            break
    
    # 3. TFG < 60 ml/min/1.73m²
    if 'filtrado_glomerular' in df.columns:
        df['recuima_tfg_lt60'] = (df['filtrado_glomerular'] < 60).astype(int)
    
    # 4. > 7 derivaciones ECG afectadas
    # Count ECG leads with ST changes
    ecg_lead_cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
                     'd1', 'd2', 'd3', 'avl', 'avf', 'avr', 'v3r', 'v4r']
    available_ecg_cols = [col for col in ecg_lead_cols if col in df.columns]
    
    if available_ecg_cols:
        # Sum of leads with changes (assuming 1 = affected, 0 = not affected)
        df['ecg_leads_affected_count'] = df[available_ecg_cols].fillna(0).sum(axis=1)
        df['recuima_ecg_gt7'] = (df['ecg_leads_affected_count'] > 7).astype(int)
    
    # 5. Killip IV
    if 'indice_killip' in df.columns:
        # Handle different formats: 'IV', 4, 'I', 1, etc.
        def is_killip_iv(val):
            if pd.isna(val):
                return 0
            val_str = str(val).upper().strip()
            if val_str in ('IV', '4'):
                return 1
            return 0
        df['recuima_killip_iv'] = df['indice_killip'].apply(is_killip_iv)
    
    # 6. FV/TV (already parsed as comp_fv_tv)
    if 'comp_fv_tv' in df.columns:
        df['recuima_fv_tv'] = df['comp_fv_tv'].astype(int)
    
    # 7. BAV alto grado (already parsed as comp_bav_alto_grado)
    if 'comp_bav_alto_grado' in df.columns:
        df['recuima_bav_alto_grado'] = df['comp_bav_alto_grado'].astype(int)
    
    # Calculate RECUIMA score if all variables present
    recuima_vars = ['recuima_edad_gt70', 'recuima_tas_lt100', 'recuima_tfg_lt60',
                    'recuima_ecg_gt7', 'recuima_killip_iv', 'recuima_fv_tv', 
                    'recuima_bav_alto_grado']
    recuima_weights = [1, 1, 3, 1, 1, 2, 1]  # Total max = 10
    
    available_recuima = [v for v in recuima_vars if v in df.columns]
    
    if len(available_recuima) == len(recuima_vars):
        df['recuima_score_calculated'] = sum(
            df[var] * weight 
            for var, weight in zip(recuima_vars, recuima_weights)
        )
        df['recuima_risk_category'] = df['recuima_score_calculated'].apply(
            lambda x: 'alto' if x >= 4 else 'bajo'
        )
        logger.info("✅ RECUIMA score calculated successfully")
    else:
        missing = set(recuima_vars) - set(available_recuima)
        logger.warning(f"⚠️ Cannot calculate RECUIMA score. Missing variables: {missing}")
    
    return df


def process_dataset(
    input_path: Path,
    output_path: Optional[Path] = None,
    save_formats: List[str] = ['xlsx', 'csv', 'parquet']
) -> pd.DataFrame:
    """
    Process the RECUIMA dataset to extract complications variables.
    
    Args:
        input_path: Path to input Excel/CSV file
        output_path: Base path for output (without extension)
        save_formats: List of formats to save ('xlsx', 'csv', 'parquet')
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"📂 Loading dataset from: {input_path}")
    
    # Load dataset
    if input_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == '.csv':
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
    elif input_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    logger.info(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for complicaciones column
    if 'complicaciones' not in df.columns:
        raise ValueError("Column 'complicaciones' not found in dataset")
    
    # Parse complicaciones column
    logger.info("🔄 Parsing 'complicaciones' column...")
    
    parsed_results = df['complicaciones'].apply(parse_complicaciones_string)
    parsed_df = pd.DataFrame(parsed_results.tolist())
    
    # Merge with original dataframe
    df = pd.concat([df, parsed_df], axis=1)
    
    # Log parsing statistics
    logger.info("📈 Parsing statistics:")
    for col in parsed_df.columns:
        count = parsed_df[col].sum()
        pct = count / len(df) * 100
        logger.info(f"   {col}: {count} ({pct:.1f}%)")
    
    # Create RECUIMA-specific derived variables
    logger.info("🧮 Creating RECUIMA derived variables...")
    df = create_recuima_derived_variables(df)
    
    # Save processed dataset
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in save_formats:
            if fmt == 'xlsx':
                out_file = output_path.with_suffix('.xlsx')
                df.to_excel(out_file, index=False)
                logger.info(f"💾 Saved: {out_file}")
            elif fmt == 'csv':
                out_file = output_path.with_suffix('.csv')
                df.to_csv(out_file, index=False, encoding='utf-8')
                logger.info(f"💾 Saved: {out_file}")
            elif fmt == 'parquet':
                out_file = output_path.with_suffix('.parquet')
                df.to_parquet(out_file, index=False)
                logger.info(f"💾 Saved: {out_file}")
    
    return df


def generate_report(df: pd.DataFrame) -> str:
    """Generate a summary report of the parsed complications."""
    
    report_lines = [
        "=" * 70,
        "REPORTE DE PARSEO DE COMPLICACIONES PARA ESCALAS CLÍNICAS",
        "=" * 70,
        "",
        f"Total de registros procesados: {len(df)}",
        f"Registros con complicaciones: {df['complicaciones'].notna().sum()}",
        "",
        "=" * 70,
        "VARIABLES PARA ESCALA RECUIMA",
        "=" * 70,
        ""
    ]
    
    # RECUIMA variables
    recuima_vars = {
        'comp_fv_tv': 'FV/TV (Fibrilación/Taquicardia Ventricular) - 2 pts',
        'comp_bav_alto_grado': 'BAV de Alto Grado (2do Mobitz II, 3er grado) - 1 pt',
        'recuima_edad_gt70': 'Edad > 70 años - 1 pt',
        'recuima_tas_lt100': 'TAS < 100 mmHg - 1 pt',
        'recuima_tfg_lt60': 'TFG < 60 ml/min/1.73m² - 3 pts',
        'recuima_ecg_gt7': '> 7 derivaciones ECG afectadas - 1 pt',
        'recuima_killip_iv': 'Killip-Kimball IV - 1 pt',
    }
    
    for var, desc in recuima_vars.items():
        if var in df.columns:
            count = df[var].sum()
            pct = count / len(df) * 100
            report_lines.append(f"  {desc}")
            report_lines.append(f"    → {count} casos ({pct:.1f}%)")
            report_lines.append("")
    
    # RECUIMA score distribution
    if 'recuima_score_calculated' in df.columns:
        report_lines.extend([
            "",
            "DISTRIBUCIÓN DEL SCORE RECUIMA CALCULADO:",
            "-" * 40,
        ])
        score_dist = df['recuima_score_calculated'].value_counts().sort_index()
        for score, count in score_dist.items():
            pct = count / len(df) * 100
            risk = "ALTO" if score >= 4 else "Bajo"
            report_lines.append(f"  Score {score}: {count} ({pct:.1f}%) - Riesgo {risk}")
        
        # Risk categories
        if 'recuima_risk_category' in df.columns:
            report_lines.extend([
                "",
                "CATEGORÍAS DE RIESGO RECUIMA:",
                "-" * 40,
            ])
            for cat in ['bajo', 'alto']:
                count = (df['recuima_risk_category'] == cat).sum()
                pct = count / len(df) * 100
                report_lines.append(f"  Riesgo {cat.upper()}: {count} ({pct:.1f}%)")
    
    # Other complications
    report_lines.extend([
        "",
        "=" * 70,
        "OTRAS COMPLICACIONES PARSEADAS",
        "=" * 70,
        ""
    ])
    
    other_vars = {
        'comp_fv': 'Fibrilación Ventricular (FV)',
        'comp_tv': 'Taquicardia Ventricular (TV)',
        'comp_shock': 'Shock Cardiogénico',
        'comp_fallo_bomba': 'Fallo de Bomba / IC',
        'comp_pcr': 'Paro Cardiorrespiratorio (PCR)',
        'comp_otras_arritmias': 'Otras Arritmias',
        'comp_mecanicas': 'Complicaciones Mecánicas',
        'comp_hemorragia': 'Hemorragia',
        'comp_embolia_pulmonar': 'Embolia Pulmonar',
        'comp_reinfarto': 'Reinfarto',
        'comp_isquemia_recurrente': 'Isquemia Recurrente',
        'comp_infarto_vd': 'Infarto de Ventrículo Derecho',
    }
    
    for var, desc in other_vars.items():
        if var in df.columns:
            count = df[var].sum()
            pct = count / len(df) * 100
            report_lines.append(f"  {desc}: {count} ({pct:.1f}%)")
    
    return "\n".join(report_lines)


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "DATA" / "recuima-020425.xlsx"
    output_path = project_root / "DATA" / "recuima-020425-with-parsed-complications"
    
    # Process dataset
    df = process_dataset(
        input_path=input_path,
        output_path=output_path,
        save_formats=['xlsx', 'csv', 'parquet']
    )
    
    # Generate and print report
    report = generate_report(df)
    print(report)
    
    # Save report
    report_path = project_root / "DATA" / "complicaciones_parsing_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"📄 Report saved to: {report_path}")
    
    return df


if __name__ == "__main__":
    main()
