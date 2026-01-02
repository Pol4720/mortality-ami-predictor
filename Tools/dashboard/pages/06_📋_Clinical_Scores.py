"""Clinical Scores Calculator page."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from app import initialize_state
from src.scoring import get_score, list_scores

# Initialize
initialize_state()

# Page config
st.title("📋 Clinical Scores Calculator")
st.markdown("---")

st.warning("""
⚠️ **Educational Purpose Only**

These are approximate implementations for research and educational purposes.
**NOT for clinical use.** Always consult appropriate medical guidelines.
""")

st.markdown("---")

# Get available scores
available_scores = list_scores()

if not available_scores:
    st.error("❌ No clinical scores available")
    st.stop()

# Score selection
score_options = {
    "grace": "GRACE Score (Global Registry of Acute Coronary Events)",
    "timi": "TIMI Score (Thrombolysis In Myocardial Infarction)",
    "recuima": "RECUIMA Score (Registro Cubano de Infarto de Miocardio Agudo)",
}

# Filter to available scores
available_options = {k: v for k, v in score_options.items() if k in available_scores}

if not available_options:
    st.error("❌ No supported clinical scores available")
    st.stop()

selected_score = st.selectbox(
    "Select Clinical Score",
    list(available_options.keys()),
    format_func=lambda k: available_options[k],
    help="Choose a clinical score to calculate"
)

st.markdown("---")

# GRACE Score Calculator
if selected_score == "grace":
    st.subheader("GRACE Score Calculator")
    
    st.markdown("""
    The GRACE score estimates in-hospital and 6-month mortality for patients with acute coronary syndrome.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Patient Demographics & Vitals")
        
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=65,
            help="Patient age in years"
        )
        
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=0,
            max_value=250,
            value=80,
            help="Beats per minute"
        )
        
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            help="Systolic BP in mmHg"
        )
    
    with col2:
        st.markdown("#### Clinical Parameters")
        
        creatinine = st.number_input(
            "Creatinine (mg/dL)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Serum creatinine level"
        )
        
        killip_class = st.selectbox(
            "Killip Class",
            ["I", "II", "III", "IV"],
            index=0,
            help="Heart failure classification"
        )
        
        st.markdown("#### ECG & Biomarkers")
        
        st_deviation = st.checkbox(
            "ST Segment Deviation",
            value=False,
            help="ST elevation or depression on ECG"
        )
        
        elevated_enzymes = st.checkbox(
            "Elevated Cardiac Enzymes",
            value=False,
            help="Elevated troponin or CK-MB"
        )
        
        cardiac_arrest = st.checkbox(
            "Cardiac Arrest at Admission",
            value=False,
            help="Cardiac arrest on presentation"
        )
    
    st.markdown("---")
    
    if st.button("🧮 Calculate GRACE Score", type="primary", use_container_width=True):
        try:
            # Get GRACE scorer
            grace_scorer = get_score("grace")
            
            # Map Killip class
            killip_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
            
            # Compute score
            result = grace_scorer.compute(
                age=age,
                heart_rate=heart_rate,
                systolic_bp=systolic_bp,
                creatinine=creatinine,
                killip_class=killip_map[killip_class],
                st_deviation=st_deviation,
                elevated_enzymes=elevated_enzymes,
                cardiac_arrest=cardiac_arrest,
            )
            
            st.success("✅ GRACE Score Calculated")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "GRACE Score",
                    f"{result['score']:.1f}",
                    help="Total GRACE risk score"
                )
            
            with col2:
                risk_cat = result['risk_category'].capitalize()
                risk_color = {
                    "Low": "🟢",
                    "Intermediate": "🟡",
                    "High": "🔴"
                }.get(risk_cat, "⚪")
                
                st.metric(
                    "Risk Category",
                    f"{risk_color} {risk_cat}",
                    help="Risk stratification"
                )
            
            # Additional information
            with st.expander("📊 Risk Interpretation"):
                st.markdown("""
                **GRACE Score Risk Categories:**
                - **Low Risk (≤108)**: <1% in-hospital mortality
                - **Intermediate Risk (109-140)**: 1-3% in-hospital mortality
                - **High Risk (>140)**: >3% in-hospital mortality
                
                Higher scores indicate greater risk of adverse outcomes including death and MI.
                """)
        
        except Exception as e:
            st.error(f"❌ Error calculating GRACE score: {e}")
            st.exception(e)

# TIMI Score Calculator
elif selected_score == "timi":
    st.subheader("TIMI Score Calculator")
    
    st.info("""
    The TIMI score estimates risk in patients with unstable angina and non-ST elevation MI.
    **This implementation is a placeholder - full TIMI calculator coming soon.**
    """)
    
    st.markdown("""
    **TIMI Risk Score Components:**
    1. Age ≥65 years
    2. ≥3 CAD risk factors
    3. Known CAD (stenosis ≥50%)
    4. ASA use in past 7 days
    5. Severe angina (≥2 episodes in 24h)
    6. ST deviation ≥0.5mm
    7. Elevated cardiac biomarkers
    
    **Score ranges from 0-7 points**
    """)

# RECUIMA Score Calculator
elif selected_score == "recuima":
    st.subheader("🇨🇺 Escala RECUIMA")
    
    st.markdown("""
    **Escala predictiva de muerte hospitalaria por infarto agudo de miocardio**
    
    Desarrollada por el **Dr. Maikel Santos Medina** a partir del Registro Cubano de Infarto (RECUIMA).
    
    *Universidad de Ciencias Médicas de Santiago de Cuba - Hospital General Docente "Dr. Ernesto Guevara de la Serna", Las Tunas*
    """)
    
    st.info("""
    📖 Esta escala fue construida y validada en población cubana, utilizando datos del 
    Registro Cubano de Infarto de Miocardio Agudo (RECUIMA) entre enero 2018 y diciembre 2020.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Datos Demográficos y Hemodinámicos")
        
        age_recuima = st.number_input(
            "Edad (años)",
            min_value=0,
            max_value=120,
            value=65,
            help="Edad del paciente en años",
            key="recuima_age"
        )
        
        systolic_bp_recuima = st.number_input(
            "Tensión Arterial Sistólica (mmHg)",
            min_value=30,
            max_value=250,
            value=120,
            help="TAS en mmHg. Factor de riesgo si < 100 mmHg",
            key="recuima_sbp"
        )
        
        gfr_recuima = st.number_input(
            "Filtrado Glomerular (ml/min/1.73m²)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
            step=1.0,
            help="FGR estimado. Factor de riesgo si < 60 ml/min/1.73m²",
            key="recuima_gfr"
        )
        
        st.markdown("#### Hallazgos Electrocardiográficos")
        
        ecg_leads = st.number_input(
            "Derivaciones ECG Afectadas",
            min_value=0,
            max_value=12,
            value=2,
            help="Número de derivaciones con cambios del ST (0-12). Factor de riesgo si > 7",
            key="recuima_ecg"
        )
    
    with col2:
        st.markdown("#### Clasificación Clínica")
        
        killip_class_recuima = st.selectbox(
            "Clase Killip-Kimball",
            ["I - Sin insuficiencia cardíaca", 
             "II - Estertores, S3, congestión venosa",
             "III - Edema pulmonar franco",
             "IV - Shock cardiogénico"],
            index=0,
            help="Clasificación de insuficiencia cardíaca. Factor de riesgo si Killip IV",
            key="recuima_killip"
        )
        
        st.markdown("#### Arritmias y Trastornos de Conducción")
        
        vf_vt_recuima = st.checkbox(
            "Fibrilación Ventricular / Taquicardia Ventricular",
            value=False,
            help="Presencia de FV o TV durante la hospitalización",
            key="recuima_vfvt"
        )
        
        avb_recuima = st.checkbox(
            "Bloqueo AV de Alto Grado",
            value=False,
            help="BAV de segundo grado Mobitz II o BAV completo (tercer grado)",
            key="recuima_avb"
        )
    
    st.markdown("---")
    
    # Show current risk factors
    with st.expander("📋 Ver factores de riesgo actuales", expanded=True):
        factors = []
        if age_recuima > 70:
            factors.append("✓ Edad > 70 años")
        if systolic_bp_recuima < 100:
            factors.append("✓ TAS < 100 mmHg")
        if gfr_recuima < 60:
            factors.append("✓ Filtrado glomerular < 60 ml/min/1.73m²")
        if ecg_leads > 7:
            factors.append("✓ > 7 derivaciones ECG afectadas")
        if "IV" in killip_class_recuima:
            factors.append("✓ Killip-Kimball IV (Shock cardiogénico)")
        if vf_vt_recuima:
            factors.append("✓ Fibrilación/Taquicardia ventricular")
        if avb_recuima:
            factors.append("✓ Bloqueo AV de alto grado")
        
        if factors:
            st.warning(f"**Factores de riesgo presentes ({len(factors)}/7):**\n\n" + "\n".join(factors))
        else:
            st.success("**No hay factores de riesgo presentes**")
    
    if st.button("🧮 Calcular Escala RECUIMA", type="primary", use_container_width=True):
        try:
            # Get RECUIMA scorer
            recuima_scorer = get_score("recuima")
            
            # Map Killip class
            killip_value = 4 if "IV" in killip_class_recuima else (
                3 if "III" in killip_class_recuima else (
                    2 if "II" in killip_class_recuima else 1
                )
            )
            
            # Compute score
            result = recuima_scorer.compute(
                age=age_recuima,
                systolic_bp=systolic_bp_recuima,
                gfr=gfr_recuima,
                ecg_leads_affected=ecg_leads,
                killip_class=killip_value,
                vf_vt=vf_vt_recuima,
                high_grade_avb=avb_recuima,
            )
            
            st.success("✅ Escala RECUIMA Calculada")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Puntuación RECUIMA",
                    f"{result['score']}/7",
                    help="Puntuación total de la escala RECUIMA"
                )
            
            with col2:
                risk_cat = "Alto" if result['risk_category'] == "high" else "Bajo"
                risk_color = "🔴" if result['risk_category'] == "high" else "🟢"
                
                st.metric(
                    "Categoría de Riesgo",
                    f"{risk_color} {risk_cat}",
                    help="Estratificación del riesgo de muerte hospitalaria"
                )
            
            with col3:
                st.metric(
                    "Probabilidad Estimada",
                    f"{result['probability']*100:.1f}%",
                    help="Probabilidad estimada de muerte hospitalaria"
                )
            
            # Component breakdown
            st.markdown("---")
            st.markdown("#### 📊 Desglose de Componentes")
            
            component_names = {
                "age_gt_70": "Edad > 70 años",
                "sbp_lt_100": "TAS < 100 mmHg",
                "gfr_lt_60": "FGR < 60 ml/min/1.73m²",
                "ecg_leads_gt_7": "> 7 derivaciones ECG",
                "killip_iv": "Killip-Kimball IV",
                "vf_vt": "FV/TV",
                "high_grade_avb": "BAV alto grado",
            }
            
            cols = st.columns(4)
            for idx, (key, value) in enumerate(result['components'].items()):
                with cols[idx % 4]:
                    icon = "✅" if value == 1 else "❌"
                    st.markdown(f"{icon} **{component_names.get(key, key)}**")
            
            # Risk interpretation
            with st.expander("📖 Interpretación del Riesgo"):
                st.markdown(f"""
                ### Resultado: Riesgo {risk_cat}
                
                **Puntuación obtenida:** {result['score']} de 7 puntos posibles
                
                **Categorías de Riesgo RECUIMA:**
                - **Riesgo Bajo (< 3 puntos):** Menor probabilidad de muerte hospitalaria
                - **Riesgo Alto (≥ 3 puntos):** Mayor probabilidad de muerte hospitalaria, requiere monitorización intensiva
                
                **Recomendaciones según nivel de riesgo:**
                """)
                
                if result['risk_category'] == "high":
                    st.error("""
                    ⚠️ **Paciente de Alto Riesgo**
                    - Considerar ingreso en Unidad de Cuidados Intensivos Coronarios
                    - Monitorización hemodinámica continua
                    - Evaluación urgente para estrategia de reperfusión
                    - Vigilancia estrecha de arritmias
                    """)
                else:
                    st.success("""
                    ✅ **Paciente de Bajo Riesgo**
                    - Monitorización estándar en unidad coronaria
                    - Seguimiento según protocolo institucional
                    - Evaluación de estrategia de reperfusión según indicación
                    """)
        
        except Exception as e:
            st.error(f"❌ Error al calcular la escala RECUIMA: {e}")
            st.exception(e)

# Score comparison
st.markdown("---")

with st.expander("📚 About Clinical Scores"):
    st.markdown("""
    ### Clinical Risk Scores in ACS
    
    **GRACE Score:**
    - Validated for in-hospital and 6-month mortality
    - Includes age, heart rate, BP, creatinine, Killip class, cardiac arrest, ST changes, and cardiac biomarkers
    - Widely used in international guidelines
    
    **TIMI Score:**
    - Developed for UA/NSTEMI risk stratification
    - Simpler, point-based system (0-7 points)
    - Predicts 14-day risk of death, MI, or urgent revascularization
    
    **RECUIMA Score (🇨🇺 Cuban Registry):**
    - Developed and validated in Cuban population
    - Specifically designed for in-hospital mortality prediction in AMI
    - Uses 7 easily obtainable clinical variables
    - Two risk categories: Low and High
    - Variables:
        1. Age > 70 years
        2. Systolic BP < 100 mmHg
        3. GFR < 60 ml/min/1.73m²
        4. > 7 ECG leads affected
        5. Killip-Kimball class IV
        6. Ventricular fibrillation/tachycardia
        7. High-grade AV block
    
    **Important Notes:**
    - These scores are tools to support clinical decision-making
    - Should not replace clinical judgment
    - Local validation may be necessary
    - Use in conjunction with other clinical information
    
    ### References
    - GRACE: Fox KA, et al. BMJ 2006
    - TIMI: Antman EM, et al. JAMA 2000
    - RECUIMA: Santos Medina M. Tesis Doctoral, Universidad de Ciencias Médicas de Santiago de Cuba, 2020
    """)

