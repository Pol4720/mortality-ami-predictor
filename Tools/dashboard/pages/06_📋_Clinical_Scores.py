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
    
    if st.button("🧮 Calculate GRACE Score", type="primary", width='stretch'):
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
    
    **Important Notes:**
    - These scores are tools to support clinical decision-making
    - Should not replace clinical judgment
    - Local validation may be necessary
    - Use in conjunction with other clinical information
    
    ### References
    - GRACE: Fox KA, et al. BMJ 2006
    - TIMI: Antman EM, et al. JAMA 2000
    """)

