"""Dashboard configuration and page setup."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def configure_page():
    """Configure Streamlit page settings."""
    # Find logo if available
    assets_dir = Path(__file__).parent / "assets"
    logo = None
    
    if assets_dir.exists():
        for ext in ("logo.png", "logo.jpg", "logo.jpeg", "logo.ico"):
            logo_path = assets_dir / ext
            if logo_path.exists():
                logo = str(logo_path)
                break
        
        if logo is None:
            # Fallback: pick first image file
            imgs = list(assets_dir.glob("*.png")) + list(assets_dir.glob("*.jpg"))
            if imgs:
                logo = str(imgs[0])
    
    st.set_page_config(
        page_title="AMI Mortality/Arrhythmia Dashboard",
        page_icon=logo if logo else "🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# AMI Predictor Dashboard\n\nPredicting in-hospital mortality and ventricular arrhythmias."
        }
    )


def apply_custom_css():
    """Apply custom CSS styling to the dashboard."""
    st.markdown("""
        <style>
        /* Main content area */
        .main {
            padding: 2rem;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
        }
        
        /* Info/warning/success boxes */
        .stAlert {
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 0.5rem;
            font-weight: 500;
        }
        
        /* Dataframe styling */
        .dataframe {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
