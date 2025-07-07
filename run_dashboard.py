#!/usr/bin/env python3
"""Launch the AI Token Factory Dashboard."""

import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from data.simulation_data import (
    create_sample_neoclouds,
    create_sample_inference_providers,
    create_sample_applications,
    load_market_data
)
from analytics.dashboard import create_dashboard

def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="AI Token Factory Economics Stack",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load simulation data
    with st.spinner("Loading simulation data..."):
        neoclouds = create_sample_neoclouds()
        inference_providers = create_sample_inference_providers()
        applications = create_sample_applications()
        market_data = load_market_data()
    
    # Create dashboard
    create_dashboard(neoclouds, inference_providers, applications, market_data)

if __name__ == "__main__":
    main()