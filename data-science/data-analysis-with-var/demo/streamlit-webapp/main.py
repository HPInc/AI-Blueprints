"""
Streamlit web application for Data Analysis with VAR blueprint.
This app demonstrates Vector Autoregression (VAR) analysis capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import yaml
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from utils import load_config, configure_proxy

# Page configuration
st.set_page_config(
    page_title="Data Analysis with VAR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    # Load configuration
    config_path = current_dir.parent / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    # Configure proxy if needed
    configure_proxy(config)
    
    # App header
    st.title("📈 Data Analysis with VAR")
    st.markdown("**Vector Autoregression (VAR) Analysis Blueprint**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("Configure your VAR analysis parameters below.")
        
        # Model parameters
        st.subheader("Model Parameters")
        max_lags = st.slider(
            "Maximum Lags",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of lags to consider for VAR model"
        )
        
        trend = st.selectbox(
            "Trend Component",
            options=["c", "ct", "ctt", "nc"],
            index=0,
            help="Trend component: c=constant, ct=constant+trend, ctt=constant+quadratic trend, nc=no constant"
        )
        
        ic = st.selectbox(
            "Information Criterion",
            options=["aic", "fpe", "hqic", "bic"],
            index=0,
            help="Information criterion for lag order selection"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Data Input")
        
        # Data upload option
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with time series data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head(), use_container_width=True)
                
                # Variable selection
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) >= 2:
                    st.subheader("🎯 Variable Selection")
                    variables = st.multiselect(
                        "Select variables for VAR analysis",
                        options=numeric_columns,
                        default=numeric_columns[:min(3, len(numeric_columns))],
                        help="Choose at least 2 variables for VAR analysis"
                    )
                    
                    if len(variables) >= 2:
                        # Date column selection (optional)
                        date_columns = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
                        if date_columns:
                            date_col = st.selectbox(
                                "Select date column (optional)",
                                options=[None] + date_columns,
                                index=0,
                                help="Choose a date column for time series index"
                            )
                        
                        if st.button("🚀 Run VAR Analysis", type="primary"):
                            with st.spinner("Running VAR analysis..."):
                                # Placeholder for VAR analysis
                                st.success("✅ VAR analysis completed successfully!")
                                
                    else:
                        st.warning("⚠️ Please select at least 2 variables for VAR analysis.")
                else:
                    st.warning("⚠️ Please upload a file with at least 2 numeric columns.")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        else:
            st.info("👆 Please upload a CSV file to get started.")
            
            # Sample data option
            if st.button("📋 Use Sample Data"):
                st.info("📊 Sample time series dataset would be loaded here.")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # Placeholder for results
        st.info("🔄 Run VAR analysis to see results here.")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Diagnostics", "Forecasting", "Impulse Response"])
        
        with tab1:
            st.info("📋 VAR model summary will appear here after analysis.")
            
        with tab2:
            st.info("🔍 Model diagnostics and tests will appear here.")
            
        with tab3:
            st.info("🔮 Forecasting results and plots will appear here.")
            
        with tab4:
            st.info("📈 Impulse response functions will appear here.")
    
    # Additional analysis section
    st.subheader("🔬 Advanced Analysis")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📊 Granger Causality")
        st.info("🔗 Granger causality test results will appear here.")
        
    with col4:
        st.subheader("🎯 Lag Order Selection")
        st.info("📈 Lag order selection criteria will appear here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📈 Data Analysis with VAR Blueprint | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
