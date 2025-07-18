"""
Streamlit web application for Classification with SVM blueprint.
This app demonstrates Support Vector Machine classification capabilities.
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
    page_title="Classification with SVM",
    page_icon="🎯",
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
    st.title("🎯 Classification with SVM")
    st.markdown("**Support Vector Machine Classification Blueprint**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        st.markdown("Configure your SVM classification parameters below.")
        
        # Model parameters
        st.subheader("Model Parameters")
        kernel = st.selectbox(
            "Kernel Type",
            options=["rbf", "linear", "poly", "sigmoid"],
            index=0,
            help="Kernel function for SVM"
        )
        
        C = st.slider(
            "Regularization (C)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Regularization parameter"
        )
        
        gamma = st.selectbox(
            "Gamma",
            options=["scale", "auto"],
            index=0,
            help="Kernel coefficient"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Data Input")
        
        # Data upload option
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with features and target variable"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head(), use_container_width=True)
                
                # Feature selection
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) >= 2:
                    st.subheader("🎯 Feature Selection")
                    features = st.multiselect(
                        "Select feature columns",
                        options=numeric_columns,
                        default=numeric_columns[:-1] if len(numeric_columns) > 1 else numeric_columns,
                        help="Choose columns to use as features"
                    )
                    
                    target = st.selectbox(
                        "Select target column",
                        options=numeric_columns,
                        index=len(numeric_columns)-1 if len(numeric_columns) > 0 else 0,
                        help="Choose the target variable column"
                    )
                    
                    if features and target:
                        if st.button("🚀 Train SVM Model", type="primary"):
                            with st.spinner("Training SVM model..."):
                                # Placeholder for model training
                                st.success("✅ Model trained successfully!")
                                
                else:
                    st.warning("⚠️ Please upload a file with at least 2 numeric columns.")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        else:
            st.info("👆 Please upload a CSV file to get started.")
            
            # Sample data option
            if st.button("📋 Use Sample Data"):
                st.info("📊 Sample dataset would be loaded here.")
    
    with col2:
        st.subheader("📈 Results")
        
        # Placeholder for results
        st.info("🔄 Train a model to see results here.")
        
        # Visualization placeholders
        st.subheader("📊 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Decision Boundary", "Feature Analysis"])
        
        with tab1:
            st.info("🎯 Confusion matrix will appear here after training.")
            
        with tab2:
            st.info("🗺️ Decision boundary plot will appear here after training.")
            
        with tab3:
            st.info("📊 Feature analysis will appear here after training.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🎯 Classification with SVM Blueprint | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
