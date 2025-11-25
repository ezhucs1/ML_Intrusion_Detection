"""
Interactive Web Demo for Intrusion Detection System (Binary Model)
Redesigned for clear presentation and demonstration
Run with: streamlit run src/web_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.test_model import ModelTester
from src.data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Intrusion Detection System - Binary Model Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .attack-detected {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 4px solid #c62828;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .normal-traffic {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 4px solid #2e7d32;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-high {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_binary_model():
    """Load binary model (cached)"""
    try:
        models_dir = 'models_size_100000'
        tester = ModelTester(models_dir=models_dir, classification_mode='binary')
        tester.load_model()
        return tester
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("⚠️ Please train the binary model first by running: `python src/train_model.py --binary --models-dir models_size_100000 --n-samples 100000`")
        return None

def get_feature_importance_analysis(tester, features_dict):
    """Get feature importance and contribution analysis"""
    if tester is None:
        return None
    
    # Use RandomForest model (best model for feature importance)
    if tester.all_models and 'RandomForest' in tester.all_models:
        model_to_use = tester.all_models['RandomForest']
    else:
        model_to_use = tester.model
    
    if not hasattr(model_to_use, 'feature_importances_'):
        return None
    
    # Get feature importance scores
    importances = model_to_use.feature_importances_
    
    # Convert dict to ordered array
    feature_array = np.array([features_dict.get(fname, 0) for fname in tester.feature_names])
    
    # Scale features
    scaled_features = tester.scaler.transform([feature_array])
    
    # Calculate contribution: importance * scaled_feature_value
    contributions = importances * scaled_features[0]
    
    # Create DataFrame
    feature_contrib = pd.DataFrame({
        'Feature': tester.feature_names,
        'Importance': importances,
        'Scaled_Value': scaled_features[0],
        'Contribution': contributions
    }).sort_values('Contribution', key=abs, ascending=False)
    
    return feature_contrib

def predict_single_sample(tester, features_dict):
    """Make prediction on a single sample"""
    if tester is None:
        return None, None, None
    
    # Use RandomForest model (best model)
    if tester.all_models and 'RandomForest' in tester.all_models:
        model_to_use = tester.all_models['RandomForest']
    else:
        model_to_use = tester.model
    
    # Convert dict to ordered array
    feature_array = np.array([features_dict.get(fname, 0) for fname in tester.feature_names])
    
    # Scale features
    scaled_features = tester.scaler.transform([feature_array])
    
    # Predict
    prediction_encoded = model_to_use.predict(scaled_features)[0]
    probability = model_to_use.predict_proba(scaled_features)[0]
    
    # Binary mode: 0=Benign, 1=Attack
    prediction_label = 'Attack' if prediction_encoded == 1 else 'Benign'
    confidence = probability[prediction_encoded] * 100
    
    return prediction_label, confidence, probability

def process_csv_file(uploaded_file, tester, max_samples=1000):
    """Process uploaded CSV file and return predictions"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file, nrows=max_samples)
        
        # Initialize processor for test data
        processor = DataProcessor()
        processor.classification_mode = 'binary'
        processor.feature_columns = tester.feature_names  # Set feature columns from trained model
        
        # Preprocess test data (fit_encoder=False for testing)
        X, y = processor.preprocess_data(df, classification_mode='binary')
        
        # Use tester's scaler (from trained model)
        X_scaled = tester.scaler.transform(X)
        
        # Get predictions using RandomForest (best model)
        model_to_use = tester.all_models['RandomForest'] if tester.all_models and 'RandomForest' in tester.all_models else tester.model
        
        predictions = model_to_use.predict(X_scaled)
        probabilities = model_to_use.predict_proba(X_scaled)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in predictions]
        results_df['Attack_Confidence'] = [prob[1] * 100 for prob in probabilities]
        results_df['Benign_Confidence'] = [prob[0] * 100 for prob in probabilities]
        
        # Add actual label if available
        if 'Label' in df.columns:
            results_df['Actual_Label'] = df['Label'].str.strip().str.upper()
            results_df['Is_Correct'] = (
                (results_df['Prediction'] == 'Attack') & (results_df['Actual_Label'] != 'BENIGN')
            ) | (
                (results_df['Prediction'] == 'Benign') & (results_df['Actual_Label'] == 'BENIGN')
            )
        
        # Store processed features for later use
        feature_data = {}
        for idx, row_idx in enumerate(df.index):
            sample_features = {}
            for feat_name in tester.feature_names:
                if feat_name in X.columns:
                    sample_features[feat_name] = float(X.iloc[idx][feat_name])
                else:
                    sample_features[feat_name] = 0.0
            feature_data[row_idx] = sample_features
        
        return results_df, feature_data, y if 'Label' in df.columns else None
        
    except Exception as e:
        import traceback
        st.error(f"Error processing CSV: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Binary Classification Model - Interactive Demo</div>', unsafe_allow_html=True)
    
    # Load model
    tester = load_binary_model()
    
    if tester is None:
        st.stop()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.success(f"✅ **Model**: {tester.best_model_name}")
        st.info(f"📈 **Mode**: Binary Classification")
        st.info(f"🔢 **Features**: {len(tester.feature_names)}")
        
        st.markdown("---")
        st.header("🎯 Demo Modes")
        demo_mode = st.radio(
            "Choose demo mode:",
            ["📁 Upload CSV File", "🔍 Explore Single Sample", "📊 Compare Samples"],
            index=0
        )
    
    # Main content based on demo mode
    if demo_mode == "📁 Upload CSV File":
        st.header("📁 Upload Test Data (2018 CSV Files)")
        st.markdown("Upload a CSV file from your 2018 test data to see predictions and analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file from Testing_data/ directory")
        
        if uploaded_file is not None:
            with st.spinner("Processing CSV file..."):
                results_df, feature_data, y_actual = process_csv_file(uploaded_file, tester, max_samples=1000)
            
            if results_df is not None and feature_data is not None:
                st.success(f"✅ Processed {len(results_df)} samples")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_attacks = (results_df['Prediction'] == 'Attack').sum()
                    st.metric("🚨 Attacks Detected", total_attacks)
                with col2:
                    total_benign = (results_df['Prediction'] == 'Benign').sum()
                    st.metric("✅ Benign Traffic", total_benign)
                with col3:
                    avg_confidence = results_df['Attack_Confidence'].mean()
                    st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")
                with col4:
                    if y_actual is not None:
                        accuracy = results_df['Is_Correct'].mean() * 100 if 'Is_Correct' in results_df.columns else None
                        if accuracy is not None:
                            st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                    else:
                        st.metric("📈 Samples", len(results_df))
                
                # Sample selector
                st.markdown("---")
                st.subheader("🔍 Explore Individual Samples")
                
                sample_idx = st.slider("Select sample to analyze", 0, len(results_df)-1, 0, key="csv_sample_slider")
                selected_sample = results_df.iloc[sample_idx]
                
                # Prediction display
                col_pred1, col_pred2 = st.columns([2, 1])
                
                with col_pred1:
                    if selected_sample['Prediction'] == 'Attack':
                        st.markdown(
                            f'<div class="attack-detected">🚨 ATTACK DETECTED</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="normal-traffic">✅ NORMAL TRAFFIC</div>',
                            unsafe_allow_html=True
                        )
                
                with col_pred2:
                    st.metric("Attack Confidence", f"{selected_sample['Attack_Confidence']:.1f}%")
                    st.metric("Benign Confidence", f"{selected_sample['Benign_Confidence']:.1f}%")
                    if 'Actual_Label' in selected_sample:
                        actual = selected_sample['Actual_Label']
                        is_correct = selected_sample['Is_Correct']
                        st.metric("Actual Label", actual)
                        if is_correct:
                            st.success("✅ Correct Prediction")
                        else:
                            st.error("❌ Incorrect Prediction")
                
                # Feature importance for this sample
                st.markdown("---")
                st.subheader("💡 Why This Prediction? (Feature Importance Analysis)")
                
                # Extract features for this sample (use preprocessed feature data)
                sample_idx = selected_sample.name
                if sample_idx in feature_data:
                    sample_features = feature_data[sample_idx]
                else:
                    # Fallback: try to extract from dataframe
                    sample_features = {}
                    for feat_name in tester.feature_names:
                        if feat_name in results_df.columns:
                            sample_features[feat_name] = float(selected_sample[feat_name])
                        else:
                            sample_features[feat_name] = 0.0
                
                # Get feature importance
                feature_contrib = get_feature_importance_analysis(tester, sample_features)
                
                if feature_contrib is not None:
                    top_contrib = feature_contrib.head(10)
                    
                    col_chart, col_table = st.columns([2, 1])
                    
                    with col_chart:
                        try:
                            import plotly.express as px
                            
                            contrib_plot = top_contrib.copy()
                            contrib_plot['Contribution_Abs'] = contrib_plot['Contribution'].abs()
                            contrib_plot['Direction'] = contrib_plot['Contribution'].apply(
                                lambda x: 'Attack Indicator' if x > 0 else 'Normal Indicator'
                            )
                            contrib_plot = contrib_plot.sort_values('Contribution_Abs', ascending=True)
                            
                            fig = px.bar(
                                contrib_plot,
                                x='Contribution',
                                y='Feature',
                                orientation='h',
                                color='Direction',
                                color_discrete_map={'Attack Indicator': '#FF4444', 'Normal Indicator': '#44FF44'},
                                title='Top 10 Features Contributing to This Prediction',
                                labels={'Contribution': 'Contribution Score', 'Feature': 'Feature Name'}
                            )
                            fig.update_layout(height=400, showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.info("⚠️ Plotly not available. Install with: pip install plotly")
                    
                    with col_table:
                        st.markdown("**Top Contributing Features**")
                        display_table = top_contrib[['Feature', 'Contribution']].copy()
                        display_table['Contribution'] = display_table['Contribution'].apply(lambda x: f"{x:.4f}")
                        display_table['Impact'] = display_table['Contribution'].apply(
                            lambda x: "🔴 Attack" if float(x) > 0 else "🟢 Normal"
                        )
                        st.dataframe(display_table[['Feature', 'Contribution', 'Impact']], use_container_width=True, height=400)
                
                # Show sample data
                with st.expander("📋 View Sample Data", expanded=False):
                    st.dataframe(selected_sample, use_container_width=True)
                
                # Overall statistics
                st.markdown("---")
                st.subheader("📊 Overall Statistics")
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.markdown("**Prediction Distribution**")
                    pred_counts = results_df['Prediction'].value_counts()
                    try:
                        import plotly.express as px
                        fig_pie = px.pie(
                            values=pred_counts.values,
                            names=pred_counts.index,
                            title="Predictions Distribution",
                            color_discrete_map={'Attack': '#FF4444', 'Benign': '#44FF44'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    except ImportError:
                        st.bar_chart(pred_counts)
                
                with col_stat2:
                    st.markdown("**Confidence Distribution**")
                    try:
                        import plotly.express as px
                        fig_hist = px.histogram(
                            results_df,
                            x='Attack_Confidence',
                            nbins=20,
                            title="Attack Confidence Distribution",
                            labels={'Attack_Confidence': 'Attack Confidence (%)', 'count': 'Number of Samples'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except ImportError:
                        st.bar_chart(results_df['Attack_Confidence'].value_counts().head(10))
    
    elif demo_mode == "🔍 Explore Single Sample":
        st.header("🔍 Explore Single Sample")
        st.markdown("Manually input or adjust network flow features to see how the model responds.")
        
        # Feature input interface
        st.subheader("📝 Network Flow Features")
        
        # Key features with sliders
        col1, col2 = st.columns(2)
        
        with col1:
            syn_count = st.slider("SYN Flag Count", 0, 1000, 10, help="High SYN count indicates port scanning")
            flow_packets_ps = st.slider("Flow Packets/s", 0, 100000, 10000, help="High packet rate may indicate DDoS")
            flow_duration = st.slider("Flow Duration (μs)", 0, 10000000, 1000000, help="Short duration may indicate scanning")
            total_fwd_packets = st.slider("Total Forward Packets", 0, 5000, 100)
        
        with col2:
            rst_count = st.slider("RST Flag Count", 0, 100, 0, help="High RST count indicates connection issues")
            flow_bytes_ps = st.slider("Flow Bytes/s", 0, 100000000, 1000000, help="High byte rate indicates heavy traffic")
            down_up_ratio = st.slider("Down/Up Ratio", 0.01, 10.0, 0.67, help="Asymmetric traffic may indicate attacks")
            total_bwd_packets = st.slider("Total Backward Packets", 0, 5000, 50)
        
        # Build feature dictionary
        features = {}
        for feat_name in tester.feature_names:
            if feat_name == 'SYN Flag Count':
                features[feat_name] = syn_count
            elif feat_name == 'RST Flag Count':
                features[feat_name] = rst_count
            elif feat_name == 'Flow Packets/s':
                features[feat_name] = flow_packets_ps
            elif feat_name == 'Flow Bytes/s':
                features[feat_name] = flow_bytes_ps
            elif feat_name == 'Flow Duration':
                features[feat_name] = flow_duration
            elif feat_name == 'Down/Up Ratio':
                features[feat_name] = down_up_ratio
            elif feat_name == 'Total Fwd Packets':
                features[feat_name] = total_fwd_packets
            elif feat_name == 'Total Backward Packets':
                features[feat_name] = total_bwd_packets
            else:
                features[feat_name] = 0.0
        
        # Make prediction
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                prediction, confidence, probability = predict_single_sample(tester, features)
            
            # Display prediction
            col_pred1, col_pred2 = st.columns([2, 1])
            
            with col_pred1:
                if prediction == 'Attack':
                    st.markdown(
                        f'<div class="attack-detected">🚨 ATTACK DETECTED</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="normal-traffic">✅ NORMAL TRAFFIC</div>',
                        unsafe_allow_html=True
                    )
            
            with col_pred2:
                st.metric("Attack Confidence", f"{probability[1]*100:.1f}%")
                st.metric("Benign Confidence", f"{probability[0]*100:.1f}%")
            
            # Feature importance
            st.markdown("---")
            st.subheader("💡 Why This Prediction? (Feature Importance)")
            
            feature_contrib = get_feature_importance_analysis(tester, features)
            
            if feature_contrib is not None:
                top_contrib = feature_contrib.head(10)
                
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    try:
                        import plotly.express as px
                        
                        contrib_plot = top_contrib.copy()
                        contrib_plot['Contribution_Abs'] = contrib_plot['Contribution'].abs()
                        contrib_plot['Direction'] = contrib_plot['Contribution'].apply(
                            lambda x: 'Attack Indicator' if x > 0 else 'Normal Indicator'
                        )
                        contrib_plot = contrib_plot.sort_values('Contribution_Abs', ascending=True)
                        
                        fig = px.bar(
                            contrib_plot,
                            x='Contribution',
                            y='Feature',
                            orientation='h',
                            color='Direction',
                            color_discrete_map={'Attack Indicator': '#FF4444', 'Normal Indicator': '#44FF44'},
                            title='Top 10 Features Contributing to This Prediction',
                            labels={'Contribution': 'Contribution Score', 'Feature': 'Feature Name'}
                        )
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.info("⚠️ Plotly not available")
                
                with col_table:
                    st.markdown("**Top Contributing Features**")
                    display_table = top_contrib[['Feature', 'Contribution']].copy()
                    display_table['Contribution'] = display_table['Contribution'].apply(lambda x: f"{x:.4f}")
                    display_table['Impact'] = display_table['Contribution'].apply(
                        lambda x: "🔴 Attack" if float(x) > 0 else "🟢 Normal"
                    )
                    st.dataframe(display_table[['Feature', 'Contribution', 'Impact']], use_container_width=True, height=400)
    
    elif demo_mode == "📊 Compare Samples":
        st.header("📊 Compare Normal vs Attack Samples")
        st.markdown("Compare how the model analyzes normal traffic vs attack traffic.")
        
        # Sample presets
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            st.subheader("✅ Normal Traffic Sample")
            normal_features = {
                'SYN Flag Count': 5,
                'RST Flag Count': 0,
                'Flow Packets/s': 5000,
                'Flow Bytes/s': 500000,
                'Flow Duration': 2000000,
                'Down/Up Ratio': 0.67,
                'Total Fwd Packets': 50,
                'Total Backward Packets': 30
            }
            
            # Fill remaining features
            for feat_name in tester.feature_names:
                if feat_name not in normal_features:
                    normal_features[feat_name] = 0.0
        
        with col_preset2:
            st.subheader("🚨 Attack Traffic Sample")
            attack_features = {
                'SYN Flag Count': 600,
                'RST Flag Count': 20,
                'Flow Packets/s': 55000,
                'Flow Bytes/s': 10000000,
                'Flow Duration': 200000,
                'Down/Up Ratio': 0.1,
                'Total Fwd Packets': 1000,
                'Total Backward Packets': 50
            }
            
            # Fill remaining features
            for feat_name in tester.feature_names:
                if feat_name not in attack_features:
                    attack_features[feat_name] = 0.0
        
        # Make predictions
        normal_pred, normal_conf, normal_prob = predict_single_sample(tester, normal_features)
        attack_pred, attack_conf, attack_prob = predict_single_sample(tester, attack_features)
        
        # Display side by side
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("### ✅ Normal Traffic")
            if normal_pred == 'Attack':
                st.markdown('<div class="attack-detected">🚨 ATTACK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="normal-traffic">✅ NORMAL TRAFFIC</div>', unsafe_allow_html=True)
            st.metric("Attack Confidence", f"{normal_prob[1]*100:.1f}%")
            st.metric("Benign Confidence", f"{normal_prob[0]*100:.1f}%")
            
            # Key features
            st.markdown("**Key Features:**")
            st.write(f"• SYN Flags: {normal_features.get('SYN Flag Count', 0)}")
            st.write(f"• Flow Packets/s: {normal_features.get('Flow Packets/s', 0):,}")
            st.write(f"• Flow Duration: {normal_features.get('Flow Duration', 0):,} μs")
        
        with col_comp2:
            st.markdown("### 🚨 Attack Traffic")
            if attack_pred == 'Attack':
                st.markdown('<div class="attack-detected">🚨 ATTACK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="normal-traffic">✅ NORMAL TRAFFIC</div>', unsafe_allow_html=True)
            st.metric("Attack Confidence", f"{attack_prob[1]*100:.1f}%")
            st.metric("Benign Confidence", f"{attack_prob[0]*100:.1f}%")
            
            # Key features
            st.markdown("**Key Features:**")
            st.write(f"• SYN Flags: {attack_features.get('SYN Flag Count', 0)}")
            st.write(f"• Flow Packets/s: {attack_features.get('Flow Packets/s', 0):,}")
            st.write(f"• Flow Duration: {attack_features.get('Flow Duration', 0):,} μs")
        
        # Feature importance comparison
        st.markdown("---")
        st.subheader("💡 Feature Importance Comparison")
        
        normal_contrib = get_feature_importance_analysis(tester, normal_features)
        attack_contrib = get_feature_importance_analysis(tester, attack_features)
        
        if normal_contrib is not None and attack_contrib is not None:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("**Normal Traffic - Top Features**")
                top_normal = normal_contrib.head(5)
                try:
                    import plotly.express as px
                    fig_normal = px.bar(
                        top_normal,
                        x='Contribution',
                        y='Feature',
                        orientation='h',
                        title="Normal Traffic Features",
                        color='Contribution',
                        color_continuous_scale=['green', 'yellow', 'red']
                    )
                    st.plotly_chart(fig_normal, use_container_width=True)
                except ImportError:
                    st.dataframe(top_normal[['Feature', 'Contribution']])
            
            with col_chart2:
                st.markdown("**Attack Traffic - Top Features**")
                top_attack = attack_contrib.head(5)
                try:
                    import plotly.express as px
                    fig_attack = px.bar(
                        top_attack,
                        x='Contribution',
                        y='Feature',
                        orientation='h',
                        title="Attack Traffic Features",
                        color='Contribution',
                        color_continuous_scale=['green', 'yellow', 'red']
                    )
                    st.plotly_chart(fig_attack, use_container_width=True)
                except ImportError:
                    st.dataframe(top_attack[['Feature', 'Contribution']])

if __name__ == "__main__":
    main()
