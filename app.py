"""
E-MAS: Efficient Multi-Scale Attention System
Streamlit Web Application for Dermoscopic Skin Lesion Classification

Features:
- Inference on uploaded dermoscopic images
- Grad-CAM explainability visualization
- Model training interface
- Evaluation metrics and reports
- Model architecture information
"""

import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="E-MAS Skin Lesion Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.emas import create_emas_model, EMAS
from data.datasets import (
    HAM10000_CLASSES, HAM10000_CLASS_NAMES,
    PH2_CLASSES, PH2_CLASS_NAMES
)
from utils.preprocess import preprocess_image, denormalize_image, tensor_to_numpy
from utils.gradcam import GradCAM, visualize_gradcam, generate_multi_branch_gradcam


# ============== Session State ==============

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    if 'device' not in st.session_state:
        st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None


# ============== Model Loading ==============

def get_available_checkpoints(checkpoint_dir='checkpoints'):
    """Get list of available model checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            checkpoints.append(file)
    
    return sorted(checkpoints, reverse=True)


def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    try:
        device = st.session_state.device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        num_classes = checkpoint.get('num_classes', 7)
        dataset_type = checkpoint.get('dataset', 'ham10000')
        class_names = checkpoint.get('class_names', None)
        
        # Create model
        model = create_emas_model(num_classes=num_classes, pretrained=False, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        model_info = {
            'num_classes': num_classes,
            'dataset': dataset_type,
            'class_names': class_names,
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'val_acc': checkpoint.get('val_acc', 0.0),
            'path': checkpoint_path
        }
        
        return model, model_info
    
    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        return None, None


# ============== Inference ==============

def run_inference(model, image, device):
    """
    Run inference on a single image.
    
    Returns:
        predictions: Dictionary with class probabilities
        predicted_class: Index of predicted class
        confidence: Confidence score
    """
    # Preprocess image
    input_tensor = preprocess_image(image, input_size=224, normalize=True)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get predictions
    probs = probabilities[0].cpu().numpy()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    
    # Create predictions dictionary
    predictions = {i: float(probs[i]) for i in range(len(probs))}
    
    return predictions, predicted_class, confidence


def get_class_name(class_idx, model_info):
    """Get class name from index."""
    if model_info['class_names'] and class_idx in model_info['class_names']:
        idx_to_class = {v: k for k, v in model_info['class_names'].items()}
        class_code = idx_to_class.get(class_idx, f'Class_{class_idx}')
        
        # Get full name
        if model_info['dataset'] == 'ham10000' or model_info['dataset'] == 'combined':
            return HAM10000_CLASS_NAMES.get(class_code, class_code)
        else:
            return PH2_CLASS_NAMES.get(class_code, class_code)
    
    return f'Class {class_idx}'


def display_prediction_results(predictions, predicted_class, confidence, model_info):
    """Display prediction results in the UI."""
    
    # Get class names
    num_classes = len(predictions)
    class_names = [get_class_name(i, model_info) for i in range(num_classes)]
    
    # Create columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Prediction Result")
        
        # Main prediction
        predicted_name = get_class_name(predicted_class, model_info)
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4; margin: 0;">{predicted_name}</h2>
            <p style="font-size: 1.5rem; margin: 0;">Confidence: {confidence*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top-3 predictions
        st.markdown("### Top-3 Predictions")
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for idx, (class_idx, prob) in enumerate(sorted_preds, 1):
            class_name = get_class_name(class_idx, model_info)
            st.progress(prob, text=f"{idx}. {class_name}: {prob*100:.2f}%")
    
    with col2:
        # Probability bar chart
        st.markdown("### Class Probabilities")
        
        df = pd.DataFrame({
            'Class': class_names,
            'Probability': [predictions[i] * 100 for i in range(num_classes)]
        })
        
        fig = px.bar(
            df, x='Probability', y='Class', orientation='h',
            color='Probability', color_continuous_scale='Blues',
            text=df['Probability'].apply(lambda x: f'{x:.2f}%')
        )
        fig.update_layout(
            xaxis_title='Probability (%)',
            yaxis_title='',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ============== Grad-CAM ==============

def generate_gradcam_visualization(model, image, model_info, device):
    """Generate Grad-CAM visualization for all branches."""
    
    # Preprocess image
    input_tensor = preprocess_image(image, input_size=224, normalize=True)
    input_tensor = input_tensor.to(device)
    
    # Get original image for overlay
    orig_array = np.array(image.resize((224, 224)))
    
    gradcam_results = {}
    
    # MobileNetV2 branch
    try:
        mobilenet_layer = model.mobilenet_reduce[0]
        gradcam_mn = GradCAM(model, mobilenet_layer, device)
        cam_mn = gradcam_mn.generate_cam(input_tensor)
        overlay_mn = visualize_gradcam(orig_array, cam_mn, alpha=0.5)
        gradcam_results['MobileNetV2'] = overlay_mn
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM for MobileNetV2: {e}")
    
    # EfficientNet-B0 branch
    try:
        efficientnet_layer = model.efficientnet_reduce[0]
        gradcam_en = GradCAM(model, efficientnet_layer, device)
        cam_en = gradcam_en.generate_cam(input_tensor)
        overlay_en = visualize_gradcam(orig_array, cam_en, alpha=0.5)
        gradcam_results['EfficientNet-B0'] = overlay_en
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM for EfficientNet-B0: {e}")
    
    # Fused features (ASPP output)
    try:
        fused_layer = model.aspp.conv_reduction[0]
        gradcam_fused = GradCAM(model, fused_layer, device)
        cam_fused = gradcam_fused.generate_cam(input_tensor)
        overlay_fused = visualize_gradcam(orig_array, cam_fused, alpha=0.5)
        gradcam_results['Fused (ASPP)'] = overlay_fused
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM for fused features: {e}")
    
    return gradcam_results


def display_gradcam_results(gradcam_results, original_image):
    """Display Grad-CAM results."""
    st.markdown("### Explainability: Grad-CAM Heatmaps")
    st.markdown("""
    <div class="info-box">
        Grad-CAM highlights the regions of the image that the model focused on 
        when making its prediction. Red/yellow areas indicate higher importance.
    </div>
    """, unsafe_allow_html=True)
    
    # Display in columns
    cols = st.columns(len(gradcam_results) + 1)
    
    # Original image
    with cols[0]:
        st.markdown("**Original**")
        st.image(original_image.resize((224, 224)), use_container_width=True)
    
    # Grad-CAM overlays
    for idx, (branch_name, overlay) in enumerate(gradcam_results.items(), 1):
        with cols[idx]:
            st.markdown(f"**{branch_name}**")
            st.image(overlay, use_container_width=True)


# ============== Training Tab ==============

def training_tab():
    """Training interface tab."""
    st.markdown("## Model Training")
    
    st.markdown("""
    <div class="info-box">
        Train the E-MAS model on HAM10000, PH2, or Combined datasets.
        Training progress and metrics will be displayed in real-time.
    </div>
    """, unsafe_allow_html=True)
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset = st.selectbox(
            "Dataset",
            ["ham10000", "ph2", "combined"],
            help="Select dataset for training"
        )
    
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=40)
        batch_size = st.number_input("Batch Size", min_value=4, max_value=128, value=32)
    
    with col3:
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f"
        )
        split_mode = st.selectbox(
            "Split Mode", ["holdout", "5fold"],
            help="Holdout: 70/15/15 split | 5-fold: Cross-validation"
        )
    
    # Dataset paths
    st.markdown("### Dataset Paths")
    
    if dataset == "combined":
        col1, col2 = st.columns(2)
        with col1:
            ham_dir = st.text_input("HAM10000 Directory", value="data/ham10000")
            ham_csv = st.text_input("HAM10000 CSV Path", value="data/ham10000/HAM10000_metadata.csv")
        with col2:
            ph2_dir = st.text_input("PH2 Directory", value="data/ph2")
    else:
        data_dir = st.text_input("Dataset Directory", value=f"data/{dataset}")
        ham_csv = None
        if dataset == "ham10000":
            ham_csv = st.text_input("Metadata CSV (optional)", value="")
    
    # Start training button
    if st.button("Start Training", type="primary"):
        # Validate paths
        if dataset == "combined":
            data_dir_arg = f'"{{\\"ham10000\\": \\"{ham_dir}\\", \\"ph2\\": \\"{ph2_dir}\\"}}"'
        else:
            data_dir_arg = data_dir
        
        # Show training command
        cmd = f"""python train.py \\
    --dataset {dataset} \\
    --data-dir {data_dir_arg} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --split-mode {split_mode}"""
        
        if ham_csv:
            cmd += f" \\\n    --ham-csv {ham_csv}"
        
        st.code(cmd, language="bash")
        
        st.info("""
        Training is initiated via command line. For real-time training in the UI,
        please run the training script separately and return to evaluate the results.
        
        The trained model will be saved to the `checkpoints/` directory.
        """)
        
        # Display training curves placeholder
        st.markdown("### Training Progress")
        st.info("Run training via command line to see live progress here.")


# ============== Evaluation Tab ==============

def evaluation_tab():
    """Evaluation interface tab."""
    st.markdown("## Model Evaluation")
    
    st.markdown("""
    <div class="info-box">
        Evaluate a trained model on test data. Generates comprehensive metrics 
        including confusion matrix, ROC curves, and per-class performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Select checkpoint
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        st.warning("No checkpoints found in `checkpoints/` directory. Train a model first.")
        return
    
    selected_checkpoint = st.selectbox("Select Model Checkpoint", checkpoints)
    checkpoint_path = os.path.join("checkpoints", selected_checkpoint)
    
    # Dataset selection
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Dataset", ["ham10000", "ph2", "combined"])
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=32)
    
    # Dataset paths
    if dataset == "combined":
        col1, col2 = st.columns(2)
        with col1:
            ham_dir = st.text_input("HAM10000 Directory", value="data/ham10000", key="eval_ham")
            ham_csv = st.text_input("HAM10000 CSV", value="data/ham10000/HAM10000_metadata.csv", key="eval_csv")
        with col2:
            ph2_dir = st.text_input("PH2 Directory", value="data/ph2", key="eval_ph2")
    else:
        data_dir = st.text_input("Dataset Directory", value=f"data/{dataset}", key="eval_dir")
        ham_csv = st.text_input("Metadata CSV (optional)", value="", key="eval_csv_single") if dataset == "ham10000" else None
    
    # Evaluate button
    if st.button("Run Evaluation", type="primary"):
        # Build command
        if dataset == "combined":
            data_dir_arg = f'"{{\\"ham10000\\": \\"{ham_dir}\\", \\"ph2\\": \\"{ph2_dir}\\"}}"'
        else:
            data_dir_arg = data_dir
        
        cmd = f"""python evaluate.py \\
    --checkpoint {checkpoint_path} \\
    --dataset {dataset} \\
    --data-dir {data_dir_arg} \\
    --batch-size {batch_size}"""
        
        if ham_csv:
            cmd += f" \\\n    --ham-csv {ham_csv}"
        
        st.code(cmd, language="bash")
        
        st.info("""
        Evaluation is run via command line. Reports will be saved to the `reports/` directory
        including JSON metrics, CSV summary, confusion matrix, and ROC curves.
        """)
        
        # Show available reports
        st.markdown("### Available Reports")
        if os.path.exists("reports"):
            reports = [f for f in os.listdir("reports") if f.endswith(('.json', '.csv', '.png'))]
            if reports:
                st.write("Recent reports:")
                for report in sorted(reports, reverse=True)[:10]:
                    st.write(f"- {report}")
            else:
                st.info("No reports generated yet.")


# ============== About Model Tab ==============

def about_model_tab():
    """About model tab with architecture details."""
    st.markdown("## E-MAS Model Architecture")
    
    st.markdown("""
    <div class="info-box">
        <strong>E-MAS: Efficient Multi-Scale Attention System</strong> for 
        dermoscopic skin lesion classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture overview
    st.markdown("### Architecture Overview")
    
    arch_col1, arch_col2 = st.columns([2, 1])
    
    with arch_col1:
        st.markdown("""
        **Key Components:**
        
        1. **Dual Backbone Feature Extraction**
           - MobileNetV2: Efficient feature extraction with depthwise separable convolutions
           - EfficientNet-B0: Strong representational capability through compound scaling
        
        2. **Point-wise Feature Fusion**
           - Element-wise multiplication: F_fused = F_mobilenet ⊙ F_efficientnet
           - Amplifies co-activated regions, suppresses irrelevant features
        
        3. **Atrous Spatial Pyramid Pooling (ASPP)**
           - 1×1 convolution branch
           - 3×3 convolutions with dilation rates [6, 12, 18]
           - Global average pooling branch
           - Multi-scale contextual feature extraction
        
        4. **Squeeze-and-Excitation (SE) Attention**
           - Channel-wise adaptive recalibration
           - Prioritizes informative features
        
        5. **Classification Head**
           - Global Average Pooling
           - Two dense layers (256 → 128 → num_classes)
           - Softmax activation
        """)
    
    with arch_col2:
        st.markdown("### Model Specifications")
        
        specs_data = {
            'Parameter': [
                'Input Size',
                'Backbones',
                'Fusion Method',
                'ASPP Dilation Rates',
                'SE Reduction',
                'Dropout',
                'Optimizer',
                'Learning Rate',
                'Batch Size'
            ],
            'Value': [
                '224 × 224 × 3',
                'MobileNetV2 + EfficientNet-B0',
                'Point-wise Multiplication',
                '[6, 12, 18]',
                '16',
                '0.3',
                'Adam',
                '0.001',
                '32'
            ]
        }
        
        specs_df = pd.DataFrame(specs_data)
        st.dataframe(specs_df, hide_index=True, use_container_width=True)
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    perf_data = {
        'Dataset': ['HAM10000', 'PH2', 'Combined'],
        'Accuracy (%)': ['98.73%', '97.78%', '98.20%'],
        'AUC (%)': ['98.21%', '99.01%', '99.00%'],
        'Sensitivity (%)': ['95.12%', '98.80%', '96.40%'],
        'Precision (%)': ['94.37%', '97.10%', '97.10%'],
        'Specificity (%)': ['99.78%', '99.37%', '99.20%']
    }
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    
    # Class information
    st.markdown("### Supported Classes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**HAM10000 (7 classes):**")
        for code, name in HAM10000_CLASS_NAMES.items():
            st.write(f"- **{code}**: {name}")
    
    with col2:
        st.markdown("**PH2 (3 classes):**")
        for code, name in PH2_CLASS_NAMES.items():
            st.write(f"- **{code}**: {name}")
    
    # Citation
    st.markdown("### Citation")
    st.markdown("""
    ```
    @article{emas2024,
      title={E-MAS: An Efficient Multi-Scale Attention System for Dermoscopic 
             Image-Based Skin Cancer Classification},
      author={Aftab, Muhammad and Ali, Muhammad Mumtaz and Dong, Zigang and 
              Zhang, Chengjuan and Zhenfei, Wang and Jiang, Yanan and Liu, Kangdong},
      journal={},
      year={2024}
    }
    ```
    """)


# ============== Main Inference Tab ==============

def inference_tab():
    """Main inference interface."""
    st.markdown("## Dermoscopic Image Classification")
    
    # Model selection
    st.markdown("### Model Selection")
    
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        st.warning("""
        No trained checkpoints found in `checkpoints/` directory.
        
        Please train a model first using the **Train** tab, or place a pre-trained
        checkpoint in the `checkpoints/` directory.
        """)
        
        # Option to use default model
        if st.button("Initialize Untrained Model (for testing)"):
            with st.spinner("Initializing model..."):
                model = create_emas_model(num_classes=7, pretrained=True, device=st.session_state.device)
                st.session_state.model = model
                st.session_state.model_info = {
                    'num_classes': 7,
                    'dataset': 'ham10000',
                    'class_names': {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)},
                    'epoch': 0,
                    'val_acc': 0.0
                }
            st.success("Model initialized! Note: This is an untrained model.")
    else:
        selected_checkpoint = st.selectbox("Select Checkpoint", checkpoints)
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                checkpoint_path = os.path.join("checkpoints", selected_checkpoint)
                model, model_info = load_checkpoint(checkpoint_path)
                
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_info = model_info
                    st.success(f"Model loaded! Validation accuracy: {model_info['val_acc']:.4f}")
    
    # Image upload
    st.markdown("### Upload Dermoscopic Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a dermoscopic image (JPG or PNG format)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Uploaded Image**")
            st.image(image, use_container_width=True)
        
        with col2:
            # Check if model is loaded
            if st.session_state.model is None:
                st.warning("Please load a model first!")
            else:
                # Run inference
                if st.button("Run Classification", type="primary"):
                    with st.spinner("Analyzing image..."):
                        predictions, predicted_class, confidence = run_inference(
                            st.session_state.model,
                            image,
                            st.session_state.device
                        )
                    
                    # Display results
                    display_prediction_results(
                        predictions,
                        predicted_class,
                        confidence,
                        st.session_state.model_info
                    )
                    
                    # Grad-CAM visualization
                    st.markdown("---")
                    with st.spinner("Generating Grad-CAM visualizations..."):
                        gradcam_results = generate_gradcam_visualization(
                            st.session_state.model,
                            image,
                            st.session_state.model_info,
                            st.session_state.device
                        )
                    
                    display_gradcam_results(gradcam_results, image)


# ============== Main App ==============

def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏥 E-MAS Skin Lesion Classifier</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Efficient Multi-Scale Attention System for Dermoscopic Image Classification</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        **E-MAS** is a deep learning model for automated skin lesion classification
        from dermoscopic images.
        
        **Features:**
        - Dual backbone architecture (MobileNetV2 + EfficientNet-B0)
        - Multi-scale feature extraction with ASPP
        - Channel attention with SE blocks
        - Grad-CAM explainability
        """)
        
        st.markdown("---")
        
        st.markdown("### Device")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"Running on: **{device_name}**")
        
        st.markdown("---")
        
        # Checkpoints info
        st.markdown("### Checkpoints")
        checkpoints = get_available_checkpoints()
        st.write(f"Available: **{len(checkpoints)}**")
        
        # Reports info
        st.markdown("### Reports")
        if os.path.exists("reports"):
            reports = [f for f in os.listdir("reports") if f.endswith('.json')]
            st.write(f"Available: **{len(reports)}**")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Inference",
        "🚀 Train",
        "📊 Evaluate",
        "ℹ️ About Model"
    ])
    
    with tab1:
        inference_tab()
    
    with tab2:
        training_tab()
    
    with tab3:
        evaluation_tab()
    
    with tab4:
        about_model_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        E-MAS: Efficient Multi-Scale Attention System | 
        Research Paper Implementation
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
