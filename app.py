"""
app.py — Streamlit app for anomaly detection testing

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import json
import logging
import tempfile
from pathlib import Path
from PIL import Image
import io
import base64
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Anomaly Detection with PatchCore")
st.markdown("Upload an image to detect anomalies using the trained model")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_dir = st.text_input("Model directory", value="outputs")
    image_size = st.slider("Image size", 128, 512, 256, step=32)
    confidence_threshold = st.slider("Anomaly threshold", 0.0, 1.0, 0.5, step=0.05)

# Load model once (caching)
@st.cache_resource
def load_model(model_dir):
    """Load the trained PatchCore model"""
    try:
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
        
        model_path = Path(model_dir)
        
        # Find checkpoint
        ckpt_files = list(model_path.rglob("*.ckpt"))
        if not ckpt_files:
            st.error(f"❌ No .ckpt checkpoint found in {model_dir}")
            return None, None, None
        
        ckpt_path = str(ckpt_files[0])
        
        engine = Engine()
        model = Patchcore()
        
        st.success(f"✅ Model loaded from: {ckpt_path}")
        return engine, model, ckpt_path
        
    except ImportError:
        st.error("❌ anomalib not installed. Install with: pip install anomalib")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Load metrics if available
@st.cache_resource
def load_metrics(model_dir):
    """Load training metrics"""
    metrics_file = Path(model_dir) / "metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return {}

# Inference function
def predict_anomaly(image, engine, model, ckpt_path, image_size):
    """Run anomaly detection on image"""
    try:
        from anomalib.data import PredictDataset
        
        # Save image temporarily (cross-platform)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        image.save(tmp_path)
        
        # Create prediction dataset
        dataset = PredictDataset(path=str(tmp_path), image_size=(image_size, image_size))
        
        # Run inference
        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=ckpt_path,
        )
        
        # Extract results
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            
            # ImageBatch is an object with attributes, not a dict
            anomaly_score = float(pred.anomaly_score) if hasattr(pred, 'anomaly_score') else 0.0
            pred_label = int(pred.pred_label) if hasattr(pred, 'pred_label') else 0
            
            return anomaly_score, pred_label
        
        return None, None
        
    except Exception as e:
        st.error(f"❌ Inference error: {e}")
        log.exception(e)
        return None, None

# Main UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

with col2:
    st.subheader("📊 Model Info")
    metrics = load_metrics(model_dir)
    
    if metrics:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            if "accuracy" in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            if "precision" in metrics:
                st.metric("Precision", f"{metrics['precision']:.3f}")
        with metric_col2:
            if "recall" in metrics:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            if "f1_score" in metrics:
                st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    else:
        st.info("No metrics found. Train model first with: python train.py")

# Prediction
if uploaded_file:
    st.divider()
    st.subheader("🎯 Predictions")
    
    if st.button("🚀 Detect Anomaly", type="primary"):
        with st.spinner("Running inference..."):
            engine, model, ckpt_path = load_model(model_dir)
            
            if engine and model:
                anomaly_score, pred_label = predict_anomaly(
                    image, engine, model, ckpt_path, image_size
                )
                
                if anomaly_score is not None:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Anomaly Score", f"{anomaly_score:.3f}")
                    
                    with col2:
                        is_anomalous = anomaly_score > confidence_threshold
                        status = "🔴 ANOMALY" if is_anomalous else "🟢 NORMAL"
                        st.metric("Status", status)
                    
                    with col3:
                        confidence = abs(anomaly_score - confidence_threshold)
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    # Detailed results
                    st.divider()
                    st.subheader("📋 Detailed Results")
                    
                    results = {
                        "anomaly_score": float(anomaly_score),
                        "pred_label": int(pred_label),
                        "is_anomalous": bool(anomaly_score > confidence_threshold),
                        "threshold": float(confidence_threshold),
                        "image_size": image_size,
                    }
                    
                    st.json(results)
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(results, indent=2),
                        file_name="anomaly_result.json",
                        mime="application/json"
                    )

# Footer
st.divider()
st.markdown("""
### 📚 How to use:
1. **Train**: `python train.py --category bottle`
2. **Test with Streamlit**: `streamlit run app.py`
3. Upload an image and click "Detect Anomaly"

### 🔗 Other interfaces:
- **FastAPI**: `uvicorn server:app --reload --port 8080`
- **CLI**: `python score.py --input image.png --ckpt outputs/model.ckpt`
""")
