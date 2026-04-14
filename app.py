import streamlit as st
import cv2
import onnxruntime as ort
import numpy as np
from collections import deque
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")
st.title("🛡️ Smart Surveillance System")
st.markdown("### Real-time Violence Detection (RWF-2000 Model)")

# --- 1. DYNAMIC PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure these files are in /Users/Zhuanz/Desktop/Surveillance_Project/
model_path = os.path.join(current_dir, "/Users/Zhuanz/Desktop/Surveillance_Project/violence_model_v1.onnx")
data_path = os.path.join(current_dir, "/Users/Zhuanz/Desktop/Surveillance_Project/violence_model_v1.onnx.data")
# --- 2. SIDEBAR STATUS CHECK ---
st.sidebar.header("System Status")
if os.path.exists(model_path):
    st.sidebar.success("✅ Model File Found")
else:
    st.sidebar.error("⚠️ .onnx File Missing!")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Explicitly use CPU for stability on Mac
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

try:
    session = load_model()
    st.sidebar.success("✅ AI Brain Online")
except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.stop()

# --- 4. CAMERA & AI INFERENCE LOGIC ---
run = st.checkbox('Start Live Intelligence Feed')
frame_placeholder = st.empty()

# Initialize session states
if 'buffer' not in st.session_state:
    st.session_state.buffer = deque(maxlen=16)
if 'fight_score' not in st.session_state:
    st.session_state.fight_score = 0.0

if run:
    # Open camera (Index 0 is standard Mac Facetime camera)
    cap = cv2.VideoCapture(0)
    
    # Wait for hardware to initialize
    time.sleep(1.0) 
    
    if not cap.isOpened():
        st.error("Cannot access camera. Check Privacy Settings or close other apps using the camera.")
        st.stop()

    # Lower resolution = Higher Speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while run:
        ret, frame = cap.read()
        if not ret:
            # Instead of crashing, we wait a moment and try again
            time.sleep(0.1)
            continue
        
        frame_count += 1

        # AI Processing (Every 4th frame)
        if frame_count % 4 == 0:
            # Resize and Normalize
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            st.session_state.buffer.append(np.transpose(img, (2, 0, 1)))

            if len(st.session_state.buffer) == 16:
                # Prepare 5D input for the video model: (1, 16, 3, 224, 224)
                input_data = np.expand_dims(np.array(st.session_state.buffer), axis=0).astype(np.float32)
                
                # Inference
                outputs = session.run(None, {session.get_inputs()[0].name: input_data})
                
                # Probability calculation
                probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
                st.session_state.fight_score = probs[1]

        # --- VISUAL OVERLAY ---
        display_frame = frame.copy()
        if st.session_state.fight_score > 0.70:
            # Alert Mode (Red Box)
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 15)
            cv2.putText(display_frame, f"VIOLENCE DETECTED: {st.session_state.fight_score:.2f}", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Normal Mode (Green Text)
            cv2.putText(display_frame, f"Monitoring... ({st.session_state.fight_score:.2f})", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Use 'use_column_width' for older Streamlit or 'use_container_width' for newer
        frame_placeholder.image(display_frame, channels="BGR", use_column_width=True)
        
    cap.release()
else:
    st.info("System Standby. Please check the box to begin surveillance.")    