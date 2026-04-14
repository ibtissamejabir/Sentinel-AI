import streamlit as st
import cv2
import onnxruntime as ort
import numpy as np
from collections import deque

st.set_page_config(page_title="AI Surveillance", layout="wide")
st.title("🛡️ Smart Surveillance System")

# 1. LOAD THE MODEL
# This adds the 'Green Box' you were looking for
@st.cache_resource
def load_model():
    # 1. Define paths clearly
    model_path = "/Users/Zhuanz/Desktop/fight detection/violence_model_v1.onnx"
    
    # 2. Tell ONNX exactly where to look for the .data file
    options = ort.SessionOptions()
    
    # This is the trick: it forces the model to look in the current folder
    return ort.InferenceSession(model_path, sess_options=options)

# Add this right after st.title to see if the files are visible
import os
st.sidebar.write("### Folder Check")
files = os.listdir(".")
if "/Users/Zhuanz/Desktop/fight detection/violence_model_v1.onnx.data" in files:
    st.sidebar.success("Found .data file")
else:
    st.sidebar.error("MISSING .data file in current folder!")
try:
    session = load_model()
    st.success("✅ AI Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model Error: {e}")
    st.stop()

# 2. SETUP THE CAMERA
run = st.checkbox('Start Live Intelligence Feed')
frame_placeholder = st.empty()
buffer = deque(maxlen=16) # This stores 16 frames for the AI to analyze

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret: break

        # 3. AI PRE-PROCESSING (Crucial for your ResNet/LSTM model)
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalization (Standard for Medical/Surveillance AI)
        img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        buffer.append(np.transpose(img, (2, 0, 1)))

        # 4. RUN DETECTION
        if len(buffer) == 16:
            input_data = np.expand_dims(np.array(buffer), axis=0).astype(np.float32)
            outputs = session.run(None, {session.get_inputs()[0].name: input_data})
            
            # Probability calculation
            probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
            fight_score = probs[1]

            # If the score is high, draw the alert
            if fight_score > 0.70:
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 25)
                cv2.putText(frame, f"VIOLENCE DETECTED: {fight_score:.2f}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        # Show the frame in the browser
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
    cap.release()
else:
    st.info("System is on Standby. Check the box above to start.")