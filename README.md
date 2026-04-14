#  Sentinel-AI: Real-Time Violence Detection System

A deep learning-based surveillance application that monitors live video feeds to detect violent activity in real-time. Built with **Python**, **ONNX Runtime**, and **Streamlit**.

## Overview
Traditional surveillance requires constant human monitoring. **Sentinel-AI** automates this process by using a temporal-aware AI model to analyze sequences of movement, identifying "fight" signatures with high confidence.

### Key Features:
* **Live Webcam Integration:** Real-time processing of standard camera feeds.
* **Temporal Analysis:** Processes 16-frame sequences to understand motion (not just single images).
* **Edge-Optimized:** Uses **ONNX Runtime** for high-performance inference on CPU/Consumer hardware.
* **Instant Alerts:** Visual UI changes (Red/Green status) based on probability thresholds.

##  Tech Stack
* **Language:** Python 3.x
* **UI Framework:** Streamlit
* **Inference Engine:** ONNX Runtime
* **Computer Vision:** OpenCV
* **Model Architecture:** RWF-2000 (ResNet-LSTM Hybrid)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ibtissamejabir/Sentinel-AI.git](https://github.com/ibtissamejabir/Sentinel-AI.git)
   cd Sentinel-AI
