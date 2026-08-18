#!/usr/bin/env python3
"""
=====================================================
ECOSORT - STANDALONE PC VERSION (WITH LIVE CAMERA PREVIEW)
=====================================================
- Runs entirely on your laptop (no Raspberry Pi needed).
- Connects to Arduino over USB Serial (COM port).
- Uses your laptop's webcam for AI inference.
- Hosts the live dashboard (Flask) for the website.
- Sends sorting commands: 'O'(Organic), 'T'(Trash), 'R'(Recycle), 'U'(Manual).
- Auto-detects Arduino COM port.
- Opens dashboard in browser automatically.
- Shows a live OpenCV window with AI overlays.
"""

import cv2
import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
from tensorflow import keras  # <-- IMPORTANT: Uses the fixed import
import time
import json
import threading
import os
import sys
import webbrowser
from flask import Flask, jsonify, request

# ============================================================
# 1. CONFIGURATION
# ============================================================
IMG_SIZE = 224  # <-- Matches your model's input shape
MODEL_PATH = 'eco_sort_model.keras'      # Your trained model file
CONFIDENCE_THRESHOLD = 70                # % below which -> Manual ('U')
BAUD_RATE = 9600
SHOW_VIDEO_WINDOW = True                # Set False to disable the OpenCV window

# Map AI class names to Arduino commands
CLASS_NAMES = ['Recycle', 'Compost', 'Trash']
CLASS_TO_CMD = {
    'Recycle':   'R',
    'Compost':   'O',
    'Organic':   'O',
    'Trash':     'T',
    'Garbage':   'T',
}

# Stats tracking
stats = {'organic': 0, 'trash': 0, 'recycle': 0, 'manual': 0}
STATS_FILE = 'pc_stats.json'

# Global variables for video display
latest_prediction = {'label': '—', 'confidence': 0, 'command': 'U'}
frame_lock = threading.Lock()

# ============================================================
# 2. STATS MANAGEMENT
# ============================================================
def load_stats():
    global stats
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
        print(f"📊 Loaded stats: {stats}")
    except:
        stats = {'organic': 0, 'trash': 0, 'recycle': 0, 'manual': 0}
        save_stats()

def save_stats():
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

load_stats()

# ============================================================
# 3. LOAD AI MODEL
# ============================================================
print("🧠 Loading AI model...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file '{MODEL_PATH}' not found.")
    print("   Please train one first using main.py or gui_main.py")
    input("Press Enter to exit...")
    sys.exit(1)

model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# ============================================================
# 4. WEBCAM SETUP
# ============================================================
print("📸 Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found. Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No webcam found. Please connect a camera.")
        print("   If you have a camera, try changing the index in the code.")
        input("Press Enter to exit...")
        sys.exit(1)
print("✅ Webcam ready.")

# ============================================================
# 5. FIND ARDUINO (Auto-detect COM port)
# ============================================================
def find_arduino_port():
    """Lists all serial ports and lets the user select the Arduino."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("❌ No serial ports found. Is your Arduino plugged in?")
        return None

    print("\n🔌 Available Serial Ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device} - {port.description}")

    print("\n👉 Enter the number of your Arduino port (e.g., 0, 1, 2):")
    try:
        choice = int(input("> "))
        if 0 <= choice < len(ports):
            return ports[choice].device
        else:
            print("❌ Invalid selection.")
            return None
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None

def connect_serial():
    """Attempts to connect to the selected Arduino port."""
    global ser
    while True:
        port = find_arduino_port()
        if port is None:
            return False

        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"✅ Arduino connected on {port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to {port}: {e}")
            retry = input("Try another port? (y/n): ").lower()
            if retry != 'y':
                return False

ser = None
if not connect_serial():
    print("❌ Could not connect to Arduino. Exiting.")
    sys.exit(1)

# ============================================================
# 6. CLASSIFICATION FUNCTION (shared)
# ============================================================
def classify_frame(frame):
    """Returns (command, label, confidence)"""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100

    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Manual'

    if confidence < CONFIDENCE_THRESHOLD:
        label = 'Manual'
        command = 'U'
    else:
        command = CLASS_TO_CMD.get(label, 'U')

    return command, label, confidence

# ============================================================
# 7. SERIAL LISTENER THREAD (reads "CAPTURE" & "FULL" messages)
# ============================================================
def serial_listener():
    """Listens for Arduino messages and handles them."""
    print("🔊 Serial listener started. Waiting for 'CAPTURE' from Arduino...")
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()

                if line == "CAPTURE":
                    # --- Take photo and classify ---
                    print("\n📸 Capture triggered by Arduino.")
                    ret, frame = cap.read()
                    if ret:
                        command, label, conf = classify_frame(frame)
                        ser.write(command.encode())
                        print(f"📤 Sent: '{command}' -> {label} ({conf:.1f}%)")

                        # Update stats
                        if label == 'Manual':
                            stats['manual'] += 1
                        elif command == 'O':
                            stats['organic'] += 1
                        elif command == 'T':
                            stats['trash'] += 1
                        elif command == 'R':
                            stats['recycle'] += 1
                        save_stats()

                        # Update global prediction for video display
                        with frame_lock:
                            global latest_prediction
                            latest_prediction = {'label': label, 'confidence': conf, 'command': command}
                    else:
                        print("❌ Failed to capture frame. Sending 'U'.")
                        ser.write(b'U')

                elif line.startswith("FULL:"):
                    # Bin full notification
                    _, bin_name = line.split(":")
                    print(f"⚠️ BIN FULL: {bin_name}")

                else:
                    # Print any other Arduino debug messages
                    print(f"Arduino: {line}")

        except Exception as e:
            print(f"Serial error: {e}")
            time.sleep(1)
        time.sleep(0.01)

# ============================================================
# 8. VIDEO DISPLAY THREAD (shows live camera with AI overlays)
# ============================================================
def video_display():
    """Continuously shows camera feed with AI predictions overlaid."""
    if not SHOW_VIDEO_WINDOW:
        return

    print("🖥️ Starting video display window (press 'q' to close).")
    cv2.namedWindow('EcoSort AI Preview', cv2.WINDOW_NORMAL)

    # For continuous classification (so the window shows live guesses even without Arduino trigger)
    # but we'll use the shared prediction from the serial listener to avoid re-classifying.
    # However, we want the video to update even if no CAPTURE happens, so we'll run classification in this thread.
    # We'll do it in a loop.

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Run classification on the current frame for live preview
        command, label, conf = classify_frame(frame)

        # Draw overlay
        h, w = frame.shape[:2]

        # Draw a semi-transparent rectangle at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Display label and confidence
        text = f"{label}  {conf:.1f}%"
        color = (0, 255, 0) if conf > 70 else (0, 165, 255) if conf > 40 else (0, 0, 255)
        cv2.putText(frame, text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw confidence bar
        bar_width = int((w - 60) * (conf / 100))
        cv2.rectangle(frame, (30, 60), (30 + bar_width, 70), color, -1)

        # Show the frame
        cv2.imshow('EcoSort AI Preview', frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("🖥️ Video window closed.")

# ============================================================
# 9. FLASK API SERVER (Dashboard backend)
# ============================================================
app = Flask(__name__)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    global stats
    stats = {'organic': 0, 'trash': 0, 'recycle': 0, 'manual': 0}
    save_stats()
    print("🔄 Stats reset by admin.")
    return jsonify({"status": "reset", "stats": stats})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "camera_ready": cap.isOpened(),
        "serial_connected": ser.is_open
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ============================================================
# 10. OPEN DASHBOARD AUTOMATICALLY (Optional)
# ============================================================
def open_browser():
    """Opens the dashboard in your default browser after 2 seconds."""
    time.sleep(2)
    webbrowser.open('http://localhost:5000/api/stats')
    print("\n🌐 Dashboard opened in your browser.")
    print("   (If it didn't open, go to http://localhost:5000/api/stats)")

# ============================================================
# 11. MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("="*55)
    print("          🚀 ECOSORT - PC STANDALONE (with Camera Preview)")
    print("="*55)
    print(f"📊 Stats file: {STATS_FILE}")
    print(f"🔌 Arduino: {ser.port} @ {BAUD_RATE} baud")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📸 Camera: {'OK' if cap.isOpened() else 'FAIL'}")
    print("="*55)
    print("   Arduino commands: 'O'(Organic), 'T'(Trash), 'R'(Recycle), 'U'(Idle)")
    print("   Press 'q' in the video window to close it.")
    print("="*55)

    # Start serial listener in background
    t_serial = threading.Thread(target=serial_listener, daemon=True)
    t_serial.start()

    # Start video display in background (if enabled)
    if SHOW_VIDEO_WINDOW:
        t_video = threading.Thread(target=video_display, daemon=True)
        t_video.start()

    # Open browser automatically (optional)
    t_browser = threading.Thread(target=open_browser, daemon=True)
    t_browser.start()

    print("\n✅ System ready. Place items on the conveyor belt.")
    print("📊 Dashboard: http://localhost:5000/api/stats")
    print("   Press Ctrl+C to stop.\n")

    # Run Flask (blocks main thread)
    run_flask()