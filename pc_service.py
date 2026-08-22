#!/usr/bin/env python3
"""
EcoSort standalone PC service.

Features:
- Dedicated background AI inference engine
- Safe main-thread OpenCV GUI event loop
- Instant keyboard ('C'/Space) and HTTP (/api/capture) trigger handling
- Direct sidecar .txt metadata logging and image snapshot saves
"""
import cv2
import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import json
import threading
import os
import sys
import webbrowser
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
import csv
import io
from queue import Queue, Full

IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'eco_sort_model.keras'
CONFIDENCE_THRESHOLD = 70
MIN_THRESHOLD = 30
MAX_THRESHOLD = 95
BAUD_RATE = 9600
SHOW_VIDEO_WINDOW = True
HTML_FILE = 'code_ver14.html'

CLASS_NAMES = ['Recycle', 'Compost', 'Trash']
CLASS_TO_CMD = {'Recycle':'R', 'Compost':'O', 'Organic':'O', 'Trash':'T', 'Garbage':'T'}

CATEGORY_WEIGHT_KG = {'recycle': 0.05, 'organic': 0.18, 'trash': 0.12, 'manual': 0.0}

STATS_FILE = BASE_DIR / 'pc_stats.json'
PHOTO_ROOT = BASE_DIR / 'employee_photos'
CAPTURE_DIR = PHOTO_ROOT / 'captures'
RETRAIN_DIR = PHOTO_ROOT / 'retraining'
PENDING_DIR = PHOTO_ROOT / 'pending'
ARCHIVE_DIRS = {c: PHOTO_ROOT / c for c in ('recycle', 'organic', 'trash', 'manual')}
QUEUE_FILE = PHOTO_ROOT / 'manual_queue.json'
RECENT_FILE = PHOTO_ROOT / 'recent_scans.json'
TELEMETRY_FILE = PHOTO_ROOT / 'telemetry.log'
CAPTURE_QUEUE_SIZE = 8

for d in [PHOTO_ROOT, CAPTURE_DIR, PENDING_DIR, RETRAIN_DIR, *ARCHIVE_DIRS.values()]:
    d.mkdir(parents=True, exist_ok=True)

stats = {'organic': 0, 'trash': 0, 'recycle': 0, 'manual': 0, 'total_kg': 0.0}
manual_queue = []
recent_scans = []
scan_id_counter = 0

latest_prediction = {'label': '—', 'confidence': 0, 'command': 'U'}
queue_lock = threading.Lock()
ser = None
CONFIDENCE_THRESHOLD = max(MIN_THRESHOLD, min(MAX_THRESHOLD, CONFIDENCE_THRESHOLD))

latest_frame = None
latest_jpeg = None
latest_frame_lock = threading.Lock()
telemetry_lock = threading.Lock()
serial_log = []
servo_state = 'Idle'
last_command = 'U'
last_inference_ms = 0.0
camera_fps = 0.0
camera_frame_count = 0
camera_fps_started = time.time()
inference_lock = threading.Lock()
capture_id_lock = threading.Lock()
capture_task_queue = Queue(maxsize=CAPTURE_QUEUE_SIZE)
capture_worker_thread = None
capture_worker_start_lock = threading.Lock()
capture_status_lock = threading.Lock()
capture_status = {
    'state': 'idle', 'request_id': 0, 'scan_id': None, 'error': '',
    'time': '', 'filename': '', 'metadata_filename': '',
    'directory': str(CAPTURE_DIR), 'queue_depth': 0
}
capture_request_id = 0

live_prediction = {'label': '—', 'confidence': 0.0, 'command': 'U', 'scores': {}, 'inference_ms': 0.0}
live_prediction_frame = None
prediction_lock = threading.Lock()

bin_fill = {'recycle': None, 'organic': None, 'trash': None, 'manual': None}
demo_mode = False
demo_thread = None
demo_stop = threading.Event()

def load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, value):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)

def recalc_weight():
    stats['total_kg'] = round(
        stats.get('recycle', 0) * CATEGORY_WEIGHT_KG['recycle'] +
        stats.get('organic', 0) * CATEGORY_WEIGHT_KG['organic'] +
        stats.get('trash', 0) * CATEGORY_WEIGHT_KG['trash'],
        2
    )

def save_stats():
    recalc_weight()
    save_json(STATS_FILE, stats)

def load_state():
    global stats, manual_queue, recent_scans, scan_id_counter
    stats = load_json(STATS_FILE, {'organic':0,'trash':0,'recycle':0,'manual':0,'total_kg':0.0})
    stats.setdefault('total_kg', 0.0)
    manual_queue = load_json(QUEUE_FILE, [])
    recent_scans = load_json(RECENT_FILE, [])
    scan_id_counter = max([int(x.get('id', 0)) for x in manual_queue + recent_scans] + [0])
    stats['manual'] = len(manual_queue)
    save_stats()

load_state()

if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found.")
    input("Press Enter to exit...")
    sys.exit(1)

model = keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No webcam found.")
    input("Press Enter to exit...")
    sys.exit(1)

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Is the Arduino plugged in?")
        return None
    print("\nAvailable Serial Ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device} - {port.description}")
    while True:
        try:
            choice = int(input("Enter Arduino port number: "))
            if 0 <= choice < len(ports):
                return ports[choice].device
        except ValueError:
            pass
        print("Invalid selection.")

def connect_serial():
    global ser
    port = find_arduino_port()
    if port is None:
        return False
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino connected on {port}")
        return True
    except Exception as exc:
        print(f"Could not connect to {port}: {exc}")
        return False

if not connect_serial():
    ser = None
    print('Arduino not connected. Running camera + Flask in keyboard/demo mode.')

def classify_frame(frame):
    """Run fast, thread-safe model inference directly via TensorFlow tensor call."""
    global last_inference_ms
    if frame is None or frame.size == 0:
        raise ValueError('Empty camera frame')
    started = time.perf_counter()
    
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    with inference_lock:
        tf_img = tf.convert_to_tensor(img, dtype=tf.float32)
        preds = model(tf_img, training=False).numpy()[0]
        
    last_inference_ms = (time.perf_counter() - started) * 1000.0
    
    class_scores = {}
    for idx, name in enumerate(CLASS_NAMES):
        class_scores[name] = round(float(preds[idx] * 100), 1) if idx < len(preds) else 0.0
        
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id] * 100)
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Manual'
    command = CLASS_TO_CMD.get(label, 'U') if confidence >= CONFIDENCE_THRESHOLD else 'U'
    final_label = label if confidence >= CONFIDENCE_THRESHOLD else 'Manual'
    return command, final_label, confidence, class_scores, last_inference_ms

def safe_filename(text):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(text))[:80]

def save_scan_metadata(image_path, scan_id, label, confidence, scores, command, source):
    """Save plain-text metadata sidecar file."""
    meta_path = image_path.with_suffix('.txt')
    lines = [
        f'ID: {scan_id}',
        f'Label: {label}',
        f'Final confidence: {float(confidence):.1f}%',
        f'Recycle confidence: {float((scores or {}).get("Recycle", 0)):.1f}%',
        f'Compost confidence: {float((scores or {}).get("Compost", 0)):.1f}%',
        f'Trash confidence: {float((scores or {}).get("Trash", 0)):.1f}%',
        f'Command: {command}',
        f'Source: {source}',
        f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}',
    ]
    meta_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return meta_path

def save_capture_snapshot(frame, label, confidence, scan_id, scores, command, source):
    """Save annotated JPEG snapshot."""
    filename = f'capture_{scan_id:06d}.jpg'
    path = CAPTURE_DIR / filename
    image = frame.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 92), (18, 24, 20), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)
    cv2.putText(image, f'EcoSort: {label}  {float(confidence):.1f}%', (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (245,245,245), 2)
    cv2.putText(image, f'Recycle {float(scores.get("Recycle",0)):.1f}%  Compost {float(scores.get("Compost",0)):.1f}%  Trash {float(scores.get("Trash",0)):.1f}%', (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215,230,220), 1)
    cv2.putText(image, f'{source} -> {command}', (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190,210,198), 1)
    
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]) or not path.exists():
        raise IOError(f'Could not save capture image: {path}')
    
    save_scan_metadata(path, scan_id, label, confidence, scores, command, source)
    return filename

def save_annotated_manual_photo(frame, label, confidence, scan_id, scores=None):
    img = frame.copy()
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 98), (20, 24, 22), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.putText(img, f"EcoSort: {label}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (245,245,245), 2)
    cv2.putText(img, f"AI confidence: {confidence:.1f}%", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210,225,215), 1)
    
    scores = scores or {}
    score_text = f"Recycle {float(scores.get('Recycle', 0)):.1f}%   Compost {float(scores.get('Compost', 0)):.1f}%   Trash {float(scores.get('Trash', 0)):.1f}%"
    cv2.putText(img, score_text, (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,235,225), 1)
    
    name = f"scan_{scan_id:06d}_manual_{confidence:.1f}.jpg"
    path = PENDING_DIR / name
    ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok or not path.exists():
        raise IOError(f'Could not save annotated scan image: {path}')
    return name

def add_recent(item, label, confidence, status, scan_id, scores=None, command=None, source=None, capture_filename=None):
    recent_scans.insert(0, {
        'id': scan_id,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'item': item,
        'tag': label,
        'confidence': round(float(confidence), 1),
        'scores': {k: round(float(v), 1) for k, v in (scores or {}).items()},
        'command': command or '',
        'source': source or '',
        'status': status,
        'capture_filename': capture_filename or ''
    })
    del recent_scans[80:]
    save_json(RECENT_FILE, recent_scans)

def add_manual_scan(frame, label, confidence, scan_id, scores=None, command='U', source='Camera scan'):
    filename = save_annotated_manual_photo(frame, label, confidence, scan_id, scores=scores)
    image_path = PENDING_DIR / filename
    save_scan_metadata(image_path, scan_id, label, confidence, scores or {}, command, source)
    item = {
        'id': scan_id,
        'filename': filename,
        'label': label,
        'confidence': round(float(confidence), 1),
        'scores': {k: round(float(v), 1) for k, v in (scores or {}).items()},
        'command': command,
        'source': source,
        'created': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with queue_lock:
        manual_queue.append(item)
        stats['manual'] = len(manual_queue)
        save_json(QUEUE_FILE, manual_queue)
        
    index_path = PHOTO_ROOT / 'photo_index.txt'
    with open(index_path, 'a', encoding='utf-8') as f:
        f.write(f'ID {scan_id} | {filename} | Final {float(confidence):.1f}% | Recycle {float((scores or {}).get("Recycle",0)):.1f}% | Compost {float((scores or {}).get("Compost",0)):.1f}% | Trash {float((scores or {}).get("Trash",0)):.1f}%\n')
    
    add_recent('Manual camera scan', label, confidence, 'Awaiting employee sort', scan_id, scores=scores, command=command, source=source, capture_filename=None)
    save_stats()
    return item

def log_serial(text):
    global serial_log
    with telemetry_lock:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        serial_log.insert(0, {'time': timestamp[11:], 'text': text})
        serial_log = serial_log[:60]
        with TELEMETRY_FILE.open('a', encoding='utf-8') as f:
            f.write(f'{timestamp} {text}\n')
        print(text)

def set_servo_state(state):
    global servo_state
    servo_state = state
    log_serial(f'[{state}]')

def parse_bin_message(line):
    parts = line.split(':')
    if len(parts) >= 3 and parts[0].upper() in ('FILL','ULTRA'):
        key = parts[1].strip().lower()
        try:
            value = max(0, min(100, float(parts[2])))
            if key in bin_fill:
                bin_fill[key] = round(value, 1)
                return True
        except ValueError:
            return False
    if len(parts) >= 3 and parts[0].upper() == 'DISTANCE':
        key = parts[1].strip().lower()
        try:
            distance_cm = float(parts[2])
            value = max(0, min(100, (35.0 - distance_cm) / 30.0 * 100.0))
            if key in bin_fill:
                bin_fill[key] = round(value, 1)
                return True
        except ValueError:
            return False
    return False

def camera_capture_loop():
    global latest_frame, latest_jpeg, camera_frame_count, camera_fps, camera_fps_started
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok_jpg, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        now = time.time()
        with latest_frame_lock:
            latest_frame = frame.copy()
            latest_jpeg = jpg.tobytes() if ok_jpg else None
            camera_frame_count += 1
            elapsed = now - camera_fps_started
            if elapsed >= 1.0:
                camera_fps = camera_frame_count / elapsed
                camera_frame_count = 0
                camera_fps_started = now
        time.sleep(0.005)

def get_current_frame():
    with latest_frame_lock:
        return latest_frame.copy() if latest_frame is not None else None

def continuous_inference_loop():
    """Independent background loop ensuring AI predictions are ALWAYS live."""
    global live_prediction, live_prediction_frame
    while True:
        frame = get_current_frame()
        if frame is not None:
            try:
                command, label, conf, scores, inf_ms = classify_frame(frame)
                with prediction_lock:
                    live_prediction.update({
                        'label': label, 'confidence': conf, 'command': command,
                        'scores': scores, 'inference_ms': inf_ms
                    })
                    live_prediction_frame = frame.copy()
            except Exception as exc:
                log_serial(f'LIVE INFERENCE ERROR: {type(exc).__name__}: {exc}')
        time.sleep(0.12)  # ~8 inferences/sec

def commit_live_prediction(source='CAPTURE', send_to_arduino=True, frame=None, prediction_snapshot=None):
    """Snapshot current prediction values and execute complete capture log."""
    global latest_prediction, scan_id_counter, last_command, last_inference_ms
    try:
        prediction = dict(prediction_snapshot or live_prediction)
        if frame is None:
            frame = get_current_frame()

        if frame is None or frame.size == 0:
            log_serial(f'{source}: camera frame unavailable')
            return False

        label = str(prediction.get('label') or 'Manual')
        conf = float(prediction.get('confidence') or 0.0)
        scores = dict(prediction.get('scores') or {})
        inf_ms = float(prediction.get('inference_ms') or 0.0)

        if label in ('—', '', None):
            log_serial(f'{source}: live model initialising, try again in 1s...')
            return False

        if label == 'Manual' or conf < CONFIDENCE_THRESHOLD:
            label = 'Manual'
            command = 'U'
        else:
            command = CLASS_TO_CMD.get(label, 'U')

        prediction_to_save = {
            'label': label, 'confidence': conf, 'command': command,
            'scores': scores, 'inference_ms': inf_ms
        }

        with prediction_lock:
            latest_prediction = dict(prediction_to_save)

        if send_to_arduino and ser and ser.is_open:
            try:
                ser.write(command.encode('ascii'))
                ser.flush()
                log_serial(f'PC -> Arduino: {command}')
            except Exception as exc:
                log_serial(f'ARDUINO WRITE ERROR: {type(exc).__name__}: {exc}')

        last_command = command
        last_inference_ms = inf_ms
        with capture_id_lock:
            scan_id_counter += 1
            sid = scan_id_counter

        capture_filename = save_capture_snapshot(frame, label, conf, sid, scores, command, source)
        
        destination = {'R':'Recycle','O':'Organic','T':'Trash','U':'Manual / Hold'}.get(command, 'Unknown')
        set_servo_state(destination)
        
        log_serial(f'{source} COMMITTED -> {label} ({conf:.1f}%) | R {float(scores.get("Recycle",0)):.1f}% | C {float(scores.get("Compost",0)):.1f}% | T {float(scores.get("Trash",0)):.1f}%')

        if label == 'Manual':
            item = add_manual_scan(frame, label, conf, sid, scores=scores, command=command, source=source)
            log_serial(f'MANUAL QUEUE: scan #{item["id"]} saved as {item["filename"]}')
        else:
            if command == 'O': stats['organic'] += 1
            elif command == 'T': stats['trash'] += 1
            elif command == 'R': stats['recycle'] += 1
            add_recent('Camera scan', label, conf, 'Processed', sid, scores=scores, command=command, source=source, capture_filename=capture_filename)
        
        save_stats()
        return True
    
    except Exception as exc:
        log_serial(f'{source}: COMMIT ERROR {type(exc).__name__}: {exc}')
        return False

def set_capture_status(state, **values):
    with capture_status_lock:
        capture_status.update({'state': state, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, **values)
        capture_status['queue_depth'] = capture_task_queue.qsize()

def get_capture_status():
    with capture_status_lock:
        status = dict(capture_status)
    status['queue_depth'] = capture_task_queue.qsize()
    return status

def capture_worker():
    log_serial('Capture worker running')
    while True:
        task = capture_task_queue.get()
        try:
            log_serial(f"Capture worker dequeued request {task['request_id']}")
            set_capture_status('saving', request_id=task['request_id'], error='')
            if commit_live_prediction(task['source'], task['send_to_arduino'], task['frame'], task['prediction']):
                set_capture_status(
                    'committed', request_id=task['request_id'], scan_id=scan_id_counter,
                    filename=f'capture_{scan_id_counter:06d}.jpg',
                    metadata_filename=f'capture_{scan_id_counter:06d}.txt', error=''
                )
            else:
                set_capture_status('failed', request_id=task['request_id'], error='Capture commit returned false')
        except Exception as exc:
            set_capture_status('failed', request_id=task['request_id'], error=f'{type(exc).__name__}: {exc}')
            log_serial(f"{task['source']}: CAPTURE WORKER ERROR {type(exc).__name__}: {exc}")
        finally:
            capture_task_queue.task_done()

def start_capture_worker():
    global capture_worker_thread
    with capture_worker_start_lock:
        if capture_worker_thread is None or not capture_worker_thread.is_alive():
            capture_worker_thread = threading.Thread(target=capture_worker, daemon=True, name='capture-worker')
            capture_worker_thread.start()
            log_serial('Capture worker start requested')

def request_capture(source='CAPTURE', send_to_arduino=True, frame=None):
    """Snapshot the live frame and prediction, then enqueue persistence."""
    global capture_request_id
    if frame is None:
        frame = get_current_frame()
    if frame is None or frame.size == 0:
        log_serial(f'{source}: camera frame unavailable')
        return False
    with prediction_lock:
        prediction_snapshot = dict(live_prediction)
    if prediction_snapshot.get('label') in ('—', '', None):
        log_serial(f'{source}: live model initialising, try again in 1s...')
        return False
    with capture_id_lock:
        capture_request_id += 1
        request_id = capture_request_id
    start_capture_worker()
    try:
        capture_task_queue.put_nowait({
            'request_id': request_id, 'source': source,
            'send_to_arduino': send_to_arduino, 'frame': frame.copy(),
            'prediction': prediction_snapshot
        })
    except Full:
        set_capture_status('failed', request_id=request_id, error='Capture queue is full')
        log_serial(f'{source}: capture queue is full')
        return False
    set_capture_status('queued', request_id=request_id, error='')
    log_serial(f'{source}: capture request {request_id} queued')
    return True

def serial_listener():
    while True:
        try:
            if not ser or not ser.is_open:
                time.sleep(0.25)
                continue
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue
                log_serial(f"Arduino: {line}")
                if parse_bin_message(line):
                    continue
                upper = line.upper()
                if upper.startswith('SERVO:') or upper.startswith('SORTED:'):
                    set_servo_state(line.split(':', 1)[1].strip())
                    continue
                if line == 'CAPTURE':
                    log_serial('Arduino CAPTURE detected')
                    request_capture('Arduino CAPTURE', send_to_arduino=True)
                elif upper.startswith('FULL:'):
                    bin_name = line.split(':', 1)[1].strip().lower()
                    if bin_name in bin_fill:
                        bin_fill[bin_name] = 100.0
        except Exception as exc:
            print(f'Serial error: {exc}')
            time.sleep(1)
        time.sleep(0.01)

def run_main_gui_loop():
    """OpenCV display window running on Main Thread."""
    if not SHOW_VIDEO_WINDOW:
        while True:
            time.sleep(1)
            
    window = 'EcoSort Camera Preview'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Live AI preview active: SPACE/C = scan, D = simulation, Q = quit.")
    
    while True:
        frame = get_current_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        preview = frame.copy()
        with prediction_lock:
            pred = dict(live_prediction)
            
        label = pred.get('label', '—')
        conf = float(pred.get('confidence', 0.0) or 0.0)
        inf_ms = float(pred.get('inference_ms', 0.0) or 0.0)
        scores = pred.get('scores') or {}

        overlay = preview.copy()
        cv2.rectangle(overlay, (8, 8), (preview.shape[1]-8, 105), (15, 20, 17), -1)
        cv2.addWeighted(overlay, 0.72, preview, 0.28, 0, preview)
        cv2.putText(preview, f'LIVE: {label}  {conf:.1f}%', (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 110), 2)
        cv2.putText(preview, f'Inference: {inf_ms:.0f} ms   Threshold: {CONFIDENCE_THRESHOLD:.0f}%', (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 230, 225), 1)
        cv2.putText(preview, f'R {float(scores.get("Recycle",0)):.1f}%   C {float(scores.get("Compost",0)):.1f}%   T {float(scores.get("Trash",0)):.1f}%', (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 205, 198), 1)
        cv2.putText(preview, 'SPACE/C: scan   D: demo   Q: quit', (24, preview.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 235, 230), 1)

        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (32, ord('c'), ord('C')):
            request_capture(source='Keyboard CAPTURE', send_to_arduino=True, frame=frame)
        elif key in (ord('d'), ord('D')):
            toggle_demo_local()
            
    cv2.destroyAllWindows()


app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path='')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.get('/')
def dashboard():
    return send_from_directory(BASE_DIR, HTML_FILE) if (BASE_DIR / HTML_FILE).exists() else Response('EcoSort HTML file not found.', status=404)

@app.get('/api/stats')
def get_stats():
    stats['manual'] = len(manual_queue)
    recalc_weight()
    return jsonify(stats)

@app.get('/api/manual-queue')
def get_manual_queue():
    with queue_lock:
        return jsonify(manual_queue)

@app.get('/api/photo/<path:filename>')
def get_photo(filename):
    safe = os.path.basename(filename)
    for folder in [PENDING_DIR, RETRAIN_DIR, CAPTURE_DIR, *ARCHIVE_DIRS.values()]:
        if (folder / safe).exists():
            return send_from_directory(folder, safe)
    return Response('Not found', status=404)

@app.get('/api/photo-meta/<path:filename>')
def get_photo_meta(filename):
    safe = os.path.basename(filename)
    base = Path(safe).with_suffix('.txt').name
    for folder in [PENDING_DIR, RETRAIN_DIR, CAPTURE_DIR, *ARCHIVE_DIRS.values()]:
        path = folder / base
        if path.exists():
            return send_from_directory(folder, base, mimetype='text/plain')
    return Response('Not found', status=404)

@app.get('/api/recent-scans')
def get_recent_scans():
    return jsonify(recent_scans)

@app.get('/api/mjpeg')
def mjpeg():
    def generate():
        while True:
            with latest_frame_lock:
                jpg = latest_jpeg
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.06)
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get('/api/telemetry')
def telemetry():
    with prediction_lock:
        prediction = dict(latest_prediction)
        live = dict(live_prediction)
    with telemetry_lock:
        logs = list(serial_log[:25])
    return jsonify({
        'serial_connected': bool(ser and ser.is_open), 'port': getattr(ser, 'port', None), 
        'baud': BAUD_RATE, 'camera_fps': round(camera_fps,1), 'latency_ms': round(last_inference_ms,1), 
        'inference_ms': round(last_inference_ms,1), 'prediction': prediction, 'live_prediction': live, 
        'servo_state': servo_state, 'last_command': last_command, 'serial_log': logs, 'bin_fill': bin_fill, 
        'threshold': CONFIDENCE_THRESHOLD, 'demo_mode': demo_mode,
        'scan_in_progress': get_capture_status()['state'] in ('queued', 'saving'),
        'capture': get_capture_status()
    })

@app.post('/api/settings/threshold')
def update_threshold():
    global CONFIDENCE_THRESHOLD
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get('threshold', CONFIDENCE_THRESHOLD))
        CONFIDENCE_THRESHOLD = max(MIN_THRESHOLD, min(MAX_THRESHOLD, value))
        log_serial(f'Threshold set to {CONFIDENCE_THRESHOLD:.0f}%')
        return jsonify({'threshold': CONFIDENCE_THRESHOLD})
    except (TypeError, ValueError):
        return jsonify({'error':'Invalid threshold'}), 400

def _archive_photo(item, folder):
    old_path = PENDING_DIR / item['filename']
    folder.mkdir(parents=True, exist_ok=True)
    new_name = f"scan_{int(item['id']):06d}_{safe_filename(folder.name)}_{float(item['confidence']):.1f}.jpg"
    new_path = folder / new_name
    if old_path.exists():
        os.replace(old_path, new_path)
    old_meta = old_path.with_suffix('.txt')
    if old_meta.exists():
        os.replace(old_meta, new_path.with_suffix('.txt'))
    item['filename'] = new_name
    return item

@app.post('/api/manual/resolve')
def resolve_manual():
    data = request.get_json(silent=True) or {}
    scan_id = int(data.get('id', 0)); category = str(data.get('category', '')).lower()
    if category not in ('recycle','compost','trash','manual'): return jsonify({'error':'Invalid category'}), 400
    with queue_lock:
        item = next((x for x in manual_queue if int(x['id']) == scan_id), None)
        if item is None: return jsonify({'error':'Scan not found'}), 404
        manual_queue[:] = [x for x in manual_queue if int(x['id']) != scan_id]
        stats['manual'] = len(manual_queue)
        folder = ARCHIVE_DIRS['organic' if category == 'compost' else category]
        item = _archive_photo(item, folder)
        save_json(QUEUE_FILE, manual_queue)
    if category == 'recycle': stats['recycle'] += 1
    elif category == 'compost': stats['organic'] += 1
    elif category == 'trash': stats['trash'] += 1
    cmd = {'recycle':'R','compost':'O','trash':'T','manual':'U'}[category]
    try:
        if ser and ser.is_open: ser.write(cmd.encode()); set_servo_state({'R':'Recycle','O':'Organic','T':'Trash','U':'Manual / Hold'}[cmd])
    except Exception: pass
    add_recent(f'Employee sorted scan #{scan_id}', category.title(), item['confidence'], 'Employee sorted', scan_id)
    save_stats(); return jsonify({'status':'ok','item':item,'stats':stats})

@app.post('/api/manual/retrain')
def submit_retraining():
    data = request.get_json(silent=True) or {}; scan_id = int(data.get('id', 0))
    with queue_lock:
        item = next((x for x in manual_queue if int(x['id']) == scan_id), None)
        if item is None: return jsonify({'error':'Scan not found'}), 404
        manual_queue[:] = [x for x in manual_queue if int(x['id']) != scan_id]
        stats['manual'] = len(manual_queue)
        item = _archive_photo(item, RETRAIN_DIR)
        item['submitted_for_retraining'] = True
        save_json(QUEUE_FILE, manual_queue)
    add_recent(f'Retraining submission #{scan_id}', item['label'], item['confidence'], 'Training dataset', scan_id)
    save_stats(); return jsonify({'status':'ok','item':item})

def export_records():
    rows = list(recent_scans)
    for row in rows:
        yield row

@app.get('/api/export.json')
def export_json():
    return jsonify(list(export_records()))

@app.get('/api/export.csv')
def export_csv():
    output = io.StringIO(); rows = list(export_records()); fields = ['id','time','item','tag','confidence','status']
    writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
    for row in rows: writer.writerow({k: row.get(k,'') for k in fields})
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=ecosort_audit_log.csv'})

def toggle_demo_local():
    global demo_mode, demo_thread
    demo_mode = not demo_mode
    if demo_mode and (demo_thread is None or not demo_thread.is_alive()):
        demo_stop.clear()
        demo_thread = threading.Thread(target=demo_loop, daemon=True)
        demo_thread.start()
        log_serial('Simulation mode enabled')
    elif not demo_mode:
        demo_stop.set()
        log_serial('Simulation mode disabled')
    return demo_mode

@app.get('/api/demo/status')
def demo_status():
    return jsonify({'enabled': demo_mode})

def process_demo_scan():
    global scan_id_counter, latest_prediction
    if not demo_mode: return
    scan_id_counter += 1
    names = ['Recycle','Compost','Trash']
    label = names[(scan_id_counter-1) % 3]
    conf = round(84 + ((scan_id_counter * 7) % 13), 1)
    command = CLASS_TO_CMD[label]
    scores = {x: 0.0 for x in CLASS_NAMES}; scores[label] = conf
    scores[names[(scan_id_counter) % 3]] = round((100-conf)*0.75,1); scores[names[(scan_id_counter+1)%3]] = round(100 - scores[label] - scores[names[(scan_id_counter)%3]],1)
    
    with prediction_lock: 
        latest_prediction = {'label':label,'confidence':conf,'command':command,'scores':scores,'inference_ms':last_inference_ms}
    
    log_serial(f'SIM:CAPTURE -> {label} {conf:.1f}%'); set_servo_state(f'Demo {label}')
    
    if label == 'Recycle': stats['recycle'] += 1; bin_fill['recycle'] = min(100.0, float(bin_fill['recycle'] or 0) + 4.0)
    elif label == 'Compost': stats['organic'] += 1; bin_fill['organic'] = min(100.0, float(bin_fill['organic'] or 0) + 4.0)
    else: stats['trash'] += 1; bin_fill['trash'] = min(100.0, float(bin_fill['trash'] or 0) + 4.0)
    
    add_recent('Demo conveyor scan', label, conf, 'Simulation', scan_id_counter); save_stats()

def demo_loop():
    while not demo_stop.is_set():
        if demo_mode: process_demo_scan()
        demo_stop.wait(4.0)

@app.post('/api/demo/toggle')
def toggle_demo():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    if enabled != demo_mode:
        toggle_demo_local()
    return jsonify({'enabled': demo_mode})

@app.post('/api/capture')
def api_capture():
    frame = get_current_frame()
    if frame is None:
        return jsonify({'ok': False, 'error': 'Camera frame unavailable'}), 503

    if not request_capture('Dashboard CAPTURE', send_to_arduino=True, frame=frame):
        return jsonify({'ok': False, 'error': 'Live prediction is not ready yet'}), 409

    return jsonify({'ok': True, 'status': 'captured'}), 202

@app.post('/api/reset')
def reset_stats():
    global stats, manual_queue, recent_scans, scan_id_counter
    with queue_lock:
        for item in manual_queue:
            (PENDING_DIR / item['filename']).unlink(missing_ok=True)
        manual_queue = []; stats = {'organic':0,'trash':0,'recycle':0,'manual':0,'total_kg':0.0}; recent_scans=[]; scan_id_counter=0
        save_json(QUEUE_FILE, manual_queue); save_json(RECENT_FILE, recent_scans); save_stats()
    return jsonify({'status':'reset','stats':stats})

@app.get('/api/health')
def health_check():
    return jsonify({'status':'online','model_loaded':True,'camera_ready':cap.isOpened(),'serial_connected':bool(ser and ser.is_open),'arduino_optional':True,'port':getattr(ser,'port',None),'baud':BAUD_RATE,'camera_fps':round(camera_fps,1),'latency_ms':round(last_inference_ms,1),'threshold':CONFIDENCE_THRESHOLD,'demo_mode':demo_mode})

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000/')

if __name__ == '__main__':
    print('=' * 55)
    print('ECOSORT - PC STANDALONE')
    print('=' * 55)
    
    # 1. Start Camera Hardware Stream
    threading.Thread(target=camera_capture_loop, daemon=True, name='camera-capture').start()
    
    # 2. Start Background AI Engine
    threading.Thread(target=continuous_inference_loop, daemon=True, name='ai-inference').start()

    # 3. Start capture persistence before accepting any trigger
    start_capture_worker()

    # 4. Start Serial Hardware Listener
    threading.Thread(target=serial_listener, daemon=True, name='serial-listener').start()
    
    # 5. Start Flask Web Server
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    # 6. Launch Browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    print('System ready at http://localhost:5000/')
    
    # 7. Run GUI Event Loop directly on Main Thread
    run_main_gui_loop()
