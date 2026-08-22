import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import cv2
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- GLOBALS ----------
IMG_SIZE = 224

# CRITICAL: This exact mapping is required to match pc_service.py:
# Index 0: Recycle, Index 1: Compost, Index 2: Trash
CLASS_MAPPING = {'R': 0, 'O': 1, 'N': 2}
CLASS_NAMES = {0: 'Recycle (R)', 1: 'Organic (O)', 2: 'Trash (N)'}

# ---------- TF DATA PIPELINE & MODEL ----------
def build_model():
    # pc_service.py scales pixel values to [0, 1]. We match that input shape here.
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data Augmentation prevents overfitting on small datasets (100-200 images)
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)
    x = RandomZoom(0.1)(x)
    
    # MobileNetV2 internally expects inputs in [-1, 1].
    # This rescaling layer converts pc_service's [0, 1] inputs seamlessly into [-1, 1].
    x = Rescaling(scale=2.0, offset=-1.0)(x)
    
    base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base.trainable = False  # Freeze base model
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.35)(x)
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_dataset(folder_path):
    images = []
    labels = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_ext):
                path = os.path.join(class_dir, fname)
                img = cv2.imread(path) # Read as BGR to exactly match pc_service webcam frames
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Scale to [0, 1] exactly as pc_service.py does
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(class_id)
                
    return np.array(images), np.array(labels)

# ---------- KERAS CALLBACK FOR GUI UPDATES ----------
class LiveUpdateCallback(Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Log to UI
        msg = f"Epoch {epoch+1}: Loss={logs.get('loss'):.4f}, Acc={logs.get('accuracy'):.4f}"
        if 'val_loss' in logs:
            msg += f" | ValLoss={logs.get('val_loss'):.4f}, ValAcc={logs.get('val_accuracy'):.4f}"
        self.app.root.after(0, lambda: self.app.log(msg))
        
        # Update Graphs Live[cite: 3]
        self.app.root.after(0, lambda: self.app.plot_update(
            self.train_loss, self.train_acc, self.val_loss, self.val_acc))

# ---------- GUI APPLICATION ----------
class DatasetTrainerApp:
    def __init__(self, root):
        self.root = root
        root.title("EcoSort – Dataset Training Engine")
        root.geometry("900x700")
        
        self.training = False
        self.model = None
        
        # Variables
        self.dataset_path = tk.StringVar(value="./TRAIN")
        self.epochs = tk.IntVar(value=15)
        self.batch_size = tk.IntVar(value=16) # Smaller batch is better for ~150 images
        self.model_name = tk.StringVar(value="eco_sort_model.keras")
        self.status_var = tk.StringVar(value="Ready")

        self.build_ui()
        self.log("EcoSort Dataset Trainer initialized.\nEnsure your folder has N, O, R subfolders.")

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT (Controls)[cite: 3]
        left_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        ttk.Label(left_frame, text="Dataset Folder (e.g. TRAIN):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(left_frame, textvariable=self.dataset_path, width=25).grid(row=1, column=0, sticky='w')
        ttk.Button(left_frame, text="Browse", command=self.browse_dataset).grid(row=1, column=1, padx=5)

        ttk.Label(left_frame, text="Epochs:").grid(row=2, column=0, sticky='w', pady=(10, 0))
        ttk.Spinbox(left_frame, from_=1, to=100, increment=1, textvariable=self.epochs, width=10).grid(row=3, column=0, sticky='w')

        ttk.Label(left_frame, text="Batch Size:").grid(row=4, column=0, sticky='w', pady=(10, 0))
        ttk.Spinbox(left_frame, from_=4, to=64, increment=4, textvariable=self.batch_size, width=10).grid(row=5, column=0, sticky='w')

        ttk.Label(left_frame, text="Save Model As:").grid(row=6, column=0, sticky='w', pady=(10, 0))
        ttk.Entry(left_frame, textvariable=self.model_name, width=25).grid(row=7, column=0, sticky='w')

        ttk.Button(left_frame, text="🚀 Start Training", command=self.start_training).grid(row=8, column=0, columnspan=2, pady=20, ipadx=20, ipady=5)
        ttk.Label(left_frame, textvariable=self.status_var, foreground="blue", font=("Helvetica", 10, "bold")).grid(row=9, column=0, columnspan=2, pady=5)

        # RIGHT (Graph + Log)[cite: 3]
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        graph_frame = ttk.LabelFrame(right_frame, text="Live Training Metrics", padding=5)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(7, 3))
        self.fig.suptitle("Loss & Accuracy Tracking")
        self.ax1.set_title('Model Loss')
        self.ax2.set_title('Model Accuracy')
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='normal', wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select folder containing N, O, R subfolders")
        if path:
            self.dataset_path.set(path)

    def plot_update(self, train_loss, train_acc, val_loss, val_acc):
        self.ax1.clear()
        self.ax1.plot(train_loss, 'b-o', label='Train Loss')
        if val_loss and len(val_loss) == len(train_loss):
            self.ax1.plot(val_loss, 'r-o', label='Val Loss')
        self.ax1.legend(); self.ax1.grid(True)
        self.ax1.set_title('Model Loss')

        self.ax2.clear()
        self.ax2.plot(train_acc, 'b-o', label='Train Acc')
        if val_acc and len(val_acc) == len(train_acc):
            self.ax2.plot(val_acc, 'r-o', label='Val Acc')
        self.ax2.legend(); self.ax2.grid(True)
        self.ax2.set_title('Model Accuracy')
        self.canvas.draw_idle()

    def start_training(self):
        if self.training:
            messagebox.showwarning("Busy", "Training is already in progress.")
            return
        self.training = True
        self.status_var.set("Loading Data...")
        self.log("\n=== Starting Model Training ===")
        threading.Thread(target=self._training_thread, daemon=True).start()

    def _training_thread(self):
        try:
            folder = self.dataset_path.get().strip()
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder '{folder}' does not exist.")

            self.root.after(0, lambda: self.log(f"Scanning images in: {folder}"))
            X, y = load_dataset(folder)
            
            if len(X) == 0:
                raise ValueError("No valid images found. Ensure R, O, and N folders exist inside the target directory and contain images.")

            from sklearn.utils import shuffle
            X, y = shuffle(X, y, random_state=42)
            
            # Simple 80/20 validation split
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            self.root.after(0, lambda: self.log(f"Found {len(X)} total images."))
            self.root.after(0, lambda: self.log(f"Split: {len(X_train)} Training | {len(X_val)} Validation"))

            self.model = build_model()
            
            epochs = self.epochs.get()
            batch_size = self.batch_size.get()

            self.root.after(0, lambda: self.status_var.set("Training..."))
            
            # Use custom callback to live-update the GUI
            live_updater = LiveUpdateCallback(self)
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[live_updater],
                verbose=0 # Output handled by our callback instead
            )

            # Save the trained model
            save_name = self.model_name.get().strip()
            if not save_name.endswith('.keras'):
                save_name += '.keras'
            
            self.model.save(save_name)
            self.root.after(0, lambda: self.log(f"\n✅ SUCCESS! Model saved locally as: {save_name}"))
            self.root.after(0, lambda: self.status_var.set("Training Complete"))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"ERROR: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.training = False

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetTrainerApp(root)
    root.mainloop()
