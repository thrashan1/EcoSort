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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import cv2
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- GLOBALS ----------
IMG_SIZE = 160
CLASS_NAMES = ['Trash (N)', 'Compost (O)', 'Recyclable (R)']
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- BUILD MODEL ----------
def build_model(num_classes=3):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------- SCAN IMAGES ----------
def get_image_paths_and_labels(root_dir):
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subdirs.sort()
    label_map = {name: idx for idx, name in enumerate(subdirs)}
    data = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for class_name in subdirs:
        class_path = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(valid_ext):
                data.append((os.path.join(class_path, fname), label_map[class_name]))
    return data, label_map

# ---------- TF DATA PIPELINE ----------
def create_tf_dataset(file_label_pairs, batch_size=128, shuffle=True):
    paths = [p for p, l in file_label_pairs]
    labels = [l for p, l in file_label_pairs]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------- GUI APPLICATION ----------
class EcoSortApp:
    def __init__(self, root):
        self.root = root
        root.title("EcoSort – AI Waste Sorter")
        root.geometry("900x700")
        root.resizable(True, True)

        # Variables
        self.model = None
        self.training = False
        self.train_thread = None
        self.dataset_path = tk.StringVar(value="./DS1")
        self.subset_size = tk.IntVar(value=1000)
        self.epochs = tk.IntVar(value=5)
        self.batch_size = tk.IntVar(value=128)
        self.model_name = tk.StringVar(value="eco_sort_model.keras")
        self.status_var = tk.StringVar(value="Ready")

        self.build_ui()
        self.log("EcoSort GUI started.\n")

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT (controls)
        left_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        ttk.Label(left_frame, text="Dataset folder:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(left_frame, textvariable=self.dataset_path, width=30).grid(row=0, column=1, pady=2)
        ttk.Button(left_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5)

        ttk.Label(left_frame, text="Subset per class:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(left_frame, from_=100, to=10000, increment=100, textvariable=self.subset_size, width=10).grid(row=1, column=1, sticky='w')
        ttk.Label(left_frame, text="(0 = all)").grid(row=1, column=2, sticky='w')

        ttk.Label(left_frame, text="Epochs:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Spinbox(left_frame, from_=1, to=20, increment=1, textvariable=self.epochs, width=10).grid(row=2, column=1, sticky='w')

        ttk.Label(left_frame, text="Batch size:").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Spinbox(left_frame, from_=16, to=256, increment=16, textvariable=self.batch_size, width=10).grid(row=3, column=1, sticky='w')

        ttk.Label(left_frame, text="Model name:").grid(row=4, column=0, sticky='w', pady=2)
        ttk.Entry(left_frame, textvariable=self.model_name, width=20).grid(row=4, column=1, sticky='w')

        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Train on Dataset", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Live Webcam Training", command=self.start_live_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Classify Live", command=self.start_classify).pack(side=tk.LEFT, padx=5)

        save_frame = ttk.Frame(left_frame)
        save_frame.grid(row=6, column=0, columnspan=3, pady=5)
        ttk.Button(save_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)

        ttk.Label(left_frame, textvariable=self.status_var, foreground="blue").grid(row=7, column=0, columnspan=3, pady=10)

        # RIGHT (graph + log)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        graph_frame = ttk.LabelFrame(right_frame, text="Training Progress", padding=5)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(7, 3))
        self.fig.suptitle("Loss & Accuracy")
        self.ax1.set_title('Loss')
        self.ax2.set_title('Accuracy')
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='normal', wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ---------- UI helpers ----------
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select dataset folder (containing TRAIN/TEST)")
        if path:
            self.dataset_path.set(path)

    def update_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def plot_update(self, train_loss, train_acc, val_loss, val_acc):
        self.ax1.clear()
        self.ax1.plot(train_loss, 'b-o', label='Train Loss')
        if val_loss:
            self.ax1.plot(val_loss, 'r-o', label='Val Loss')
        self.ax1.legend(); self.ax1.grid(True)
        self.ax2.clear()
        self.ax2.plot(train_acc, 'b-o', label='Train Acc')
        if val_acc:
            self.ax2.plot(val_acc, 'r-o', label='Val Acc')
        self.ax2.legend(); self.ax2.grid(True)
        self.canvas.draw_idle()

    # ---------- DATASET TRAINING (background thread) ----------
    def start_training(self):
        if self.training:
            messagebox.showwarning("Busy", "Training is already running.")
            return
        self.training = True
        self.update_status("Training started...")
        self.log("=== Starting dataset training ===")
        self.train_thread = threading.Thread(target=self._train_dataset_thread, daemon=True)
        self.train_thread.start()

    def _train_dataset_thread(self):
        try:
            dataset_path = self.dataset_path.get().strip()
            if not os.path.exists(dataset_path):
                self.root.after(0, lambda: self.log(f"ERROR: Dataset path '{dataset_path}' not found."))
                self.training = False
                self.root.after(0, lambda: self.update_status("Error"))
                return

            train_path = os.path.join(dataset_path, "TRAIN")
            test_path = os.path.join(dataset_path, "TEST")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                self.root.after(0, lambda: self.log(f"ERROR: TRAIN/TEST not found in '{dataset_path}'."))
                self.training = False
                self.root.after(0, lambda: self.update_status("Error"))
                return

            self.root.after(0, lambda: self.log("Scanning images..."))
            train_pairs, _ = get_image_paths_and_labels(train_path)
            val_pairs, _ = get_image_paths_and_labels(test_path)
            self.root.after(0, lambda: self.log(f"Found {len(train_pairs)} training, {len(val_pairs)} validation images."))

            subset = self.subset_size.get()
            if subset > 0:
                from collections import defaultdict
                groups = defaultdict(list)
                for p, l in train_pairs:
                    groups[l].append((p, l))
                subset_train = []
                for lbl, items in groups.items():
                    subset_train.extend(items[:subset] if len(items) > subset else items)
                groups_val = defaultdict(list)
                for p, l in val_pairs:
                    groups_val[l].append((p, l))
                subset_val = []
                for lbl, items in groups_val.items():
                    subset_val.extend(items[:max(1, subset//2)] if len(items) > subset//2 else items)
                self.root.after(0, lambda: self.log(f"Using subset: {len(subset_train)} train, {len(subset_val)} val."))
                train_pairs, val_pairs = subset_train, subset_val

            batch_size = self.batch_size.get()
            epochs = self.epochs.get()

            train_ds = create_tf_dataset(train_pairs, batch_size=batch_size, shuffle=True)
            val_ds = create_tf_dataset(val_pairs, batch_size=batch_size, shuffle=False)

            steps_per_epoch = max(1, len(train_pairs) // batch_size)
            val_steps = max(1, len(val_pairs) // batch_size)

            self.model = build_model(3)
            self.root.after(0, lambda: self.log("Model built. Starting training..."))

            train_loss, train_acc = [], []
            val_loss, val_acc = [], []
            for epoch in range(epochs):
                if not self.training:
                    self.root.after(0, lambda: self.log("Training cancelled."))
                    break
                self.root.after(0, lambda e=epoch+1: self.update_status(f"Epoch {e}/{epochs}"))
                history = self.model.fit(
                    train_ds,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_ds,
                    validation_steps=val_steps,
                    epochs=1,
                    verbose=0
                )
                t_loss = history.history['loss'][0]
                t_acc = history.history['accuracy'][0]
                v_loss = history.history['val_loss'][0]
                v_acc = history.history['val_accuracy'][0]
                train_loss.append(t_loss); train_acc.append(t_acc)
                val_loss.append(v_loss); val_acc.append(v_acc)
                self.root.after(0, lambda e=epoch+1, tl=t_loss, ta=t_acc, vl=v_loss, va=v_acc:
                                self.log(f"Epoch {e}: Loss={tl:.4f}, Acc={ta:.4f}, ValLoss={vl:.4f}, ValAcc={va:.4f}"))
                self.root.after(0, lambda: self.plot_update(train_loss, train_acc, val_loss, val_acc))

            if self.training:
                self.root.after(0, lambda: self.log("Training completed."))
                self.root.after(0, lambda: self.update_status("Training done"))
                model_filename = self.model_name.get().strip() or "eco_sort_model.keras"
                model_path = os.path.join(MODEL_DIR, model_filename)
                self.model.save(model_path)
                self.root.after(0, lambda: self.log(f"Model saved to {model_path}"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"ERROR: {str(e)}"))
            import traceback
            self.root.after(0, lambda: self.log(traceback.format_exc()))
        finally:
            self.training = False
            self.root.after(0, lambda: self.update_status("Ready"))

    # ---------- LIVE WEBCAM TRAINING (fixed: runs in thread) ----------
    def start_live_training(self):
        if self.training:
            messagebox.showwarning("Busy", "Training is already running.")
            return
        self.training = True
        self.update_status("Live training...")
        self.log("=== Starting live webcam training ===")
        threading.Thread(target=self._live_training_thread, daemon=True).start()

    def _live_training_thread(self):
        try:
            print("\n📸 LIVE WEBCAM TRAINING MODE")
            print("   Press 1 (Trash), 2 (Compost), 3 (Recyclable) to set label.")
            print("   Press SPACE to capture that object.")
            print("   Press 't' to train the model on your captured images.")
            print("   Press 'q' to quit.\n")
            images = []
            labels_map = {ord('1'): 0, ord('2'): 1, ord('3'): 2}
            current_label = None
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: self.log("ERROR: Could not open webcam."))
                self.training = False
                return
            while self.training:
                ret, frame = cap.read()
                if not ret: break
                display = frame.copy()
                cv2.putText(display, "1:Trash  2:Compost  3:Recyclable", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                if current_label is not None:
                    cv2.putText(display, f"Label: {CLASS_NAMES[current_label]}", (10,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(display, f"Captured: {len(images)}", (10,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.imshow("Live Labelling", display)
                key = cv2.waitKey(1) & 0xFF
                if key in labels_map:
                    current_label = labels_map[key]
                    self.root.after(0, lambda lbl=current_label: self.log(f"Label set to {CLASS_NAMES[lbl]}"))
                elif key == 32 and current_label is not None:
                    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    images.append((resized, current_label))
                    self.root.after(0, lambda: self.log(f"✅ Captured {len(images)}"))
                elif key == ord('t'):
                    if len(images) >= 6:
                        break
                    else:
                        self.root.after(0, lambda: self.log(f"⚠️ Need at least 6 images (have {len(images)})"))
                elif key == ord('q'):
                    self.training = False
                    break
            cap.release()
            cv2.destroyAllWindows()
            if not self.training:
                self.root.after(0, lambda: self.log("Live training cancelled."))
                self.training = False
                self.root.after(0, lambda: self.update_status("Ready"))
                return

            if len(images) < 6:
                self.root.after(0, lambda: self.log("Not enough images. Aborting."))
                self.training = False
                self.root.after(0, lambda: self.update_status("Ready"))
                return

            X = np.array([img for img, lbl in images]) / 255.0
            y = np.array([lbl for img, lbl in images])
            from sklearn.utils import shuffle
            X, y = shuffle(X, y, random_state=42)

            model = build_model(3)
            self.root.after(0, lambda: self.log("Training on captured images..."))
            history = model.fit(X, y, epochs=8, batch_size=4, verbose=1)
            self.model = model
            self.root.after(0, lambda: self.log("Live training completed. Model ready."))
            self.root.after(0, lambda: self.plot_update(history.history['loss'], history.history['accuracy'], [], []))
            self.root.after(0, lambda: self.update_status("Live training done"))
            # Automatically start classification
            self.root.after(1000, self.start_classify)   # small delay
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error in live training: {str(e)}"))
        finally:
            self.training = False
            self.root.after(0, lambda: self.update_status("Ready"))

    # ---------- LIVE CLASSIFICATION (runs in thread) ----------
    def start_classify(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please train or load a model first.")
            return
        self.update_status("Classifying...")
        self.log("Starting live classification. Press 'q' in webcam window to quit.")
        threading.Thread(target=self._classification_thread, daemon=True).start()

    def _classification_thread(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: self.log("ERROR: Could not open webcam."))
                self.root.after(0, lambda: self.update_status("Ready"))
                return
            while True:
                ret, frame = cap.read()
                if not ret: break
                input_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) / 255.0
                input_batch = np.expand_dims(input_img, axis=0)
                preds = self.model.predict(input_batch, verbose=0)[0]
                class_id = np.argmax(preds)
                confidence = preds[class_id] * 100
                label = CLASS_NAMES[class_id]
                color = (0, 255, 0) if confidence > 70 else (0, 165, 255) if confidence > 40 else (0, 0, 255)
                cv2.putText(frame, f"{label}: {confidence:.1f}%", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                bar_w = int(confidence * 2)
                cv2.rectangle(frame, (10, 80), (10 + bar_w, 100), color, -1)
                cv2.imshow("EcoSort - Live Classification", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.root.after(0, lambda: self.log("Classification ended."))
            self.root.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Classification error: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Error"))

    # ---------- SAVE / LOAD ----------
    def save_model(self):
        if self.model is None:
            messagebox.showwarning("No Model", "No model to save.")
            return
        filename = self.model_name.get().strip() or "eco_sort_model.keras"
        path = os.path.join(MODEL_DIR, filename)
        self.model.save(path)
        self.log(f"Model saved to {path}")

    def load_model(self):
        filename = filedialog.askopenfilename(initialdir=MODEL_DIR, title="Select model file",
                                              filetypes=[("Keras models", "*.keras"), ("H5 models", "*.h5")])
        if filename:
            try:
                self.model = load_model(filename)
                self.log(f"Model loaded from {filename}")
                messagebox.showinfo("Success", "Model loaded successfully.")
            except Exception as e:
                self.log(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Could not load model: {str(e)}")

# ---------- MAIN ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = EcoSortApp(root)
    root.mainloop()
