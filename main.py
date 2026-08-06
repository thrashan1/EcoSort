import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import sys

# ---------- PHASE 1: CAPTURE ----------
images = []
labels_map = {ord('1'): 0, ord('2'): 1, ord('3'): 2}
class_names = ['Recyclable', 'Compost', 'Trash']
current_label = None

cap = cv2.VideoCapture(0)
print("Press 1,2,3 to set label, SPACE to capture, 't' to train, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    display = frame.copy()
    cv2.putText(display, "1:Recyclable  2:Compost  3:Trash", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if current_label is not None:
        cv2.putText(display, f"Label: {class_names[current_label]}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(display, f"Captured: {len(images)}", (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.imshow("Live Labelling", display)

    key = cv2.waitKey(1) & 0xFF
    if key in labels_map:
        current_label = labels_map[key]
    elif key == 32 and current_label is not None:
        resized = cv2.resize(frame, (224, 224))
        images.append((resized, current_label))
        print(f"Captured {len(images)}")
    elif key == ord('t'):
        if len(images) >= 6:   # minimum for demo
            break
        else:
            print(f"Need at least 6 images (have {len(images)})")
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

cap.release()
cv2.destroyAllWindows()

if len(images) < 6:
    print("Not enough images. Exiting.")
    sys.exit()

# ---------- PHASE 2: TRAIN ----------
X = np.array([img for img, lbl in images]) / 255.0
y = np.array([lbl for img, lbl in images])
X, y = shuffle(X, y, random_state=42)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n=== TRAINING STARTED ===")
for epoch in range(5):
    print(f"Epoch {epoch+1}/5")
    model.fit(X, y, epochs=1, batch_size=4, verbose=1)
print("=== TRAINING COMPLETE ===\n")

# ---------- PHASE 3: CLASSIFY LIVE ----------
cap = cv2.VideoCapture(0)
print("Live classification running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    input_img = cv2.resize(frame, (224, 224)) / 255.0
    input_batch = np.expand_dims(input_img, axis=0)
    preds = model.predict(input_batch, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100

    label = class_names[class_id]
    color = (0,255,0) if confidence > 70 else (0,165,255) if confidence > 40 else (0,0,255)
    cv2.putText(frame, f"{label}: {confidence:.1f}%", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # confidence bar
    bar_width = int(confidence * 2)
    cv2.rectangle(frame, (10,80), (10+bar_width, 100), color, -1)

    cv2.imshow("Live Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
