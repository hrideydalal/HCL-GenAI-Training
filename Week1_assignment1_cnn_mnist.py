# ============================================================
# Assignment 1 - CNN for Image Classification using MNIST
# ============================================================

# =========================
# 1. Import Libraries
# =========================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# =========================
# 2. Load Dataset
# =========================
# MNIST dataset contains handwritten digit images (0 to 9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension for CNN
# Shape changes from (28, 28) to (28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# =========================
# 3. Build CNN Model
# =========================
model = models.Sequential([
    # First Convolution Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the 2D feature maps into 1D
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation='relu'),

    # Dropout helps reduce overfitting
    layers.Dropout(0.5),

    # Output Layer for 10 classes (digits 0-9)
    layers.Dense(10, activation='softmax')
])

# =========================
# 4. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model architecture
model.summary()

# =========================
# 5. Train Model
# =========================
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# =========================
# 6. Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\nTraining Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
print("Test Accuracy:", test_acc)

# =========================
# 7. Predictions and Confusion Matrix
# =========================
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 8. Classification Report
# =========================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 9. Accuracy Graph
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# =========================
# 10. Loss Graph
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ============================================================
# 11. Assignment Questions - Answers
# ============================================================

# 1. Why are convolution layers better than fully connected layers for images?
# Convolution layers are better for images because they detect local patterns
# such as edges, textures, and shapes. They also use fewer parameters than
# fully connected layers, which makes them more efficient and helps preserve
# image structure.

# 2. What is the role of pooling?
# Pooling reduces the size of the feature maps. This decreases computation,
# speeds up training, and keeps the most important features while removing
# unnecessary details.

# 3. What happens if you increase the number of filters?
# Increasing the number of filters allows the model to learn more features
# from the image. This may improve accuracy, but it also increases training
# time and may lead to overfitting if the model becomes too complex.

# 4. How does dropout help?
# Dropout helps prevent overfitting by randomly turning off some neurons
# during training. This forces the model to learn more general patterns
# instead of depending too much on specific neurons.
