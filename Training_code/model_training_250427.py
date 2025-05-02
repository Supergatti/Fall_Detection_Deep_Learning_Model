# e:\AIProject\Fall_Detection_Deep_Learning_Model\train.py
# 添加GPU支持
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# e:\AIProject\Fall_Detection_Deep_Learning_Model\Training_code\model_training_250427.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import io

# Define constants
URFALL_FALLS_CSV = "urfall-cam0-falls.csv"
URFALL_ADLS_CSV = "urfall-cam0-adls.csv"
MODEL_SAVE_PATH = "fall_detection_model.h5"

# Load and preprocess data
print("Loading data...")

# Load falls data
falls_df = pd.read_csv(URFALL_FALLS_CSV)
print(f"Falls data shape: {falls_df.shape}")

# Load ADLs data (Activities of Daily Living - non-falls)
# Alternative approach to handle '//' comments
with open(URFALL_ADLS_CSV, 'r') as f:
    lines = [line for line in f if not line.startswith('//')]

# Create a StringIO object with the filtered content
filtered_content = io.StringIO('\n'.join(lines))

# Read the CSV from the filtered content
adls_df = pd.read_csv(filtered_content, skiprows=1)
print(f"ADLs data shape: {adls_df.shape}")

# Extract features and labels
# Assuming 3rd column (index 2) is the label, and columns from 4th onward (index 3+) are features
falls_features = falls_df.iloc[:, 3:].values
falls_labels = falls_df.iloc[:, 2].values
adls_features = adls_df.iloc[:, 3:].values
adls_labels = adls_df.iloc[:, 2].values

# Map labels: For falls dataset, 1 or 0 -> 1 (fall), -1 -> 0 (non-fall)
falls_labels_mapped = np.where(np.isin(falls_labels, [1, 0]), 1, 0)

# Map labels: For ADLs dataset, all are -1 -> 0 (non-fall)
adls_labels_mapped = np.zeros_like(adls_labels)

# Combine data
X = np.vstack((falls_features, adls_features))
y = np.hstack((falls_labels_mapped, adls_labels_mapped))

# Convert features to float
X = X.astype('float32')

print(f"Combined data shape: X: {X.shape}, y: {y.shape}")
print(f"Class distribution: Falls (1): {np.sum(y == 1)}, Non-Falls (0): {np.sum(y == 0)}")

# Data balancing using upsampling
fall_indices = np.where(y == 1)[0]
non_fall_indices = np.where(y == 0)[0]

# Determine which class is the minority class
if len(fall_indices) < len(non_fall_indices):
    minority_indices = fall_indices
    majority_indices = non_fall_indices
    minority_class = 1
    majority_class = 0
else:
    minority_indices = non_fall_indices
    majority_indices = fall_indices
    minority_class = 0
    majority_class = 1

print(f"Upsampling minority class ({minority_class})...")
# Upsample minority class
minority_X = X[minority_indices]
minority_y = y[minority_indices]
majority_X = X[majority_indices]
majority_y = y[majority_indices]

# Resample the minority class
minority_X_upsampled, minority_y_upsampled = resample(
    minority_X, minority_y,
    replace=True,
    n_samples=len(majority_y),
    random_state=42
)

# Combine the upsampled minority class with the majority class
X_balanced = np.vstack((minority_X_upsampled, majority_X))
y_balanced = np.hstack((minority_y_upsampled, majority_y))

print(f"Balanced data shape: X: {X_balanced.shape}, y: {y_balanced.shape}")
print(f"Balanced class distribution: Falls (1): {np.sum(y_balanced == 1)}, Non-Falls (0): {np.sum(y_balanced == 0)}")

# Shuffle the data
indices = np.arange(X_balanced.shape[0])
np.random.shuffle(indices)
X_balanced = X_balanced[indices]
y_balanced = y_balanced[indices]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

print(f"Training data shape: X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing data shape: X: {X_test.shape}, y: {y_test.shape}")

# Get number of features
num_features = X_train.shape[1]

# Reshape data for LSTM model: [samples, time steps, features]
X_train_lstm = X_train.reshape(X_train.shape[0], 1, num_features)
X_test_lstm = X_test.reshape(X_test.shape[0], 1, num_features)

print(f"LSTM input shapes: X_train: {X_train_lstm.shape}, X_test: {X_test_lstm.shape}")

# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile model
print("Creating LSTM model...")
model = create_lstm_model((1, num_features))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

def lr_scheduler(epoch, lr):
    if epoch > 20 and epoch % 10 == 0:
        return lr * 0.9
    return lr

lr_schedule = LearningRateScheduler(lr_scheduler, verbose=1)

# Train model
print("Training model...")
history = model.fit(
    X_train_lstm, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_schedule],
    verbose=1
)

# Evaluate model
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test_lstm, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate predictions and classification report
y_pred_probs = model.predict(X_test_lstm)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")