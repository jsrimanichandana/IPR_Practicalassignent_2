# Step 1: Download the Dataset
!wget -r -N -c -np https://physionet.org/files/eegmat/1.0.0/

# Step 2: Install Necessary Libraries
!pip install pyEDFlib pywavelets tensorflow scikit-learn

# Step 3: Import Libraries
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import glob
from pyedflib import highlevel

# Step 4: Load and Preprocess EEG Data
def load_eeg_data_with_labels(directory, csv_path, target_length=30000):
    eeg_data = []
    task_labels = []       # Label: 0 for relaxed, 1 for stressed
    group_labels = []      # Group: 0 for Group B, 1 for Group G

    # Load the CSV file to get subject information and group labels
    subject_info = pd.read_csv(csv_path)

    # Map Subject IDs to group quality labels (0 - Group B, 1 - Group G)
    subject_id_to_group = dict(zip(subject_info['Subject'], subject_info['Count quality']))

    # Locate all .edf files in the specified directory
    edf_files = glob.glob(directory + '/*.edf')

    for file in edf_files:
        # Load the EEG data from the file
        signals, _, _ = highlevel.read_edf(file)
        selected_signals = np.array(signals[:19])  # Use only the first 19 channels

        # Process each channel's data to have consistent length
        processed_signals = []
        for channel in selected_signals:
            if len(channel) > target_length:
                # Truncate if longer
                channel = channel[:target_length]
            elif len(channel) < target_length:
                # Pad if shorter
                channel = np.pad(channel, (0, target_length - len(channel)), 'constant')
            processed_signals.append(channel)

        # Append the processed signals to eeg_data
        eeg_data.append(np.array(processed_signals))

        # Determine task label based on filename suffix
        base_filename = file.split('/')[-1]
        
        # Extract subject ID from filename (assuming format "SubjectID_1.edf" or "SubjectID_2.edf")
        subject_id = "Subject" + base_filename.split('_')[0][-2:]  # Extracts ID like "Subject00"

        # Determine the task label: 0 for baseline (relaxed) and 1 for task (stressed)
        if "_1" in base_filename:
            task_label = 0  # baseline/relaxed
        elif "_2" in base_filename:
            task_label = 1  # task/stressed

        # Append task label
        task_labels.append(task_label)

        # Get the group label based on subject ID (0 for Group B, 1 for Group G)
        group_label = subject_id_to_group.get(subject_id, None)
        if group_label is not None:
            group_labels.append(group_label)

    # Convert lists to NumPy arrays
    eeg_data = np.array(eeg_data)
    task_labels = np.array(task_labels)
    group_labels = np.array(group_labels)
    
    return eeg_data, task_labels, group_labels

# Define the directory containing EEG .edf files and path to the subject-info CSV file
directory = '/content/physionet.org/files/eegmat/1.0.0'
csv_path = '/content/physionet.org/files/eegmat/1.0.0/subject-info.csv'

# Load EEG data with task and group labels
eeg_data, task_labels, group_labels = load_eeg_data_with_labels(directory, csv_path)

# Step 5: Apply Discrete Wavelet Transform (DWT)
def apply_dwt(data):
    dwt_data = []
    for signal in data:
        coeffs = [pywt.wavedec(ch, 'db8', level=4) for ch in signal]
        # Concatenate each channel's coefficients for all levels
        processed_signal = np.hstack([np.hstack(c) for c in coeffs])
        dwt_data.append(processed_signal)
    return np.array(dwt_data)

# Apply DWT to EEG data
dwt_data = apply_dwt(eeg_data)

# Reshape DWT output to be compatible with 3D input requirements (samples, timesteps, features)
# Here, timesteps = number of original EEG channels (19) and features = flattened coefficients
dwt_data = dwt_data.reshape(dwt_data.shape[0], 19, -1)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(dwt_data, task_labels, test_size=0.3, random_state=42)

# Define CNN-BLSTM Model
def create_cnn_blstm_model(input_shape):
    model = Sequential()
    # CNN layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # BLSTM layers for sequence learning - No Flatten here
    model.add(Bidirectional(LSTM(64, return_sequences=True)))  # BLSTM requires 3D input
    model.add(Dropout(0.5))
    model.add(Flatten())  # Flatten after LSTM to prepare for Dense layer
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Stress vs. Relax)
    return model


# Create and compile the model
input_shape = (X_train.shape[1], X_train.shape[2])  # Set input shape as (timesteps, features)
model = create_cnn_blstm_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2)

# Step 10: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", cm)

# Step 11: ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 9: Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2)

# Step 10: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", cm)

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 5))

# Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Step 11: ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

from tensorflow.keras.layers import BatchNormalization, LeakyReLU

# Define CNN-BLSTM Model with Batch Normalization and Dropout
def create_improved_cnn_blstm_model(input_shape):
    model = Sequential()
    # CNN layers for feature extraction
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # BLSTM layer for sequence learning
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))

    # Flatten and output layer
    model.add(Flatten())
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    return model

# Create and compile the model with improved architecture
input_shape = (X_train.shape[1], X_train.shape[2])  # Set input shape as (timesteps, features)
model = create_improved_cnn_blstm_model(input_shape)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the Model with Class Weights
class_weights = {0: 1., 1: 2.}  # Assuming class 1 is the minority class; adjust weights as needed
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, class_weight=class_weights)

# Step 10: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", cm)

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 5))

# Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Step 11: ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
