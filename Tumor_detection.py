from google.colab import drive, files
drive.mount('/content/drive')
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
train_dir = '/content/drive/MyDrive/Image Reconstruction using Deep
Learning/NewData/Training' test_dir = '/content/drive/MyDrive/Image Reconstruction using Deep
Learning/NewData/Testing' save_path = '/content/drive/MyDrive/Image Reconstruction using Deep Learning/models' os.makedirs(save_path, exist_ok=True)
datagen = ImageDataGenerator(
rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.2, brightness_range=[0.8, 1.2], horizontal_flip=True, validation_split=0.3
)
train_data = datagen.flow_from_directory(
train_dir, target_size=(240, 240), batch_size=16, class_mode='categorical', subset='training', shuffle=True
)
valid_data = datagen.flow_from_directory(
train_dir, target_size=(240, 240), batch_size=16, class_mode='categorical', subset='validation', shuffle=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
test_dir, target_size=(240, 240), class_mode='categorical', shuffle=False
)
print("Class indices:", train_data.class_indices)
effnet = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
x = effnet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(4, activation='softmax')(x)
model = Model(inputs=effnet.input, outputs=x)
model.compile(
optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy']
)
checkpoint = ModelCheckpoint(
os.path.join(save_path, 'best_model1.h5'), monitor='val_accuracy', save_best_only=True, verbose=1
)
earlystop = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, verbose=1)
steps_per_epoch = len(train_data)
validation_steps = len(valid_data)
history = model.fit(
train_data, epochs=15, validation_data=valid_data, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[checkpoint, earlystop, reduce_lr], verbose=2
)
final_model_path = os.path.join(save_path, 'final_model1.h5')
model.save(final_model_path)
plt.figure(figsize=(18, 7))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()
print("Train Eval:", model.evaluate(train_data, verbose=0))
print("Test Eval:", model.evaluate(test_data, verbose=0))
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data, verbose=0), axis=1)
cm = confusion_matrix(y_true, y_pred)
labels = list(train_data.class_indices.keys())
def plot_confusion(cm, labels):
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plot_confusion(cm, labels)
files.download(final_model_path)
