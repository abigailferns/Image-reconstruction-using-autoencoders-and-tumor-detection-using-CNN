import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Add, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

positive_dir = '/content/drive/MyDrive/Blurbrain/Pos'
negative_dir = '/content/drive/MyDrive/Blurbrain/Neg'
img_size = 256

def verify_path(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f" Path not found: {folder}")
    if len(os.listdir(folder)) == 0:
        raise FileNotFoundError(f" Folder is empty: {folder}")

def load_images_from_folder(folder):
    verify_path(folder)
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img_to_array(img) / 127.5 - 1)
    return images

pos_images = load_images_from_folder(positive_dir)
neg_images = load_images_from_folder(negative_dir)

img_data = np.array(pos_images + neg_images, dtype='float32')

x_train, x_test = train_test_split(img_data, test_size=0.2, random_state=42)

def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5 * ssim_loss + 0.5 * l1_loss


def build_encoder(input_img):
    filters = [64, 128, 256]
    x = input_img
    skip_connections = []
    for f in filters:
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.3)(x)
        skip_connections.append(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    return x, skip_connections


def build_decoder(encoded, skip_connections):
    filters = [256, 128, 64]
    x = encoded
    skip_connections.reverse()
    for i, f in enumerate(filters):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        if i < len(skip_connections):
            x = Add()([x, skip_connections[i]])
    return Conv2D(3, (3, 3), activation='tanh', padding='same')(x)


def build_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)
    encoded, skip_connections = build_encoder(input_img)
    decoded = build_decoder(encoded, skip_connections)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss=combined_loss)
    return autoencoder


class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        accuracy = (1 - val_loss) * 100
        print(f"Epoch {epoch+1}: Training Loss = {train_loss:.5f}, Validation Loss = {val_loss:.5f}, Accuracy = {accuracy:.2f}%")


autoencoder = build_autoencoder()
autoencoder.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
history = autoencoder.fit(
    x_train, x_train,
    epochs=400,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[reduce_lr, AccuracyCallback()]
)
autoencoder.save('/content/autoencoder4_brain_model.h5')

def plot_loss_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

plot_loss_curve(history)


def enhance_colors(image):
    image = np.uint8((image + 1) * 127.5)
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

def display_reconstructed_images(original, reconstructed, num_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow((original[i] + 1) / 2)
        plt.axis("off")
        if i == 0:
            plt.title("Original Images")

        plt.subplot(2, num_images, num_images + i + 1)
        enhanced_image = enhance_colors(reconstructed[i])
        plt.imshow(enhanced_image)
        plt.axis("off")
        if i == 0:
            plt.title("Color Enhanced Images")

    plt.show()

reconstructed_images = np.clip(autoencoder.predict(x_test), -1, 1)
display_reconstructed_images(x_test, reconstructed_images, num_images=5)
