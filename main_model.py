from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from keras_radam.training import RAdamOptimizer
from sklearn.utils import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# Training parameters
batch_size = 16
epochs = 10
num_classes = 4

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Conv2D.h5'
lrelu = tf.nn.leaky_relu

x_train = np.load('dataset/Train/Train_Capture_Image.npy')
y_train = np.load('dataset/Train/Label_Capture_Image.npy')
x_test = np.load('dataset/Validation/Validation_Capture_Image.npy')
y_test = np.load('dataset/Validation_Capture_Label.npy')

x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation(lrelu))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(lrelu))

    for i in range(5):
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(lrelu))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(lrelu))
        if i % 2 == 0:
            model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(lrelu))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation(lrelu))
    model.add(Dense(num_classes, activation='softmax'))

    model = multi_gpu_model(model, gpus=len(gpus), cpu_merge=False)

    model.compile(loss='categorical_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-4),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

hist_df = pd.DataFrame(history.history)

hist_csv_file = model_name[:-3] + '_history.csv'
with open('histories/' + hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
