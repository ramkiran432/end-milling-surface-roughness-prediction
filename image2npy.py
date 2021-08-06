import os
import cv2
import numpy as np
import random
import natsort
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

TRAIN_DIR = 'Capture'
train_folder_list = array(natsort.natsorted(os.listdir(TRAIN_DIR)))

train_input = []
train_label = []

label_encoder = LabelEncoder()  # Call LabelEncoder Class
integer_encoded = label_encoder.fit_transform(train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = natsort.natsorted(os.listdir(path))
    for img in img_list:
        img_path = os.path.join(path, img)
        src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_regulation = src[120:360, 200:440].copy()
        train_input.append([np.array(img_regulation)])
        train_label.append([np.array(onehot_encoded[index])])

train_label = np.reshape(train_label, (-1, len(integer_encoded)))
train_input = np.moveaxis(np.array(train_input), 1, -1).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
print(train_input.shape)
print(train_label.shape)
np.save('Train_Capture_Image.npy', train_input)
np.save('Label_Capture_Image.npy', train_label)
