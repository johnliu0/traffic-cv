import augment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from keras.preprocessing import image

target_size = (150, 150)

x_train = []
num_augmented = 2

for i in range(12):
    imgs = augment.generate(f'light{i}.jpg', target_size=target_size, num_imgs=num_augmented)
    for j in range(num_augmented):
        x_train.append(imgs[j].flatten())

x_train = np.asarray(x_train)

svm = OneClassSVM(gamma='auto')
svm.fit(x_train)
